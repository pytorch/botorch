# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional, Union

import torch
from botorch.acquisition import AcquisitionFunction
from botorch.optim.homotopy import Homotopy
from botorch.optim.optimize import optimize_acqf
from torch import Tensor


def prune_candidates(
    candidates: Tensor, acq_values: Tensor, prune_tolerance: float
) -> Tensor:
    r"""Prune candidates based on their distance to other candidates.

    Args:
        candidates: An `n x d` tensor of candidates.
        acq_values: An `n` tensor of candidate values.
        prune_tolerance: The minimum distance to prune candidates.

    Returns:
        An `m x d` tensor of pruned candidates.
    """
    if candidates.ndim != 2:
        raise ValueError("`candidates` must be of size `n x d`.")
    if acq_values.ndim != 1 or len(acq_values) != candidates.shape[0]:
        raise ValueError("`acq_values` must be of size `n`.")
    if prune_tolerance < 0:
        raise ValueError("`prune_tolerance` must be >= 0.")
    sorted_inds = acq_values.argsort(descending=True)
    candidates = candidates[sorted_inds]

    candidates_new = candidates[:1, :]
    for i in range(1, candidates.shape[0]):
        if (
            torch.cdist(candidates[i : i + 1, :], candidates_new).min()
            > prune_tolerance
        ):
            candidates_new = torch.cat(
                [candidates_new, candidates[i : i + 1, :]], dim=-2
            )
    return candidates_new


def optimize_acqf_homotopy(
    acq_function: AcquisitionFunction,
    bounds: Tensor,
    q: int,
    homotopy: Homotopy,
    num_restarts: int,
    raw_samples: Optional[int] = None,
    fixed_features: Optional[dict[int, float]] = None,
    options: Optional[dict[str, Union[bool, float, int, str]]] = None,
    final_options: Optional[dict[str, Union[bool, float, int, str]]] = None,
    batch_initial_conditions: Optional[Tensor] = None,
    post_processing_func: Optional[Callable[[Tensor], Tensor]] = None,
    prune_tolerance: float = 1e-4,
) -> tuple[Tensor, Tensor]:
    r"""Generate a set of candidates via multi-start optimization.

    Args:
        acq_function: An AcquisitionFunction.
        bounds: A `2 x d` tensor of lower and upper bounds for each column of `X`.
        q: The number of candidates.
        homotopy: Homotopy object that will make the necessary modifications to the
            problem when calling `step()`.
        num_restarts: The number of starting points for multistart acquisition
            function optimization.
        raw_samples: The number of samples for initialization. This is required
            if `batch_initial_conditions` is not specified.
        fixed_features: A map `{feature_index: value}` for features that
            should be fixed to a particular value during generation.
        options: Options for candidate generation.
        final_options: Options for candidate generation in the last homotopy step.
        batch_initial_conditions: A tensor to specify the initial conditions. Set
            this if you do not want to use default initialization strategy.
        post_processing_func: Post processing function (such as roundingor clamping)
            that is applied before choosing the final candidate.
    """
    candidate_list, acq_value_list = [], []
    if q > 1:
        base_X_pending = acq_function.X_pending

    for _ in range(q):
        candidates = batch_initial_conditions
        homotopy.restart()

        while not homotopy.should_stop:
            candidates, acq_values = optimize_acqf(
                q=1,
                acq_function=acq_function,
                bounds=bounds,
                num_restarts=num_restarts,
                batch_initial_conditions=candidates,
                raw_samples=raw_samples,
                fixed_features=fixed_features,
                return_best_only=False,
                options=options,
            )
            homotopy.step()

            # Prune candidates
            candidates = prune_candidates(
                candidates=candidates.squeeze(1),
                acq_values=acq_values,
                prune_tolerance=prune_tolerance,
            ).unsqueeze(1)

        # Optimize one more time with the final options
        candidates, acq_values = optimize_acqf(
            q=1,
            acq_function=acq_function,
            bounds=bounds,
            num_restarts=num_restarts,
            batch_initial_conditions=candidates,
            return_best_only=False,
            options=final_options,
        )

        # Post-process the candidates and grab the best candidate
        if post_processing_func is not None:
            candidates = post_processing_func(candidates)
            acq_values = acq_function(candidates)
        best = torch.argmax(acq_values.view(-1), dim=0)
        candidate, acq_value = candidates[best], acq_values[best]

        # Keep the new candidate and update the pending points
        candidate_list.append(candidate)
        acq_value_list.append(acq_value)
        selected_candidates = torch.cat(candidate_list, dim=-2)
        if q > 1:
            acq_function.set_X_pending(
                torch.cat([base_X_pending, selected_candidates], dim=-2)
                if base_X_pending is not None
                else selected_candidates
            )

    if q > 1:  # Reset acq_function to previous X_pending state
        acq_function.set_X_pending(base_X_pending)
    homotopy.reset()  # Reset the homotopy parameters

    return selected_candidates, torch.stack(acq_value_list)
