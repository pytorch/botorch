# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any
import warnings

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
    *,
    prune_tolerance: float = 1e-4,
    batch_initial_conditions: Tensor | None = None,
    optimize_acqf_loop_kwargs: dict[str, Any] | None = None,
    optimize_acqf_final_kwargs: dict[str, Any] | None = None,
) -> tuple[Tensor, Tensor]:
    r"""Generate a set of candidates via multi-start optimization.

    Args:
        acq_function: An AcquisitionFunction.
        bounds: A `2 x d` tensor of lower and upper bounds for each column of `X`.
        q: The number of candidates.
        homotopy: Homotopy object that will make the necessary modifications to the
            problem when calling `step()`.
        prune_tolerance: The minimum distance to prune candidates.
        batch_initial_conditions: A tensor to specify the initial conditions. Set
            this if you do not want to use default initialization strategy.
        optimize_acqf_loop_kwargs: A dictionary of keyword arguments for
            `optimize_acqf`. These settings are used in the homotopy loop.
        optimize_acqf_final_kwargs: A dictionary of keyword arguments for
            `optimize_acqf`. These settings are used for the final optimization
            after the homotopy loop.
    """
    if optimize_acqf_loop_kwargs is None:
        optimize_acqf_loop_kwargs = {}

    if optimize_acqf_final_kwargs is None:
        optimize_acqf_final_kwargs = {}

    for kwarg_dict_name, kwarg_dict in [("optimize_acqf_loop_kwargs", optimize_acqf_loop_kwargs), ("optimize_acqf_final_kwargs", optimize_acqf_final_kwargs)]:

        if "return_best_only" in kwarg_dict:
            warnings.warn(
                f"`return_best_only` is set to True in `{kwarg_dict_name}`, setting to False."
            )
            kwarg_dict["return_best_only"] = False

        if "q" in kwarg_dict:
            warnings.warn(
                f"`q` is set in `{kwarg_dict_name}`, setting to 1."
            )
            kwarg_dict["q"] = 1

        if "batch_initial_conditions" in kwarg_dict:
            warnings.warn(
                f"`batch_initial_conditions` is set in `{kwarg_dict_name}`, setting to None."
            )
            kwarg_dict.pop("batch_initial_conditions")

        for arg_name, arg_value in [("acq_function", acq_function), ("bounds", bounds)]:
            if arg_name in kwarg_dict:
                warnings.warn(
                    f"`{arg_name}` is set in `{kwarg_dict_name}` and will be "
                    "overridden in favor of the value in `optimize_acqf_homotopy`. "
                    f"({arg_name} = {arg_value} c.f. {kwarg_dict_name}[{arg_name}] = {kwarg_dict[arg_name]})"
                )
                kwarg_dict[arg_name] = arg_value

    if "post_processing_func" in optimize_acqf_loop_kwargs:
        warnings.warn(
            "`post_processing_func` is set in `optimize_acqf_loop_kwargs`, setting to None."
        )
        optimize_acqf_loop_kwargs["post_processing_func"] = None

    candidate_list, acq_value_list = [], []
    if q > 1:
        base_X_pending = acq_function.X_pending

    for _ in range(q):
        candidates = batch_initial_conditions
        homotopy.restart()

        while not homotopy.should_stop:
            candidates, acq_values = optimize_acqf(batch_initial_conditions=candidates, **optimize_acqf_loop_kwargs)
            homotopy.step()

            # Prune candidates
            candidates = prune_candidates(
                candidates=candidates.squeeze(1),
                acq_values=acq_values,
                prune_tolerance=prune_tolerance,
            ).unsqueeze(1)

        # Optimize one more time with the final options
        candidates, acq_values = optimize_acqf(batch_initial_conditions=candidates, **optimize_acqf_final_kwargs)

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
