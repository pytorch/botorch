#!/usr/bin/env python3

import warnings
from typing import Callable, Dict, Optional, Union

import torch
from torch import Tensor
from torch.nn import Module

from ..acquisition import AcquisitionFunction
from ..acquisition.analytic import AnalyticAcquisitionFunction
from ..exceptions import BadInitialCandidatesWarning, UnsupportedError
from ..gen import gen_candidates_scipy, get_best_candidates
from ..utils.sampling import draw_sobol_samples
from .initializers import (
    get_similarity_measure,
    initialize_q_batch,
    initialize_q_batch_simple,
)


def sequential_optimize(
    acq_function: AcquisitionFunction,
    bounds: Tensor,
    q: int,
    num_restarts: int,
    raw_samples: int,
    options: Dict[str, Union[bool, float, int]],
    fixed_features: Optional[Dict[int, float]] = None,
    post_processing_func: Optional[Callable[[Tensor], Tensor]] = None,
) -> Tensor:
    """
    Returns a set of candidates via sequential multi-start optimization.

    Args:
        acq_function: A qNoisyExpectedImprovement acquisition function.
        bounds: A `2 x d` tensor of lower and upper bounds for each column of X.
        q: The number of candidates.
        num_restarts:  Number of starting points for multistart acquisition
            function optimization.
        raw_samples: number of samples for initialization
        options: options for candidate generation.
        fixed_features: A map {feature_index: value} for features that
            should be fixed to a particular value during generation.
        post_processing_func: A function that post_processes an optimization result
            appropriately (i.e., according to `round-trip` transformations).

    Returns:
        The set of generated candidates.,
    """
    if not hasattr(acq_function, "X_baseline"):
        raise UnsupportedError(  # pyre-ignore: [16]
            "Sequential Optimization only supporte for acquisition functions "
            "with an `X_pending` property"
        )
    candidate_list = []
    candidates = torch.tensor([])
    base_X_baseline = acq_function.X_baseline  # pyre-ignore: [16]
    for _ in range(q):
        candidate = joint_optimize(
            acq_function=acq_function,
            bounds=bounds,
            q=1,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            options=options,
            fixed_features=fixed_features,
        )
        if post_processing_func is not None:
            candidate_shape = candidate.shape
            candidate = post_processing_func(candidate.view(-1)).view(*candidate_shape)
        candidate_list.append(candidate)
        candidates = torch.cat(candidate_list, dim=-2)
        acq_function.X_baseline = (
            torch.cat([base_X_baseline, candidates], dim=-2)
            if base_X_baseline is not None
            else candidates
        )
    # Reset acq_func to previous X_baseline state
    acq_function.X_baseline = base_X_baseline
    return candidates


def joint_optimize(
    acq_function: AcquisitionFunction,
    bounds: Tensor,
    q: int,
    num_restarts: int,
    raw_samples: int,
    options: Dict[str, Union[bool, float, int]],
    fixed_features: Optional[Dict[int, float]] = None,
    post_processing_func: Optional[Callable[[Tensor], Tensor]] = None,
) -> Tensor:
    """
    Returns a set of candidates via joint multi-start optimization.

    Args:
        acq_function:  An acquisition function Module
        bounds: A `2 x d` tensor of lower and upper bounds for each column of X.
        q: The number of candidates
        num_restarts:  Number of starting points for multistart acquisition
            function optimization.
        raw_samples: number of samples for initialization
        options: options for candidate generation
        fixed_features: A map {feature_index: value} for features that
            should be fixed to a particular value during generation.
        post_processing_func: A function that post processes an optimization result
            appropriately (i.e., according to `round-trip` transformations).
            Note: post_processing_func is not used by _joint_optimize and is only
            included to match _sequential_optimize.

    Returns:
        The set of generated candidates
    """
    if isinstance(acq_function, AnalyticAcquisitionFunction):
        # TODO: use tensor metadata (e.g. NamedTensors) when available
        q = None

    batch_initial_candidates = gen_batch_initial_candidates(
        acq_function=acq_function,
        bounds=bounds,
        q=q,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        options=options,
    )
    # optimize using random restart optimization
    batch_candidates, batch_acq_values = gen_candidates_scipy(
        initial_candidates=batch_initial_candidates,
        acquisition_function=acq_function,
        lower_bounds=bounds[0],
        upper_bounds=bounds[1],
        options=options,
        fixed_features=fixed_features,
    )
    return get_best_candidates(
        batch_candidates=batch_candidates, batch_values=batch_acq_values
    )


def gen_batch_initial_candidates(
    acq_function: Module,
    bounds: Tensor,
    q: Optional[int],
    num_restarts: int,
    raw_samples: int,
    options: Dict[str, Union[bool, float, int]],
) -> Tensor:
    seed: Optional[int] = options.get("seed")  # pyre-ignore
    batch_initial_arms: Tensor
    factor, max_factor = 1, 5
    while factor < max_factor:
        with warnings.catch_warnings(record=True) as ws:
            X_rnd = draw_sobol_samples(
                bounds=bounds,
                n=raw_samples * factor,
                q=1 if q is None else q,
                seed=seed,
            )
            if q is None:
                X_rnd = X_rnd.squeeze(dim=-2)
            with torch.no_grad():
                Y_rnd = acq_function(X_rnd)
            if options.get("simple_init", True):
                batch_initial_candidates = initialize_q_batch_simple(
                    X=X_rnd, Y=Y_rnd, n=num_restarts, options=options
                )
            else:
                sim_measure = get_similarity_measure(model=acq_function.model)
                batch_initial_candidates = initialize_q_batch(
                    X=X_rnd,
                    Y=Y_rnd,
                    n=num_restarts,
                    sim_measure=sim_measure,
                    options=options,
                )

            if not any(
                issubclass(w.category, BadInitialCandidatesWarning)  # pyre-ignore: [16]
                for w in ws  # pyre-ignore: [16]
            ):
                return batch_initial_candidates
            if factor < max_factor:
                factor += 1
    warnings.warn(
        "Unable to find non-zero acquistion function values - initial arms "
        "are being selected randomly.",
        BadInitialCandidatesWarning,  # pyre-ignore: [16]
    )
    return batch_initial_candidates
