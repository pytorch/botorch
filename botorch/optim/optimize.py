#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Methods for optimizing acquisition functions.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from botorch.acquisition.acquisition import (
    AcquisitionFunction,
    OneShotAcquisitionFunction,
)
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.generation.gen import gen_candidates_scipy
from botorch.logging import logger
from botorch.optim.initializers import (
    gen_batch_initial_conditions,
    gen_one_shot_kg_initial_conditions,
)
from botorch.optim.stopping import ExpMAStoppingCriterion
from torch import Tensor


def optimize_acqf(
    acq_function: AcquisitionFunction,
    bounds: Tensor,
    q: int,
    num_restarts: int,
    raw_samples: int,
    options: Optional[Dict[str, Union[bool, float, int, str]]] = None,
    inequality_constraints: Optional[List[Tuple[Tensor, Tensor, float]]] = None,
    equality_constraints: Optional[List[Tuple[Tensor, Tensor, float]]] = None,
    fixed_features: Optional[Dict[int, float]] = None,
    post_processing_func: Optional[Callable[[Tensor], Tensor]] = None,
    batch_initial_conditions: Optional[Tensor] = None,
    return_best_only: bool = True,
    sequential: bool = False,
    **kwargs: Any,
) -> Tuple[Tensor, Tensor]:
    r"""Generate a set of candidates via multi-start optimization.

    Args:
        acq_function: An AcquisitionFunction.
        bounds: A `2 x d` tensor of lower and upper bounds for each column of `X`.
        q: The number of candidates.
        num_restarts: The number of starting points for multistart acquisition
            function optimization.
        raw_samples: The number of samples for initialization.
        options: Options for candidate generation.
        inequality constraints: A list of tuples (indices, coefficients, rhs),
            with each tuple encoding an inequality constraint of the form
            `\sum_i (X[indices[i]] * coefficients[i]) >= rhs`
        equality constraints: A list of tuples (indices, coefficients, rhs),
            with each tuple encoding an inequality constraint of the form
            `\sum_i (X[indices[i]] * coefficients[i]) = rhs`
        fixed_features: A map `{feature_index: value}` for features that
            should be fixed to a particular value during generation.
        post_processing_func: A function that post-processes an optimization
            result appropriately (i.e., according to `round-trip`
            transformations).
        batch_initial_conditions: A tensor to specify the initial conditions. Set
            this if you do not want to use default initialization strategy.
        return_best_only: If False, outputs the solutions corresponding to all
            random restart initializations of the optimization.
        sequential: If False, uses joint optimization, otherwise uses sequential
            optimization.
        kwargs: Additonal keyword arguments.

    Returns:
        A two-element tuple containing

        - a `(num_restarts) x q x d`-dim tensor of generated candidates.
        - a tensor of associated acquisition values. If `sequential=False`,
            this is a `(num_restarts)`-dim tensor of joint acquisition values
            (with explicit restart dimension if `return_best_only=False`). If
            `sequential=True`, this is a `q`-dim tensor of expected acquisition
            values conditional on having observed canidates `0,1,...,i-1`.

    Example:
        >>> # generate `q=2` candidates jointly using 20 random restarts
        >>> # and 512 raw samples
        >>> candidates, acq_value = optimize_acqf(qEI, bounds, 2, 20, 512)

        >>> generate `q=3` candidates sequentially using 15 random restarts
        >>> # and 256 raw samples
        >>> qEI = qExpectedImprovement(model, best_f=0.2)
        >>> bounds = torch.tensor([[0.], [1.]])
        >>> candidates, acq_value_list = optimize_acqf(
        >>>     qEI, bounds, 3, 15, 256, sequential=True
        >>> )

    """
    if sequential and q > 1:
        if not return_best_only:
            raise NotImplementedError(
                "return_best_only=False only supported for joint optimization"
            )
        if isinstance(acq_function, OneShotAcquisitionFunction):
            raise NotImplementedError(
                "sequential optimization currently not supported for one-shot "
                "acquisition functions. Must have `sequential=False`."
            )
        candidate_list, acq_value_list = [], []
        candidates = torch.tensor([], device=bounds.device, dtype=bounds.dtype)
        base_X_pending = acq_function.X_pending
        for i in range(q):
            candidate, acq_value = optimize_acqf(
                acq_function=acq_function,
                bounds=bounds,
                q=1,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                options=options or {},
                inequality_constraints=inequality_constraints,
                equality_constraints=equality_constraints,
                fixed_features=fixed_features,
                post_processing_func=post_processing_func,
                batch_initial_conditions=None,
                return_best_only=True,
                sequential=False,
            )
            candidate_list.append(candidate)
            acq_value_list.append(acq_value)
            candidates = torch.cat(candidate_list, dim=-2)
            acq_function.set_X_pending(
                torch.cat([base_X_pending, candidates], dim=-2)
                if base_X_pending is not None
                else candidates
            )
            logger.info(f"Generated sequential candidate {i+1} of {q}")
        # Reset acq_func to previous X_pending state
        acq_function.set_X_pending(base_X_pending)
        return candidates, torch.stack(acq_value_list)

    options = options or {}

    if batch_initial_conditions is None:
        ic_gen = (
            gen_one_shot_kg_initial_conditions
            if isinstance(acq_function, qKnowledgeGradient)
            else gen_batch_initial_conditions
        )
        # TODO: Generating initial candidates should use parameter constraints.
        batch_initial_conditions = ic_gen(
            acq_function=acq_function,
            bounds=bounds,
            q=q,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            options=options,
        )

    batch_limit: int = options.get("batch_limit", num_restarts)
    batch_candidates_list: List[Tensor] = []
    batch_acq_values_list: List[Tensor] = []
    start_idcs = list(range(0, num_restarts, batch_limit))
    for start_idx in start_idcs:
        end_idx = min(start_idx + batch_limit, num_restarts)
        # optimize using random restart optimization
        batch_candidates_curr, batch_acq_values_curr = gen_candidates_scipy(
            initial_conditions=batch_initial_conditions[start_idx:end_idx],
            acquisition_function=acq_function,
            lower_bounds=bounds[0],
            upper_bounds=bounds[1],
            options={
                k: v
                for k, v in options.items()
                if k not in ("init_batch_limit", "batch_limit", "nonnegative")
            },
            inequality_constraints=inequality_constraints,
            equality_constraints=equality_constraints,
            fixed_features=fixed_features,
        )
        batch_candidates_list.append(batch_candidates_curr)
        batch_acq_values_list.append(batch_acq_values_curr)
        logger.info(f"Generated candidate batch {start_idx+1} of {len(start_idcs)}.")
    batch_candidates = torch.cat(batch_candidates_list)
    batch_acq_values = torch.cat(batch_acq_values_list)

    if post_processing_func is not None:
        batch_candidates = post_processing_func(batch_candidates)

    if return_best_only:
        best = torch.argmax(batch_acq_values.view(-1), dim=0)
        batch_candidates = batch_candidates[best]
        batch_acq_values = batch_acq_values[best]

    if isinstance(acq_function, OneShotAcquisitionFunction):
        if not kwargs.get("return_full_tree", False):
            batch_candidates = acq_function.extract_candidates(X_full=batch_candidates)

    return batch_candidates, batch_acq_values


def optimize_acqf_cyclic(
    acq_function: AcquisitionFunction,
    bounds: Tensor,
    q: int,
    num_restarts: int,
    raw_samples: int,
    options: Optional[Dict[str, Union[bool, float, int, str]]] = None,
    inequality_constraints: Optional[List[Tuple[Tensor, Tensor, float]]] = None,
    equality_constraints: Optional[List[Tuple[Tensor, Tensor, float]]] = None,
    fixed_features: Optional[Dict[int, float]] = None,
    post_processing_func: Optional[Callable[[Tensor], Tensor]] = None,
    batch_initial_conditions: Optional[Tensor] = None,
    cyclic_options: Optional[Dict[str, Union[bool, float, int, str]]] = None,
) -> Tuple[Tensor, Tensor]:
    r"""Generate a set of `q` candidates via cyclic optimization.

    Args:
        acq_function: An AcquisitionFunction
        bounds: A `2 x d` tensor of lower and upper bounds for each column of `X`.
        q: The number of candidates.
        num_restarts:  Number of starting points for multistart acquisition
            function optimization.
        raw_samples: Number of samples for initialization
        options: Options for candidate generation.
        inequality constraints: A list of tuples (indices, coefficients, rhs),
            with each tuple encoding an inequality constraint of the form
            `\sum_i (X[indices[i]] * coefficients[i]) >= rhs`
        equality constraints: A list of tuples (indices, coefficients, rhs),
            with each tuple encoding an inequality constraint of the form
            `\sum_i (X[indices[i]] * coefficients[i]) = rhs`
        fixed_features: A map `{feature_index: value}` for features that
            should be fixed to a particular value during generation.
        post_processing_func: A function that post-processes an optimization
            result appropriately (i.e., according to `round-trip`
            transformations).
        batch_initial_conditions: A tensor to specify the initial conditions.
            If no initial conditions are provided, the default initialization will
            be used.
        cyclic_options: Options for stopping criterion for outer cyclic optimization.

    Returns:
        A two-element tuple containing

        - a `q x d`-dim tensor of generated candidates.
        - a `q`-dim tensor of expected acquisition values, where the value at
            index `i` is the acquisition value conditional on having observed
            all candidates except candidate `i`.

    Example:
        >>> # generate `q=3` candidates cyclically using 15 random restarts
        >>> # 256 raw samples, and 4 cycles
        >>>
        >>> qEI = qExpectedImprovement(model, best_f=0.2)
        >>> bounds = torch.tensor([[0.], [1.]])
        >>> candidates, acq_value_list = optimize_acqf_cyclic(
        >>>     qEI, bounds, 3, 15, 256, cyclic_options={"maxiter": 4}
        >>> )

    """
    # for the first cycle, optimize the q candidates sequentially
    candidates, acq_vals = optimize_acqf(
        acq_function=acq_function,
        bounds=bounds,
        q=q,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        options=options,
        inequality_constraints=inequality_constraints,
        equality_constraints=equality_constraints,
        fixed_features=fixed_features,
        post_processing_func=post_processing_func,
        batch_initial_conditions=batch_initial_conditions,
        return_best_only=True,
        sequential=True,
    )
    if q > 1:
        cyclic_options = cyclic_options or {}
        stopping_criterion = ExpMAStoppingCriterion(**cyclic_options)
        stop = stopping_criterion.evaluate(fvals=acq_vals)
        base_X_pending = acq_function.X_pending
        idxr = torch.ones(q, dtype=torch.bool, device=bounds.device)
        while not stop:
            for i in range(q):
                # optimize only candidate i
                idxr[i] = 0
                acq_function.set_X_pending(
                    torch.cat([base_X_pending, candidates[idxr]], dim=-2)
                    if base_X_pending is not None
                    else candidates[idxr]
                )
                candidate_i, acq_val_i = optimize_acqf(
                    acq_function=acq_function,
                    bounds=bounds,
                    q=1,
                    num_restarts=num_restarts,
                    raw_samples=raw_samples,
                    options=options,
                    inequality_constraints=inequality_constraints,
                    equality_constraints=equality_constraints,
                    fixed_features=fixed_features,
                    post_processing_func=post_processing_func,
                    batch_initial_conditions=candidates[i].unsqueeze(0),
                    return_best_only=True,
                    sequential=True,
                )
                candidates[i] = candidate_i
                acq_vals[i] = acq_val_i
                idxr[i] = 1
            stop = stopping_criterion.evaluate(fvals=acq_vals)
        acq_function.set_X_pending(base_X_pending)
    return candidates, acq_vals


def optimize_acqf_list(
    acq_function_list: List[AcquisitionFunction],
    bounds: Tensor,
    num_restarts: int,
    raw_samples: int,
    options: Optional[Dict[str, Union[bool, float, int, str]]] = None,
    inequality_constraints: Optional[List[Tuple[Tensor, Tensor, float]]] = None,
    equality_constraints: Optional[List[Tuple[Tensor, Tensor, float]]] = None,
    fixed_features: Optional[Dict[int, float]] = None,
    post_processing_func: Optional[Callable[[Tensor], Tensor]] = None,
) -> Tuple[Tensor, Tensor]:
    r"""Generate a list of candidates from a list of acquisition functions.

    The acquisition functions are optimized in sequence, with previous candidates
    set as `X_pending`. This is also known as sequential greedy optimization.

    Args:
        acq_function_list: A list of acquisition functions.
        bounds: A `2 x d` tensor of lower and upper bounds for each column of `X`.
        num_restarts:  Number of starting points for multistart acquisition
            function optimization.
        raw_samples: Number of samples for initialization
        options: Options for candidate generation.
        inequality constraints: A list of tuples (indices, coefficients, rhs),
            with each tuple encoding an inequality constraint of the form
            `\sum_i (X[indices[i]] * coefficients[i]) >= rhs`
        equality constraints: A list of tuples (indices, coefficients, rhs),
            with each tuple encoding an inequality constraint of the form
            `\sum_i (X[indices[i]] * coefficients[i]) = rhs`
        fixed_features: A map `{feature_index: value}` for features that
            should be fixed to a particular value during generation.
        post_processing_func: A function that post-processes an optimization
            result appropriately (i.e., according to `round-trip`
            transformations).

    Returns:
        A two-element tuple containing

        - a `q x d`-dim tensor of generated candidates.
        - a `q`-dim tensor of expected acquisition values, where the value at
            index `i` is the acquisition value conditional on having observed
            all candidates except candidate `i`.

    """
    if not acq_function_list:
        raise ValueError("acq_function_list must be non-empty.")
    candidate_list, acq_value_list = [], []
    candidates = torch.tensor([], device=bounds.device, dtype=bounds.dtype)
    base_X_pending = acq_function_list[0].X_pending
    for acq_function in acq_function_list:
        if candidate_list:
            acq_function.set_X_pending(
                torch.cat([base_X_pending, candidates], dim=-2)
                if base_X_pending is not None
                else candidates
            )
        candidate, acq_value = optimize_acqf(
            acq_function=acq_function,
            bounds=bounds,
            q=1,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            options=options or {},
            inequality_constraints=inequality_constraints,
            equality_constraints=equality_constraints,
            fixed_features=fixed_features,
            post_processing_func=post_processing_func,
            return_best_only=True,
            sequential=False,
        )
        candidate_list.append(candidate)
        acq_value_list.append(acq_value)
        candidates = torch.cat(candidate_list, dim=-2)
    return candidates, torch.stack(acq_value_list)
