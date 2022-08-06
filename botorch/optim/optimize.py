#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Methods for optimizing acquisition functions.
"""

from __future__ import annotations

import warnings

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from botorch.acquisition.acquisition import (
    AcquisitionFunction,
    OneShotAcquisitionFunction,
)
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.exceptions import InputDataError, UnsupportedError
from botorch.exceptions.warnings import OptimizationWarning
from botorch.generation.gen import gen_candidates_scipy
from botorch.logging import logger
from botorch.optim.initializers import (
    gen_batch_initial_conditions,
    gen_one_shot_kg_initial_conditions,
)
from botorch.optim.stopping import ExpMAStoppingCriterion
from torch import Tensor

INIT_OPTION_KEYS = {
    # set of options for initialization that we should
    # not pass to scipy.optimize.minimize to avoid
    # warnings
    "alpha",
    "batch_limit",
    "eta",
    "init_batch_limit",
    "nonnegative",
    "n_burnin",
    "sample_around_best",
    "sample_around_best_sigma",
    "sample_around_best_prob_perturb",
    "sample_around_best_prob_perturb",
    "seed",
    "thinning",
}


def optimize_acqf(
    acq_function: AcquisitionFunction,
    bounds: Tensor,
    q: int,
    num_restarts: int,
    raw_samples: Optional[int] = None,
    options: Optional[Dict[str, Union[bool, float, int, str]]] = None,
    inequality_constraints: Optional[List[Tuple[Tensor, Tensor, float]]] = None,
    equality_constraints: Optional[List[Tuple[Tensor, Tensor, float]]] = None,
    nonlinear_inequality_constraints: Optional[List[Callable]] = None,
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
        bounds: A `2 x d` tensor of lower and upper bounds for each column of `X`
            (if inequality_constraints is provided, these bounds can be -inf and
            +inf, respectively).
        q: The number of candidates.
        num_restarts: The number of starting points for multistart acquisition
            function optimization.
        raw_samples: The number of samples for initialization. This is required
            if `batch_initial_conditions` is not specified.
        options: Options for candidate generation.
        inequality_constraints: A list of tuples (indices, coefficients, rhs),
            with each tuple encoding an inequality constraint of the form
            `\sum_i (X[indices[i]] * coefficients[i]) >= rhs`
        equality_constraints: A list of tuples (indices, coefficients, rhs),
            with each tuple encoding an inequality constraint of the form
            `\sum_i (X[indices[i]] * coefficients[i]) = rhs`
        nonlinear_inequality_constraints: A list of callables with that represent
            non-linear inequality constraints of the form `callable(x) >= 0`. Each
            callable is expected to take a `(num_restarts) x q x d`-dim tensor as an
            input and return a `(num_restarts) x q`-dim tensor with the constraint
            values. The constraints will later be passed to SLSQP. You need to pass in
            `batch_initial_conditions` in this case. Using non-linear inequality
            constraints also requires that `batch_limit` is set to 1, which will be
            done automatically if not specified in `options`.
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
    if inequality_constraints is None:
        if not (bounds.ndim == 2 and bounds.shape[0] == 2):
            raise ValueError(
                "bounds should be a `2 x d` tensor, current shape: "
                f"{list(bounds.shape)}."
            )
        # TODO: Validate constraints if provided:
        # https://github.com/pytorch/botorch/pull/1231

    if sequential and q > 1:
        if not return_best_only:
            raise NotImplementedError(
                "`return_best_only=False` only supported for joint optimization."
            )
        if isinstance(acq_function, OneShotAcquisitionFunction):
            raise NotImplementedError(
                "sequential optimization currently not supported for one-shot "
                "acquisition functions. Must have `sequential=False`."
            )
        candidate_list, acq_value_list = [], []
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
                nonlinear_inequality_constraints=nonlinear_inequality_constraints,
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

    # Handle the trivial case when all features are fixed
    if fixed_features is not None and len(fixed_features) == bounds.shape[-1]:
        X = torch.tensor(
            [fixed_features[i] for i in range(bounds.shape[-1])],
            device=bounds.device,
            dtype=bounds.dtype,
        )
        X = X.expand(q, *X.shape)
        with torch.no_grad():
            acq_value = acq_function(X)
        return X, acq_value

    initial_conditions_provided = batch_initial_conditions is not None
    if not initial_conditions_provided:
        if nonlinear_inequality_constraints:
            raise NotImplementedError(
                "`batch_initial_conditions` must be given if there are non-linear "
                "inequality constraints."
            )
        if raw_samples is None:
            raise ValueError(
                "Must specify `raw_samples` when `batch_initial_conditions` is `None`."
            )

    def _gen_initial_conditions() -> Tensor:
        ic_gen = (
            gen_one_shot_kg_initial_conditions
            if isinstance(acq_function, qKnowledgeGradient)
            else gen_batch_initial_conditions
        )
        batch_initial_conditions = ic_gen(
            acq_function=acq_function,
            bounds=bounds,
            q=q,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            fixed_features=fixed_features,
            options=options,
            inequality_constraints=inequality_constraints,
            equality_constraints=equality_constraints,
        )
        return batch_initial_conditions

    if not initial_conditions_provided:
        batch_initial_conditions = _gen_initial_conditions()

    batch_limit: int = options.get(
        "batch_limit", num_restarts if not nonlinear_inequality_constraints else 1
    )

    def _optimize_batch_candidates() -> Tuple[Tensor, Tensor, List[Warning]]:
        batch_candidates_list: List[Tensor] = []
        batch_acq_values_list: List[Tensor] = []
        batched_ics = batch_initial_conditions.split(batch_limit)
        opt_warnings = []

        scipy_kws = dict(
            acquisition_function=acq_function,
            lower_bounds=None if bounds[0].isinf().all() else bounds[0],
            upper_bounds=None if bounds[1].isinf().all() else bounds[1],
            options={k: v for k, v in options.items() if k not in INIT_OPTION_KEYS},
            inequality_constraints=inequality_constraints,
            equality_constraints=equality_constraints,
            nonlinear_inequality_constraints=nonlinear_inequality_constraints,
            fixed_features=fixed_features,
        )

        for i, batched_ics_ in enumerate(batched_ics):
            # optimize using random restart optimization
            with warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter("always", category=OptimizationWarning)
                batch_candidates_curr, batch_acq_values_curr = gen_candidates_scipy(
                    initial_conditions=batched_ics_, **scipy_kws
                )
            opt_warnings += ws
            batch_candidates_list.append(batch_candidates_curr)
            batch_acq_values_list.append(batch_acq_values_curr)
            logger.info(f"Generated candidate batch {i+1} of {len(batched_ics)}.")

        batch_candidates = torch.cat(batch_candidates_list)
        batch_acq_values = torch.cat(batch_acq_values_list)
        return batch_candidates, batch_acq_values, opt_warnings

    batch_candidates, batch_acq_values, ws = _optimize_batch_candidates()

    optimization_warning_raised = any(
        (issubclass(w.category, OptimizationWarning) for w in ws)
    )
    if optimization_warning_raised:
        first_warn_msg = (
            "Optimization failed in `gen_candidates_scipy` with the following "
            f"warning(s):\n{[w.message for w in ws]}\nBecause you specified "
            "`batch_initial_conditions`, optimization will not be retried with "
            "new initial conditions and will proceed with the current solution."
            " Suggested remediation: Try again with different "
            "`batch_initial_conditions`, or don't provide `batch_initial_conditions.`"
            if initial_conditions_provided
            else "Optimization failed in `gen_candidates_scipy` with the following "
            f"warning(s):\n{[w.message for w in ws]}\nTrying again with a new "
            "set of initial conditions."
        )
        warnings.warn(first_warn_msg, RuntimeWarning)

        if not initial_conditions_provided:
            batch_initial_conditions = _gen_initial_conditions()

            batch_candidates, batch_acq_values, ws = _optimize_batch_candidates()

            optimization_warning_raised = any(
                (issubclass(w.category, OptimizationWarning) for w in ws)
            )
            if optimization_warning_raised:
                warnings.warn(
                    "Optimization failed on the second try, after generating a "
                    "new set of initial conditions.",
                    RuntimeWarning,
                )

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
    raw_samples: Optional[int] = None,
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
        bounds: A `2 x d` tensor of lower and upper bounds for each column of `X`
            (if inequality_constraints is provided, these bounds can be -inf and
            +inf, respectively).
        q: The number of candidates.
        num_restarts:  Number of starting points for multistart acquisition
            function optimization.
        raw_samples: Number of samples for initialization. This is required
            if `batch_initial_conditions` is not specified.
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
    raw_samples: Optional[int] = None,
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
        bounds: A `2 x d` tensor of lower and upper bounds for each column of `X`
            (if inequality_constraints is provided, these bounds can be -inf and
            +inf, respectively).
        num_restarts:  Number of starting points for multistart acquisition
            function optimization.
        raw_samples: Number of samples for initialization. This is required
            if `batch_initial_conditions` is not specified.
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


def optimize_acqf_mixed(
    acq_function: AcquisitionFunction,
    bounds: Tensor,
    q: int,
    num_restarts: int,
    fixed_features_list: List[Dict[int, float]],
    raw_samples: Optional[int] = None,
    options: Optional[Dict[str, Union[bool, float, int, str]]] = None,
    inequality_constraints: Optional[List[Tuple[Tensor, Tensor, float]]] = None,
    equality_constraints: Optional[List[Tuple[Tensor, Tensor, float]]] = None,
    post_processing_func: Optional[Callable[[Tensor], Tensor]] = None,
    batch_initial_conditions: Optional[Tensor] = None,
    **kwargs: Any,
) -> Tuple[Tensor, Tensor]:
    r"""Optimize over a list of fixed_features and returns the best solution.

    This is useful for optimizing over mixed continuous and discrete domains.
    For q > 1 this function always performs sequential greedy optimization (with
    proper conditioning on generated candidates).

    Args:
        acq_function: An AcquisitionFunction
        bounds: A `2 x d` tensor of lower and upper bounds for each column of `X`
            (if inequality_constraints is provided, these bounds can be -inf and
            +inf, respectively).
        q: The number of candidates.
        num_restarts:  Number of starting points for multistart acquisition
            function optimization.
        raw_samples: Number of samples for initialization. This is required
            if `batch_initial_conditions` is not specified.
        fixed_features_list: A list of maps `{feature_index: value}`. The i-th
            item represents the fixed_feature for the i-th optimization.
        options: Options for candidate generation.
        inequality constraints: A list of tuples (indices, coefficients, rhs),
            with each tuple encoding an inequality constraint of the form
            `\sum_i (X[indices[i]] * coefficients[i]) >= rhs`
        equality constraints: A list of tuples (indices, coefficients, rhs),
            with each tuple encoding an inequality constraint of the form
            `\sum_i (X[indices[i]] * coefficients[i]) = rhs`
        post_processing_func: A function that post-processes an optimization
            result appropriately (i.e., according to `round-trip`
            transformations).
        batch_initial_conditions: A tensor to specify the initial conditions. Set
            this if you do not want to use default initialization strategy.

    Returns:
        A two-element tuple containing

        - a `q x d`-dim tensor of generated candidates.
        - an associated acquisition value.
    """
    if not fixed_features_list:
        raise ValueError("fixed_features_list must be non-empty.")

    if isinstance(acq_function, OneShotAcquisitionFunction):
        if not hasattr(acq_function, "evaluate") and q > 1:
            raise ValueError(
                "`OneShotAcquisitionFunction`s that do not implement `evaluate` "
                "are currently not supported when `q > 1`. This is needed to "
                "compute the joint acquisition value."
            )

    if q == 1:
        ff_candidate_list, ff_acq_value_list = [], []
        for fixed_features in fixed_features_list:
            candidate, acq_value = optimize_acqf(
                acq_function=acq_function,
                bounds=bounds,
                q=q,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                options=options or {},
                inequality_constraints=inequality_constraints,
                equality_constraints=equality_constraints,
                fixed_features=fixed_features,
                post_processing_func=post_processing_func,
                batch_initial_conditions=batch_initial_conditions,
                return_best_only=True,
            )
            ff_candidate_list.append(candidate)
            ff_acq_value_list.append(acq_value)

        ff_acq_values = torch.stack(ff_acq_value_list)
        best = torch.argmax(ff_acq_values)
        return ff_candidate_list[best], ff_acq_values[best]

    # For batch optimization with q > 1 we do not want to enumerate all n_combos^n
    # possible combinations of discrete choices. Instead, we use sequential greedy
    # optimization.
    base_X_pending = acq_function.X_pending
    candidates = torch.tensor([], device=bounds.device, dtype=bounds.dtype)

    for _ in range(q):
        candidate, acq_value = optimize_acqf_mixed(
            acq_function=acq_function,
            bounds=bounds,
            q=1,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            fixed_features_list=fixed_features_list,
            options=options or {},
            inequality_constraints=inequality_constraints,
            equality_constraints=equality_constraints,
            post_processing_func=post_processing_func,
            batch_initial_conditions=batch_initial_conditions,
        )
        candidates = torch.cat([candidates, candidate], dim=-2)
        acq_function.set_X_pending(
            torch.cat([base_X_pending, candidates], dim=-2)
            if base_X_pending is not None
            else candidates
        )

    acq_function.set_X_pending(base_X_pending)

    # compute joint acquisition value
    if isinstance(acq_function, OneShotAcquisitionFunction):
        acq_value = acq_function.evaluate(X=candidates, bounds=bounds)
    else:
        acq_value = acq_function(candidates)
    return candidates, acq_value


def optimize_acqf_discrete(
    acq_function: AcquisitionFunction,
    q: int,
    choices: Tensor,
    max_batch_size: int = 2048,
    unique: bool = True,
    **kwargs: Any,
) -> Tuple[Tensor, Tensor]:
    r"""Optimize over a discrete set of points using batch evaluation.

    For `q > 1` this function generates candidates by means of sequential
    conditioning (rather than joint optimization), since for all but the
    smalles number of choices the set `choices^q` of discrete points to
    evaluate quickly explodes.

    Args:
        acq_function: An AcquisitionFunction.
        q: The number of candidates.
        choices: A `num_choices x d` tensor of possible choices.
        max_batch_size: The maximum number of choices to evaluate in batch.
            A large limit can cause excessive memory usage if the model has
            a large training set.
        unique: If True return unique choices, o/w choices may be repeated
            (only relevant if `q > 1`).

    Returns:
        A three-element tuple containing

        - a `q x d`-dim tensor of generated candidates.
        - an associated acquisition value.
    """
    if isinstance(acq_function, OneShotAcquisitionFunction):
        raise UnsupportedError(
            "Discrete optimization is not supported for"
            "one-shot acquisition functions."
        )
    if choices.numel() == 0:
        raise InputDataError("`choices` must be non-emtpy.")
    choices_batched = choices.unsqueeze(-2)
    if q > 1:
        candidate_list, acq_value_list = [], []
        base_X_pending = acq_function.X_pending
        for _ in range(q):
            with torch.no_grad():
                acq_values = _split_batch_eval_acqf(
                    acq_function=acq_function,
                    X=choices_batched,
                    max_batch_size=max_batch_size,
                )
            best_idx = torch.argmax(acq_values)
            candidate_list.append(choices_batched[best_idx])
            acq_value_list.append(acq_values[best_idx])
            # set pending points
            candidates = torch.cat(candidate_list, dim=-2)
            acq_function.set_X_pending(
                torch.cat([base_X_pending, candidates], dim=-2)
                if base_X_pending is not None
                else candidates
            )
            # need to remove choice from choice set if enforcing uniqueness
            if unique:
                choices_batched = torch.cat(
                    [choices_batched[:best_idx], choices_batched[best_idx + 1 :]]
                )

        # Reset acq_func to previous X_pending state
        acq_function.set_X_pending(base_X_pending)
        return candidates, torch.stack(acq_value_list)

    with torch.no_grad():
        acq_values = _split_batch_eval_acqf(
            acq_function=acq_function, X=choices_batched, max_batch_size=max_batch_size
        )
    best_idx = torch.argmax(acq_values)
    return choices_batched[best_idx], acq_values[best_idx]


def _split_batch_eval_acqf(
    acq_function: AcquisitionFunction, X: Tensor, max_batch_size: int
) -> Tensor:
    return torch.cat([acq_function(X_) for X_ in X.split(max_batch_size)])


def _generate_neighbors(
    x: Tensor,
    discrete_choices: List[Tensor],
    X_avoid: Tensor,
    inequality_constraints: List[Tuple[Tensor, Tensor, float]],
):
    # generate all 1D perturbations
    npts = sum([len(c) for c in discrete_choices])
    X_loc = x.repeat(npts, 1)
    j = 0
    for i, c in enumerate(discrete_choices):
        X_loc[j : j + len(c), i] = c
        j += len(c)
    # remove invalid and infeasible points (also remove x)
    X_loc = _filter_invalid(X=X_loc, X_avoid=torch.cat((X_avoid, x)))
    X_loc = _filter_infeasible(X=X_loc, inequality_constraints=inequality_constraints)
    return X_loc


def _filter_infeasible(
    X: Tensor, inequality_constraints: List[Tuple[Tensor, Tensor, float]]
):
    """Remove all points from `X` that don't satisfy the constraints."""
    is_feasible = torch.ones(X.shape[0], dtype=torch.bool, device=X.device)
    for (inds, weights, bound) in inequality_constraints:
        is_feasible &= (X[..., inds] * weights).sum(dim=-1) >= bound
    return X[is_feasible]


def _filter_invalid(X: Tensor, X_avoid: Tensor):
    """Remove all occurences of `X_avoid` from `X`."""
    return X[~(X == X_avoid.unsqueeze(-2)).all(dim=-1).any(dim=-2)]


def _gen_batch_initial_conditions_local_search(
    discrete_choices: List[Tensor],
    raw_samples: int,
    X_avoid: Tensor,
    inequality_constraints: List[Tuple[Tensor, Tensor, float]],
    min_points: int,
    max_tries: int = 100,
):
    """Generate initial conditions for local search."""
    tkwargs = {"device": discrete_choices[0].device, "dtype": discrete_choices[0].dtype}
    dim = len(discrete_choices)
    X = torch.zeros(0, dim, **tkwargs)
    for _ in range(max_tries):
        X_new = torch.zeros(raw_samples, dim, **tkwargs)
        for i, c in enumerate(discrete_choices):
            X_new[:, i] = c[
                torch.randint(low=0, high=len(c), size=(raw_samples,), device=c.device)
            ]
        X = torch.unique(torch.cat((X, X_new)), dim=0)
        X = _filter_invalid(X=X, X_avoid=X_avoid)
        X = _filter_infeasible(X=X, inequality_constraints=inequality_constraints)
        if len(X) >= min_points:
            return X
    raise RuntimeError(f"Failed to generate at least {min_points} initial conditions")


def optimize_acqf_discrete_local_search(
    acq_function: AcquisitionFunction,
    discrete_choices: List[Tensor],
    q: int,
    num_restarts: int = 20,
    raw_samples: int = 4096,
    inequality_constraints: Optional[List[Tuple[Tensor, Tensor, float]]] = None,
    X_avoid: Optional[Tensor] = None,
    batch_initial_conditions: Optional[Tensor] = None,
    max_batch_size: int = 2048,
    unique: bool = True,
    **kwargs: Any,
) -> Tuple[Tensor, Tensor]:
    r"""Optimize acquisition function over a lattice.

    This is useful when d is large and enumeration of the search space
    isn't possible. For q > 1 this function always performs sequential
    greedy optimization (with proper conditioning on generated candidates).

    NOTE: While this method supports arbitrary lattices, it has only been
    thoroughly tested for {0, 1}^d. Consider it to be in alpha stage for
    the more general case.

    Args:
        acq_function: An AcquisitionFunction
        discrete_choices: A list of possible discrete choices for each dimension.
            Each element in the list is expected to be a torch tensor.
        q: The number of candidates.
        num_restarts:  Number of starting points for multistart acquisition
            function optimization.
        raw_samples: Number of samples for initialization. This is required
            if `batch_initial_conditions` is not specified.
        inequality_constraints: A list of tuples (indices, coefficients, rhs),
            with each tuple encoding an inequality constraint of the form
            `\sum_i (X[indices[i]] * coefficients[i]) >= rhs`
        X_avoid: An `n x d` tensor of candidates that we aren't allowed to pick.
        batch_initial_conditions: A tensor of size `n x 1 x d` to specify the
            initial conditions. Set this if you do not want to use default
            initialization strategy.
        max_batch_size: The maximum number of choices to evaluate in batch.
            A large limit can cause excessive memory usage if the model has
            a large training set.
        unique: If True return unique choices, o/w choices may be repeated
            (only relevant if `q > 1`).

    Returns:
        A two-element tuple containing

        - a `q x d`-dim tensor of generated candidates.
        - an associated acquisition value.
    """
    candidate_list = []
    base_X_pending = acq_function.X_pending if q > 1 else None
    base_X_avoid = X_avoid
    tkwargs = {"device": discrete_choices[0].device, "dtype": discrete_choices[0].dtype}
    dim = len(discrete_choices)
    if X_avoid is None:
        X_avoid = torch.zeros(0, dim, **tkwargs)

    inequality_constraints = inequality_constraints or []
    for i in range(q):
        # generate some starting points
        if i == 0 and batch_initial_conditions is not None:
            X0 = _filter_invalid(X=batch_initial_conditions.squeeze(1), X_avoid=X_avoid)
            X0 = _filter_infeasible(
                X=X0, inequality_constraints=inequality_constraints
            ).unsqueeze(1)
        else:
            X_init = _gen_batch_initial_conditions_local_search(
                discrete_choices=discrete_choices,
                raw_samples=raw_samples,
                X_avoid=X_avoid,
                inequality_constraints=inequality_constraints,
                min_points=num_restarts,
            )
            # pick the best starting points
            with torch.no_grad():
                acqvals_init = _split_batch_eval_acqf(
                    acq_function=acq_function,
                    X=X_init.unsqueeze(1),
                    max_batch_size=max_batch_size,
                ).unsqueeze(-1)
            X0 = X_init[acqvals_init.topk(k=num_restarts, largest=True, dim=0).indices]

        # optimize from the best starting points
        best_xs = torch.zeros(len(X0), dim, **tkwargs)
        best_acqvals = torch.zeros(len(X0), 1, **tkwargs)
        for j, x in enumerate(X0):
            curr_x, curr_acqval = x.clone(), acq_function(x.unsqueeze(1))
            while True:
                # this generates all feasible neighbors that are one bit away
                X_loc = _generate_neighbors(
                    x=curr_x,
                    discrete_choices=discrete_choices,
                    X_avoid=X_avoid,
                    inequality_constraints=inequality_constraints,
                )
                # there may not be any neighbors
                if len(X_loc) == 0:
                    break
                with torch.no_grad():
                    acqval_loc = acq_function(X_loc.unsqueeze(1))
                # break if no neighbor is better than the current point (local optimum)
                if acqval_loc.max() <= curr_acqval:
                    break
                best_ind = acqval_loc.argmax().item()
                curr_x, curr_acqval = X_loc[best_ind].unsqueeze(0), acqval_loc[best_ind]
            best_xs[j, :], best_acqvals[j] = curr_x, curr_acqval

        # pick the best
        best_idx = best_acqvals.argmax()
        candidate_list.append(best_xs[best_idx].unsqueeze(0))

        # set pending points
        candidates = torch.cat(candidate_list, dim=-2)
        if q > 1:
            acq_function.set_X_pending(
                torch.cat([base_X_pending, candidates], dim=-2)
                if base_X_pending is not None
                else candidates
            )

            # Update points to avoid if unique is True
            if unique:
                X_avoid = (
                    torch.cat([base_X_avoid, candidates], dim=-2)
                    if base_X_avoid is not None
                    else candidates
                )

    # Reset acq_func to original X_pending state
    if q > 1:
        acq_function.set_X_pending(base_X_pending)
    with torch.no_grad():
        acq_value = acq_function(candidates)  # compute joint acquisition value
    return candidates, acq_value
