#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Methods for optimizing acquisition functions.
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import Any, Callable, Optional, Union

import torch
from botorch.acquisition.acquisition import (
    AcquisitionFunction,
    OneShotAcquisitionFunction,
)
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.acquisition.multi_objective.hypervolume_knowledge_gradient import (
    qHypervolumeKnowledgeGradient,
)
from botorch.exceptions import InputDataError, UnsupportedError
from botorch.exceptions.warnings import OptimizationWarning
from botorch.generation.gen import gen_candidates_scipy, TGenCandidates
from botorch.logging import logger
from botorch.optim.initializers import (
    gen_batch_initial_conditions,
    gen_one_shot_hvkg_initial_conditions,
    gen_one_shot_kg_initial_conditions,
    TGenInitialConditions,
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
    "seed",
    "thinning",
}


@dataclasses.dataclass(frozen=True)
class OptimizeAcqfInputs:
    """
    Container for inputs to `optimize_acqf`.

    See docstring for `optimize_acqf` for explanation of parameters.
    """

    acq_function: AcquisitionFunction
    bounds: Tensor
    q: int
    num_restarts: int
    raw_samples: Optional[int]
    options: Optional[dict[str, Union[bool, float, int, str]]]
    inequality_constraints: Optional[list[tuple[Tensor, Tensor, float]]]
    equality_constraints: Optional[list[tuple[Tensor, Tensor, float]]]
    nonlinear_inequality_constraints: Optional[list[tuple[Callable, bool]]]
    fixed_features: Optional[dict[int, float]]
    post_processing_func: Optional[Callable[[Tensor], Tensor]]
    batch_initial_conditions: Optional[Tensor]
    return_best_only: bool
    gen_candidates: TGenCandidates
    sequential: bool
    ic_generator: Optional[TGenInitialConditions] = None
    timeout_sec: Optional[float] = None
    return_full_tree: bool = False
    retry_on_optimization_warning: bool = True
    ic_gen_kwargs: dict = dataclasses.field(default_factory=dict)

    @property
    def full_tree(self) -> bool:
        return self.return_full_tree or (
            not isinstance(self.acq_function, OneShotAcquisitionFunction)
        )

    def __post_init__(self) -> None:
        if self.inequality_constraints is None and not (
            self.bounds.ndim == 2 and self.bounds.shape[0] == 2
        ):
            raise ValueError(
                "bounds should be a `2 x d` tensor, current shape: "
                f"{list(self.bounds.shape)}."
            )

        d = self.bounds.shape[1]
        if self.batch_initial_conditions is not None:
            batch_initial_conditions_shape = self.batch_initial_conditions.shape
            if len(batch_initial_conditions_shape) not in (2, 3):
                raise ValueError(
                    "batch_initial_conditions must be 2-dimensional or "
                    "3-dimensional. Its shape is "
                    f"{batch_initial_conditions_shape}."
                )
            if batch_initial_conditions_shape[-1] != d:
                raise ValueError(
                    f"batch_initial_conditions.shape[-1] must be {d}. The "
                    f"shape is {batch_initial_conditions_shape}."
                )

        elif self.ic_generator is None:
            if self.nonlinear_inequality_constraints is not None:
                raise RuntimeError(
                    "`ic_generator` must be given if "
                    "there are non-linear inequality constraints."
                )
            if self.raw_samples is None:
                raise ValueError(
                    "Must specify `raw_samples` when "
                    "`batch_initial_conditions` is None`."
                )

    def get_ic_generator(self) -> TGenInitialConditions:
        if self.ic_generator is not None:
            return self.ic_generator
        elif isinstance(self.acq_function, qKnowledgeGradient):
            return gen_one_shot_kg_initial_conditions
        elif isinstance(self.acq_function, qHypervolumeKnowledgeGradient):
            return gen_one_shot_hvkg_initial_conditions
        return gen_batch_initial_conditions


def _optimize_acqf_all_features_fixed(
    *,
    bounds: Tensor,
    fixed_features: dict[int, float],
    q: int,
    acq_function: AcquisitionFunction,
) -> tuple[Tensor, Tensor]:
    """
    Helper function for `optimize_acqf` for the trivial case where
    all features are fixed.
    """
    X = torch.tensor(
        [fixed_features[i] for i in range(bounds.shape[-1])],
        device=bounds.device,
        dtype=bounds.dtype,
    )
    X = X.expand(q, *X.shape)
    with torch.no_grad():
        acq_value = acq_function(X)
    return X, acq_value


def _validate_sequential_inputs(opt_inputs: OptimizeAcqfInputs) -> None:
    # Validate that constraints across the q-dim and
    # self.sequential are not present together.
    const_err_message = (
        "Inter-point constraints are not supported for sequential optimization. "
        "But the {}th {} constraint is defined as inter-point."
    )
    if opt_inputs.inequality_constraints is not None:
        for i, constraint in enumerate(opt_inputs.inequality_constraints):
            if len(constraint[0].shape) > 1:
                raise UnsupportedError(const_err_message.format(i, "linear inequality"))
    if opt_inputs.equality_constraints is not None:
        for i, constraint in enumerate(opt_inputs.equality_constraints):
            if len(constraint[0].shape) > 1:
                raise UnsupportedError(const_err_message.format(i, "linear equality"))
    if opt_inputs.nonlinear_inequality_constraints is not None:
        for i, (_, intra_point) in enumerate(
            opt_inputs.nonlinear_inequality_constraints
        ):
            if not intra_point:
                raise UnsupportedError(
                    const_err_message.format(i, "non-linear inequality")
                )

    # TODO: Validate constraints if provided:
    # https://github.com/pytorch/botorch/pull/1231
    if opt_inputs.batch_initial_conditions is not None:
        raise UnsupportedError(
            "`batch_initial_conditions` is not supported for sequential "
            "optimization. Either avoid specifying "
            "`batch_initial_conditions` to use the custom initializer or "
            "use the `ic_generator` kwarg to generate initial conditions "
            "for the case of nonlinear inequality constraints."
        )

    if not opt_inputs.return_best_only:
        raise NotImplementedError(
            "`return_best_only=False` only supported for joint optimization."
        )
    if isinstance(opt_inputs.acq_function, OneShotAcquisitionFunction):
        raise NotImplementedError(
            "sequential optimization currently not supported for one-shot "
            "acquisition functions. Must have `sequential=False`."
        )


def _optimize_acqf_sequential_q(
    opt_inputs: OptimizeAcqfInputs,
) -> tuple[Tensor, Tensor]:
    """
    Helper function for `optimize_acqf` when sequential=True and q > 1.

    For each of `q` times, generate a single candidate greedily, then add it to
    the list of pending points.
    """
    _validate_sequential_inputs(opt_inputs)
    # When using sequential optimization, we allocate the total timeout
    # evenly across the individual acquisition optimizations.
    timeout_sec = (
        opt_inputs.timeout_sec / opt_inputs.q
        if opt_inputs.timeout_sec is not None
        else None
    )
    candidate_list, acq_value_list = [], []
    base_X_pending = opt_inputs.acq_function.X_pending

    new_inputs = dataclasses.replace(
        opt_inputs,
        q=1,
        batch_initial_conditions=None,
        return_best_only=True,
        sequential=False,
        timeout_sec=timeout_sec,
    )
    for i in range(opt_inputs.q):
        candidate, acq_value = _optimize_acqf_batch(new_inputs)

        candidate_list.append(candidate)
        acq_value_list.append(acq_value)
        candidates = torch.cat(candidate_list, dim=-2)
        new_inputs.acq_function.set_X_pending(
            torch.cat([base_X_pending, candidates], dim=-2)
            if base_X_pending is not None
            else candidates
        )
        logger.info(f"Generated sequential candidate {i+1} of {opt_inputs.q}")
    opt_inputs.acq_function.set_X_pending(base_X_pending)
    return candidates, torch.stack(acq_value_list)


def _optimize_acqf_batch(opt_inputs: OptimizeAcqfInputs) -> tuple[Tensor, Tensor]:
    options = opt_inputs.options or {}

    initial_conditions_provided = opt_inputs.batch_initial_conditions is not None

    if initial_conditions_provided:
        batch_initial_conditions = opt_inputs.batch_initial_conditions
    else:
        # pyre-ignore[28]: Unexpected keyword argument `acq_function` to anonymous call.
        batch_initial_conditions = opt_inputs.get_ic_generator()(
            acq_function=opt_inputs.acq_function,
            bounds=opt_inputs.bounds,
            q=opt_inputs.q,
            num_restarts=opt_inputs.num_restarts,
            raw_samples=opt_inputs.raw_samples,
            fixed_features=opt_inputs.fixed_features,
            options=options,
            inequality_constraints=opt_inputs.inequality_constraints,
            equality_constraints=opt_inputs.equality_constraints,
            **opt_inputs.ic_gen_kwargs,
        )

    batch_limit: int = options.get(
        "batch_limit",
        (
            opt_inputs.num_restarts
            if not opt_inputs.nonlinear_inequality_constraints
            else 1
        ),
    )

    def _optimize_batch_candidates() -> tuple[Tensor, Tensor, list[Warning]]:
        batch_candidates_list: list[Tensor] = []
        batch_acq_values_list: list[Tensor] = []
        batched_ics = batch_initial_conditions.split(batch_limit)
        opt_warnings = []
        timeout_sec = (
            opt_inputs.timeout_sec / len(batched_ics)
            if opt_inputs.timeout_sec is not None
            else None
        )

        bounds = opt_inputs.bounds
        gen_kwargs: dict[str, Any] = {
            "lower_bounds": None if bounds[0].isinf().all() else bounds[0],
            "upper_bounds": None if bounds[1].isinf().all() else bounds[1],
            "options": {k: v for k, v in options.items() if k not in INIT_OPTION_KEYS},
            "fixed_features": opt_inputs.fixed_features,
            "timeout_sec": timeout_sec,
        }

        for constraint_name in [
            "inequality_constraints",
            "equality_constraints",
            "nonlinear_inequality_constraints",
        ]:
            if (constraint := getattr(opt_inputs, constraint_name)) is not None:
                gen_kwargs[constraint_name] = constraint

        for i, batched_ics_ in enumerate(batched_ics):
            # optimize using random restart optimization
            with warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter("always", category=OptimizationWarning)
                (
                    batch_candidates_curr,
                    batch_acq_values_curr,
                ) = opt_inputs.gen_candidates(
                    batched_ics_, opt_inputs.acq_function, **gen_kwargs
                )
            opt_warnings += ws
            batch_candidates_list.append(batch_candidates_curr)
            batch_acq_values_list.append(batch_acq_values_curr)
            logger.info(f"Generated candidate batch {i+1} of {len(batched_ics)}.")

        batch_candidates = torch.cat(batch_candidates_list)
        has_scalars = batch_acq_values_list[0].ndim == 0
        if has_scalars:
            batch_acq_values = torch.stack(batch_acq_values_list)
        else:
            batch_acq_values = torch.cat(batch_acq_values_list).flatten()
        return batch_candidates, batch_acq_values, opt_warnings

    batch_candidates, batch_acq_values, ws = _optimize_batch_candidates()

    optimization_warning_raised = any(
        (issubclass(w.category, OptimizationWarning) for w in ws)
    )
    if optimization_warning_raised and opt_inputs.retry_on_optimization_warning:
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
        warnings.warn(first_warn_msg, RuntimeWarning, stacklevel=2)

        if not initial_conditions_provided:
            batch_initial_conditions = opt_inputs.get_ic_generator()(
                acq_function=opt_inputs.acq_function,
                bounds=opt_inputs.bounds,
                q=opt_inputs.q,
                num_restarts=opt_inputs.num_restarts,
                raw_samples=opt_inputs.raw_samples,
                fixed_features=opt_inputs.fixed_features,
                options=options,
                inequality_constraints=opt_inputs.inequality_constraints,
                equality_constraints=opt_inputs.equality_constraints,
                **opt_inputs.ic_gen_kwargs,
            )

            batch_candidates, batch_acq_values, ws = _optimize_batch_candidates()

            optimization_warning_raised = any(
                (issubclass(w.category, OptimizationWarning) for w in ws)
            )
            if optimization_warning_raised:
                warnings.warn(
                    "Optimization failed on the second try, after generating a "
                    "new set of initial conditions.",
                    RuntimeWarning,
                    stacklevel=2,
                )

    if opt_inputs.post_processing_func is not None:
        batch_candidates = opt_inputs.post_processing_func(batch_candidates)
        with torch.no_grad():
            acq_values_list = [
                opt_inputs.acq_function(cand)
                for cand in batch_candidates.split(batch_limit, dim=0)
            ]
            batch_acq_values = torch.cat(acq_values_list, dim=0)

    if opt_inputs.return_best_only:
        best = torch.argmax(batch_acq_values.view(-1), dim=0)
        batch_candidates = batch_candidates[best]
        batch_acq_values = batch_acq_values[best]

    if not opt_inputs.full_tree:
        batch_candidates = opt_inputs.acq_function.extract_candidates(
            X_full=batch_candidates
        )

    return batch_candidates, batch_acq_values


def optimize_acqf(
    acq_function: AcquisitionFunction,
    bounds: Tensor,
    q: int,
    num_restarts: int,
    raw_samples: Optional[int] = None,
    options: Optional[dict[str, Union[bool, float, int, str]]] = None,
    inequality_constraints: Optional[list[tuple[Tensor, Tensor, float]]] = None,
    equality_constraints: Optional[list[tuple[Tensor, Tensor, float]]] = None,
    nonlinear_inequality_constraints: Optional[list[tuple[Callable, bool]]] = None,
    fixed_features: Optional[dict[int, float]] = None,
    post_processing_func: Optional[Callable[[Tensor], Tensor]] = None,
    batch_initial_conditions: Optional[Tensor] = None,
    return_best_only: bool = True,
    gen_candidates: Optional[TGenCandidates] = None,
    sequential: bool = False,
    *,
    ic_generator: Optional[TGenInitialConditions] = None,
    timeout_sec: Optional[float] = None,
    return_full_tree: bool = False,
    retry_on_optimization_warning: bool = True,
    **ic_gen_kwargs: Any,
) -> tuple[Tensor, Tensor]:
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
            `\sum_i (X[indices[i]] * coefficients[i]) >= rhs`. `indices` and
            `coefficients` should be torch tensors. See the docstring of
            `make_scipy_linear_constraints` for an example. When q=1, or when
            applying the same constraint to each candidate in the batch
            (intra-point constraint), `indices` should be a 1-d tensor.
            For inter-point constraints, in which the constraint is applied to the
            whole batch of candidates, `indices` must be a 2-d tensor, where
            in each row `indices[i] =(k_i, l_i)` the first index `k_i` corresponds
            to the `k_i`-th element of the `q`-batch and the second index `l_i`
            corresponds to the `l_i`-th feature of that element.
        equality_constraints: A list of tuples (indices, coefficients, rhs),
            with each tuple encoding an equality constraint of the form
            `\sum_i (X[indices[i]] * coefficients[i]) = rhs`. See the docstring of
            `make_scipy_linear_constraints` for an example.
        nonlinear_inequality_constraints: A list of tuples representing the nonlinear
            inequality constraints. The first element in the tuple is a callable
            representing a constraint of the form `callable(x) >= 0`. In case of an
            intra-point constraint, `callable()`takes in an one-dimensional tensor of
            shape `d` and returns a scalar. In case of an inter-point constraint,
            `callable()` takes a two dimensional tensor of shape `q x d` and again
            returns a scalar. The second element is a boolean, indicating if it is an
            intra-point or inter-point constraint (`True` for intra-point. `False` for
            inter-point). For more information on intra-point vs inter-point
            constraints, see the docstring of the `inequality_constraints` argument to
            `optimize_acqf()`. The constraints will later be passed to the scipy
            solver. You need to pass in `batch_initial_conditions` in this case.
            Using non-linear inequality constraints also requires that `batch_limit`
            is set to 1, which will be done automatically if not specified in
            `options`.
        fixed_features: A map `{feature_index: value}` for features that
            should be fixed to a particular value during generation.
        post_processing_func: A function that post-processes an optimization
            result appropriately (i.e., according to `round-trip`
            transformations).
        batch_initial_conditions: A tensor to specify the initial conditions. Set
            this if you do not want to use default initialization strategy.
        return_best_only: If False, outputs the solutions corresponding to all
            random restart initializations of the optimization.
        gen_candidates: A callable for generating candidates (and their associated
            acquisition values) given a tensor of initial conditions and an
            acquisition function. Other common inputs include lower and upper bounds
            and a dictionary of options, but refer to the documentation of specific
            generation functions (e.g gen_candidates_scipy and gen_candidates_torch)
            for method-specific inputs. Default: `gen_candidates_scipy`
        sequential: If False, uses joint optimization, otherwise uses sequential
            optimization.
        ic_generator: Function for generating initial conditions. Not needed when
            `batch_initial_conditions` are provided. Defaults to
            `gen_one_shot_kg_initial_conditions` for `qKnowledgeGradient` acquisition
            functions and `gen_batch_initial_conditions` otherwise. Must be specified
            for nonlinear inequality constraints.
        timeout_sec: Max amount of time optimization can run for.
        return_full_tree:
        retry_on_optimization_warning: Whether to retry candidate generation with a new
            set of initial conditions when it fails with an `OptimizationWarning`.
        ic_gen_kwargs: Additional keyword arguments passed to function specified by
            `ic_generator`

    Returns:
        A two-element tuple containing

        - A tensor of generated candidates. The shape is
            -- `q x d` if `return_best_only` is True (default)
            -- `num_restarts x q x d` if `return_best_only` is False
        - a tensor of associated acquisition values. If `sequential=False`,
            this is a `(num_restarts)`-dim tensor of joint acquisition values
            (with explicit restart dimension if `return_best_only=False`). If
            `sequential=True`, this is a `q`-dim tensor of expected acquisition
            values conditional on having observed candidates `0,1,...,i-1`.

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
    # using a default of None simplifies unit testing
    if gen_candidates is None:
        gen_candidates = gen_candidates_scipy
    opt_acqf_inputs = OptimizeAcqfInputs(
        acq_function=acq_function,
        bounds=bounds,
        q=q,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        options=options,
        inequality_constraints=inequality_constraints,
        equality_constraints=equality_constraints,
        nonlinear_inequality_constraints=nonlinear_inequality_constraints,
        fixed_features=fixed_features,
        post_processing_func=post_processing_func,
        batch_initial_conditions=batch_initial_conditions,
        return_best_only=return_best_only,
        gen_candidates=gen_candidates,
        sequential=sequential,
        ic_generator=ic_generator,
        timeout_sec=timeout_sec,
        return_full_tree=return_full_tree,
        retry_on_optimization_warning=retry_on_optimization_warning,
        ic_gen_kwargs=ic_gen_kwargs,
    )
    return _optimize_acqf(opt_acqf_inputs)


def _optimize_acqf(opt_inputs: OptimizeAcqfInputs) -> tuple[Tensor, Tensor]:
    # Handle the trivial case when all features are fixed
    if (
        opt_inputs.fixed_features is not None
        and len(opt_inputs.fixed_features) == opt_inputs.bounds.shape[-1]
    ):
        return _optimize_acqf_all_features_fixed(
            bounds=opt_inputs.bounds,
            fixed_features=opt_inputs.fixed_features,
            q=opt_inputs.q,
            acq_function=opt_inputs.acq_function,
        )

    # Perform sequential optimization via successive conditioning on pending points
    if opt_inputs.sequential and opt_inputs.q > 1:
        return _optimize_acqf_sequential_q(opt_inputs=opt_inputs)

    # Batch optimization (including the case q=1)
    return _optimize_acqf_batch(opt_inputs=opt_inputs)


def optimize_acqf_cyclic(
    acq_function: AcquisitionFunction,
    bounds: Tensor,
    q: int,
    num_restarts: int,
    raw_samples: Optional[int] = None,
    options: Optional[dict[str, Union[bool, float, int, str]]] = None,
    inequality_constraints: Optional[list[tuple[Tensor, Tensor, float]]] = None,
    equality_constraints: Optional[list[tuple[Tensor, Tensor, float]]] = None,
    fixed_features: Optional[dict[int, float]] = None,
    post_processing_func: Optional[Callable[[Tensor], Tensor]] = None,
    batch_initial_conditions: Optional[Tensor] = None,
    cyclic_options: Optional[dict[str, Union[bool, float, int, str]]] = None,
    *,
    ic_generator: Optional[TGenInitialConditions] = None,
    timeout_sec: Optional[float] = None,
    return_full_tree: bool = False,
    retry_on_optimization_warning: bool = True,
    **ic_gen_kwargs: Any,
) -> tuple[Tensor, Tensor]:
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
        ic_generator: Function for generating initial conditions. Not needed when
            `batch_initial_conditions` are provided. Defaults to
            `gen_one_shot_kg_initial_conditions` for `qKnowledgeGradient` acquisition
            functions and `gen_batch_initial_conditions` otherwise. Must be specified
            for nonlinear inequality constraints.
        timeout_sec: Max amount of time optimization can run for.
        return_full_tree:
        retry_on_optimization_warning: Whether to retry candidate generation with a new
            set of initial conditions when it fails with an `OptimizationWarning`.
        ic_gen_kwargs: Additional keyword arguments passed to function specified by
            `ic_generator`

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
    opt_inputs = OptimizeAcqfInputs(
        acq_function=acq_function,
        bounds=bounds,
        q=q,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        options=options,
        inequality_constraints=inequality_constraints,
        equality_constraints=equality_constraints,
        nonlinear_inequality_constraints=None,
        fixed_features=fixed_features,
        post_processing_func=post_processing_func,
        batch_initial_conditions=batch_initial_conditions,
        return_best_only=True,
        gen_candidates=gen_candidates_scipy,
        sequential=True,
        ic_generator=ic_generator,
        timeout_sec=timeout_sec,
        return_full_tree=return_full_tree,
        retry_on_optimization_warning=retry_on_optimization_warning,
        ic_gen_kwargs=ic_gen_kwargs,
    )

    # for the first cycle, optimize the q candidates sequentially
    candidates, acq_vals = _optimize_acqf(opt_inputs)
    q = opt_inputs.q
    opt_inputs = dataclasses.replace(opt_inputs, q=1)
    acq_function = opt_inputs.acq_function

    if q > 1:
        cyclic_options = cyclic_options or {}
        stopping_criterion = ExpMAStoppingCriterion(**cyclic_options)
        stop = stopping_criterion.evaluate(fvals=acq_vals)
        base_X_pending = acq_function.X_pending
        idxr = torch.ones(q, dtype=torch.bool, device=opt_inputs.bounds.device)
        while not stop:
            for i in range(q):
                # optimize only candidate i
                idxr[i] = 0
                acq_function.set_X_pending(
                    torch.cat([base_X_pending, candidates[idxr]], dim=-2)
                    if base_X_pending is not None
                    else candidates[idxr]
                )
                opt_inputs = dataclasses.replace(
                    opt_inputs,
                    batch_initial_conditions=candidates[i].unsqueeze(0),
                    sequential=False,
                )
                candidate_i, acq_val_i = _optimize_acqf(opt_inputs)
                candidates[i] = candidate_i
                acq_vals[i] = acq_val_i
                idxr[i] = 1
            stop = stopping_criterion.evaluate(fvals=acq_vals)
        acq_function.set_X_pending(base_X_pending)
    return candidates, acq_vals


def optimize_acqf_list(
    acq_function_list: list[AcquisitionFunction],
    bounds: Tensor,
    num_restarts: int,
    raw_samples: Optional[int] = None,
    options: Optional[dict[str, Union[bool, float, int, str]]] = None,
    inequality_constraints: Optional[list[tuple[Tensor, Tensor, float]]] = None,
    equality_constraints: Optional[list[tuple[Tensor, Tensor, float]]] = None,
    nonlinear_inequality_constraints: Optional[list[tuple[Callable, bool]]] = None,
    fixed_features: Optional[dict[int, float]] = None,
    fixed_features_list: Optional[list[dict[int, float]]] = None,
    post_processing_func: Optional[Callable[[Tensor], Tensor]] = None,
    ic_generator: Optional[TGenInitialConditions] = None,
    ic_gen_kwargs: Optional[dict] = None,
) -> tuple[Tensor, Tensor]:
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
        nonlinear_inequality_constraints: A list of tuples representing the nonlinear
            inequality constraints. The first element in the tuple is a callable
            representing a constraint of the form `callable(x) >= 0`. In case of an
            intra-point constraint, `callable()`takes in an one-dimensional tensor of
            shape `d` and returns a scalar. In case of an inter-point constraint,
            `callable()` takes a two dimensional tensor of shape `q x d` and again
            returns a scalar. The second element is a boolean, indicating if it is an
            intra-point or inter-point constraint (`True` for intra-point. `False` for
            inter-point). For more information on intra-point vs inter-point
            constraints, see the docstring of the `inequality_constraints` argument to
            `optimize_acqf()`. The constraints will later be passed to the scipy
            solver. You need to pass in `batch_initial_conditions` in this case.
            Using non-linear inequality constraints also requires that `batch_limit`
            is set to 1, which will be done automatically if not specified in
            `options`.
        fixed_features: A map `{feature_index: value}` for features that
            should be fixed to a particular value during generation.
        fixed_features_list: A list of maps `{feature_index: value}`. The i-th
            item represents the fixed_feature for the i-th optimization. If
            `fixed_features_list` is provided, `optimize_acqf_mixed` is invoked.
        post_processing_func: A function that post-processes an optimization
            result appropriately (i.e., according to `round-trip`
            transformations).
        ic_generator: Function for generating initial conditions. Not needed when
            `batch_initial_conditions` are provided. Defaults to
            `gen_one_shot_kg_initial_conditions` for `qKnowledgeGradient` acquisition
            functions and `gen_batch_initial_conditions` otherwise. Must be specified
            for nonlinear inequality constraints.
        ic_gen_kwargs: Additional keyword arguments passed to function specified by
            `ic_generator`

    Returns:
        A two-element tuple containing

        - a `q x d`-dim tensor of generated candidates.
        - a `q`-dim tensor of expected acquisition values, where the value at
            index `i` is the acquisition value conditional on having observed
            all candidates except candidate `i`.
    """
    if fixed_features and fixed_features_list:
        raise ValueError(
            "Èither `fixed_feature` or `fixed_features_list` can be provided, not both."
        )
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
        if fixed_features_list:
            candidate, acq_value = optimize_acqf_mixed(
                acq_function=acq_function,
                bounds=bounds,
                q=1,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                options=options or {},
                inequality_constraints=inequality_constraints,
                equality_constraints=equality_constraints,
                nonlinear_inequality_constraints=nonlinear_inequality_constraints,
                fixed_features_list=fixed_features_list,
                post_processing_func=post_processing_func,
                ic_generator=ic_generator,
                ic_gen_kwargs=ic_gen_kwargs,
            )
        else:
            ic_gen_kwargs = ic_gen_kwargs or {}
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
                return_best_only=True,
                sequential=False,
                ic_generator=ic_generator,
                **ic_gen_kwargs,
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
    fixed_features_list: list[dict[int, float]],
    raw_samples: Optional[int] = None,
    options: Optional[dict[str, Union[bool, float, int, str]]] = None,
    inequality_constraints: Optional[list[tuple[Tensor, Tensor, float]]] = None,
    equality_constraints: Optional[list[tuple[Tensor, Tensor, float]]] = None,
    nonlinear_inequality_constraints: Optional[list[tuple[Callable, bool]]] = None,
    post_processing_func: Optional[Callable[[Tensor], Tensor]] = None,
    batch_initial_conditions: Optional[Tensor] = None,
    ic_generator: Optional[TGenInitialConditions] = None,
    ic_gen_kwargs: Optional[dict] = None,
) -> tuple[Tensor, Tensor]:
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
        nonlinear_inequality_constraints: A list of tuples representing the nonlinear
            inequality constraints. The first element in the tuple is a callable
            representing a constraint of the form `callable(x) >= 0`. In case of an
            intra-point constraint, `callable()`takes in an one-dimensional tensor of
            shape `d` and returns a scalar. In case of an inter-point constraint,
            `callable()` takes a two dimensional tensor of shape `q x d` and again
            returns a scalar. The second element is a boolean, indicating if it is an
            intra-point or inter-point constraint (`True` for intra-point. `False` for
            inter-point). For more information on intra-point vs inter-point
            constraints, see the docstring of the `inequality_constraints` argument to
            `optimize_acqf()`. The constraints will later be passed to the scipy
            solver. You need to pass in `batch_initial_conditions` in this case.
            Using non-linear inequality constraints also requires that `batch_limit`
            is set to 1, which will be done automatically if not specified in
            `options`.
        post_processing_func: A function that post-processes an optimization
            result appropriately (i.e., according to `round-trip`
            transformations).
        batch_initial_conditions: A tensor to specify the initial conditions. Set
            this if you do not want to use default initialization strategy.
        ic_generator: Function for generating initial conditions. Not needed when
            `batch_initial_conditions` are provided. Defaults to
            `gen_one_shot_kg_initial_conditions` for `qKnowledgeGradient` acquisition
            functions and `gen_batch_initial_conditions` otherwise. Must be specified
            for nonlinear inequality constraints.
        ic_gen_kwargs: Additional keyword arguments passed to function specified by
            `ic_generator`

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

    ic_gen_kwargs = ic_gen_kwargs or {}

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
                nonlinear_inequality_constraints=nonlinear_inequality_constraints,
                fixed_features=fixed_features,
                post_processing_func=post_processing_func,
                batch_initial_conditions=batch_initial_conditions,
                ic_generator=ic_generator,
                return_best_only=True,
                **ic_gen_kwargs,
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
            nonlinear_inequality_constraints=nonlinear_inequality_constraints,
            post_processing_func=post_processing_func,
            batch_initial_conditions=batch_initial_conditions,
            ic_generator=ic_generator,
            ic_gen_kwargs=ic_gen_kwargs,
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
) -> tuple[Tensor, Tensor]:
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
        A two-element tuple containing

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
    discrete_choices: list[Tensor],
    X_avoid: Tensor,
    inequality_constraints: list[tuple[Tensor, Tensor, float]],
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
    X: Tensor, inequality_constraints: list[tuple[Tensor, Tensor, float]]
):
    """Remove all points from `X` that don't satisfy the constraints."""
    is_feasible = torch.ones(X.shape[0], dtype=torch.bool, device=X.device)
    for inds, weights, bound in inequality_constraints:
        is_feasible &= (X[..., inds] * weights).sum(dim=-1) >= bound
    return X[is_feasible]


def _filter_invalid(X: Tensor, X_avoid: Tensor):
    """Remove all occurences of `X_avoid` from `X`."""
    return X[~(X == X_avoid.unsqueeze(-2)).all(dim=-1).any(dim=-2)]


def _gen_batch_initial_conditions_local_search(
    discrete_choices: list[Tensor],
    raw_samples: int,
    X_avoid: Tensor,
    inequality_constraints: list[tuple[Tensor, Tensor, float]],
    min_points: int,
    max_tries: int = 100,
):
    """Generate initial conditions for local search."""
    device = discrete_choices[0].device
    dtype = discrete_choices[0].dtype
    dim = len(discrete_choices)
    X = torch.zeros(0, dim, device=device, dtype=dtype)
    for _ in range(max_tries):
        X_new = torch.zeros(raw_samples, dim, device=device, dtype=dtype)
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
    discrete_choices: list[Tensor],
    q: int,
    num_restarts: int = 20,
    raw_samples: int = 4096,
    inequality_constraints: Optional[list[tuple[Tensor, Tensor, float]]] = None,
    X_avoid: Optional[Tensor] = None,
    batch_initial_conditions: Optional[Tensor] = None,
    max_batch_size: int = 2048,
    unique: bool = True,
) -> tuple[Tensor, Tensor]:
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
    device = discrete_choices[0].device
    dtype = discrete_choices[0].dtype
    dim = len(discrete_choices)
    if X_avoid is None:
        X_avoid = torch.zeros(0, dim, device=device, dtype=dtype)

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
        best_xs = torch.zeros(len(X0), dim, device=device, dtype=dtype)
        best_acqvals = torch.zeros(len(X0), 1, device=device, dtype=dtype)
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
