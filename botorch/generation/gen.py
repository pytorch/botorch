#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Candidate generation utilities.
"""

from __future__ import annotations

import warnings
from functools import partial
from typing import Any, Callable, Dict, List, NoReturn, Optional, Tuple, Type, Union

import numpy as np
import torch
from botorch.acquisition import AcquisitionFunction
from botorch.exceptions.warnings import OptimizationWarning
from botorch.generation.utils import _remove_fixed_features_from_optimization
from botorch.optim.parameter_constraints import (
    _arrayify,
    make_scipy_bounds,
    make_scipy_linear_constraints,
    make_scipy_nonlinear_inequality_constraints,
    NLC_TOL,
)
from botorch.optim.stopping import ExpMAStoppingCriterion
from botorch.optim.utils import _filter_kwargs, columnwise_clamp, fix_features
from scipy.optimize import minimize
from torch import Tensor
from torch.optim import Optimizer


def gen_candidates_scipy(
    initial_conditions: Tensor,
    acquisition_function: AcquisitionFunction,
    lower_bounds: Optional[Union[float, Tensor]] = None,
    upper_bounds: Optional[Union[float, Tensor]] = None,
    inequality_constraints: Optional[List[Tuple[Tensor, Tensor, float]]] = None,
    equality_constraints: Optional[List[Tuple[Tensor, Tensor, float]]] = None,
    nonlinear_inequality_constraints: Optional[List[Callable]] = None,
    options: Optional[Dict[str, Any]] = None,
    fixed_features: Optional[Dict[int, Optional[float]]] = None,
) -> Tuple[Tensor, Tensor]:
    r"""Generate a set of candidates using `scipy.optimize.minimize`.

    Optimizes an acquisition function starting from a set of initial candidates
    using `scipy.optimize.minimize` via a numpy converter.

    Args:
        initial_conditions: Starting points for optimization.
        acquisition_function: Acquisition function to be used.
        lower_bounds: Minimum values for each column of initial_conditions.
        upper_bounds: Maximum values for each column of initial_conditions.
        inequality constraints: A list of tuples (indices, coefficients, rhs),
            with each tuple encoding an inequality constraint of the form
            `\sum_i (X[indices[i]] * coefficients[i]) >= rhs`.
        equality constraints: A list of tuples (indices, coefficients, rhs),
            with each tuple encoding an inequality constraint of the form
            `\sum_i (X[indices[i]] * coefficients[i]) = rhs`.
        nonlinear_inequality_constraints: A list of callables with that represent
            non-linear inequality constraints of the form `callable(x) >= 0`. Each
            callable is expected to take a `(num_restarts) x q x d`-dim tensor as
            an input and return a `(num_restarts) x q`-dim tensor with the
            constraint values. The constraints will later be passed to SLSQP.
        options: Options used to control the optimization including "method"
            and "maxiter". Select method for `scipy.minimize` using the
            "method" key. By default uses L-BFGS-B for box-constrained problems
            and SLSQP if inequality or equality constraints are present.
        fixed_features: This is a dictionary of feature indices to values, where
            all generated candidates will have features fixed to these values.
            If the dictionary value is None, then that feature will just be
            fixed to the clamped value and not optimized. Assumes values to be
            compatible with lower_bounds and upper_bounds!

    Returns:
        2-element tuple containing

        - The set of generated candidates.
        - The acquisition value for each t-batch.

    Example:
        >>> qEI = qExpectedImprovement(model, best_f=0.2)
        >>> bounds = torch.tensor([[0., 0.], [1., 2.]])
        >>> Xinit = gen_batch_initial_conditions(
        >>>     qEI, bounds, q=3, num_restarts=25, raw_samples=500
        >>> )
        >>> batch_candidates, batch_acq_values = gen_candidates_scipy(
                initial_conditions=Xinit,
                acquisition_function=qEI,
                lower_bounds=bounds[0],
                upper_bounds=bounds[1],
            )
    """
    options = options or {}

    # if there are fixed features we may optimize over a domain of lower dimension
    reduced_domain = False
    if fixed_features:
        # TODO: We can support fixed features, see Max's comment on D33551393. We can
        # consider adding this at a later point.
        if nonlinear_inequality_constraints:
            raise NotImplementedError(
                "Fixed features are not supported when non-linear inequality "
                "constraints are given."
            )
        # if there are no constraints things are straightforward
        if not (inequality_constraints or equality_constraints):
            reduced_domain = True
        # if there are we need to make sure features are fixed to specific values
        else:
            reduced_domain = None not in fixed_features.values()

    if reduced_domain:
        _no_fixed_features = _remove_fixed_features_from_optimization(
            fixed_features=fixed_features,
            acquisition_function=acquisition_function,
            initial_conditions=initial_conditions,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            inequality_constraints=inequality_constraints,
            equality_constraints=equality_constraints,
        )
        # call the routine with no fixed_features
        clamped_candidates, batch_acquisition = gen_candidates_scipy(
            initial_conditions=_no_fixed_features.initial_conditions,
            acquisition_function=_no_fixed_features.acquisition_function,
            lower_bounds=_no_fixed_features.lower_bounds,
            upper_bounds=_no_fixed_features.upper_bounds,
            inequality_constraints=_no_fixed_features.inequality_constraints,
            equality_constraints=_no_fixed_features.equality_constraints,
            options=options,
            fixed_features=None,
        )
        clamped_candidates = _no_fixed_features.acquisition_function._construct_X_full(
            clamped_candidates
        )
        return clamped_candidates, batch_acquisition

    clamped_candidates = columnwise_clamp(
        X=initial_conditions, lower=lower_bounds, upper=upper_bounds
    )

    shapeX = clamped_candidates.shape
    x0 = clamped_candidates.view(-1)
    bounds = make_scipy_bounds(
        X=initial_conditions, lower_bounds=lower_bounds, upper_bounds=upper_bounds
    )
    constraints = make_scipy_linear_constraints(
        shapeX=clamped_candidates.shape,
        inequality_constraints=inequality_constraints,
        equality_constraints=equality_constraints,
    )

    def f_np_wrapper(x: np.ndarray, f: Callable):
        """Given a torch callable, compute value + grad given a numpy array."""
        if np.isnan(x).any():
            raise RuntimeError(
                f"{np.isnan(x).sum()} elements of the {x.size} element array "
                f"`x` are NaN."
            )
        X = (
            torch.from_numpy(x)
            .to(initial_conditions)
            .view(shapeX)
            .contiguous()
            .requires_grad_(True)
        )
        X_fix = fix_features(X, fixed_features=fixed_features)
        loss = f(X_fix).sum()
        # compute gradient w.r.t. the inputs (does not accumulate in leaves)
        gradf = _arrayify(torch.autograd.grad(loss, X)[0].contiguous().view(-1))
        if np.isnan(gradf).any():
            msg = (
                f"{np.isnan(gradf).sum()} elements of the {x.size} element "
                "gradient array `gradf` are NaN. This often indicates numerical issues."
            )
            if initial_conditions.dtype != torch.double:
                msg += " Consider using `dtype=torch.double`."
            raise RuntimeError(msg)
        fval = loss.item()
        return fval, gradf

    if nonlinear_inequality_constraints:
        # Make sure `batch_limit` is 1 for now.
        if not (len(shapeX) == 3 and shapeX[:2] == torch.Size([1, 1])):
            raise ValueError(
                "`batch_limit` must be 1 when non-linear inequality constraints "
                "are given."
            )
        constraints += make_scipy_nonlinear_inequality_constraints(
            nonlinear_inequality_constraints=nonlinear_inequality_constraints,
            f_np_wrapper=f_np_wrapper,
            x0=x0,
        )
    x0 = _arrayify(x0)

    def f(x):
        return -acquisition_function(x)

    res = minimize(
        fun=f_np_wrapper,
        args=(f,),
        x0=x0,
        method=options.get("method", "SLSQP" if constraints else "L-BFGS-B"),
        jac=True,
        bounds=bounds,
        constraints=constraints,
        callback=options.get("callback", None),
        options={k: v for k, v in options.items() if k not in ["method", "callback"]},
    )

    if "success" not in res.keys() or "status" not in res.keys():
        with warnings.catch_warnings():
            warnings.simplefilter("always", category=OptimizationWarning)
            warnings.warn(
                "Optimization failed within `scipy.optimize.minimize` with no "
                "status returned to `res.`",
                OptimizationWarning,
            )
    elif not res.success:
        with warnings.catch_warnings():
            warnings.simplefilter("always", category=OptimizationWarning)
            warnings.warn(
                f"Optimization failed within `scipy.optimize.minimize` with status "
                f"{res.status}.",
                OptimizationWarning,
            )
    candidates = fix_features(
        X=torch.from_numpy(res.x).to(initial_conditions).reshape(shapeX),
        fixed_features=fixed_features,
    )

    # SLSQP sometimes fails in the line search or may just fail to find a feasible
    # candidate in which case we just return the starting point. This happens rarely,
    # so it shouldn't be an issue given enough restarts.
    if nonlinear_inequality_constraints and any(
        nlc(candidates.view(-1)) < NLC_TOL for nlc in nonlinear_inequality_constraints
    ):
        candidates = torch.from_numpy(x0).to(candidates).reshape(shapeX)
        warnings.warn(
            "SLSQP failed to converge to a solution the satisfies the non-linear "
            "constraints. Returning the feasible starting point."
        )

    clamped_candidates = columnwise_clamp(
        X=candidates, lower=lower_bounds, upper=upper_bounds, raise_on_violation=True
    )
    with torch.no_grad():
        batch_acquisition = acquisition_function(clamped_candidates)

    return clamped_candidates, batch_acquisition


def gen_candidates_torch(
    initial_conditions: Tensor,
    acquisition_function: AcquisitionFunction,
    lower_bounds: Optional[Union[float, Tensor]] = None,
    upper_bounds: Optional[Union[float, Tensor]] = None,
    optimizer: Type[Optimizer] = torch.optim.Adam,
    options: Optional[Dict[str, Union[float, str]]] = None,
    callback: Optional[Callable[[int, Tensor, Tensor], NoReturn]] = None,
    fixed_features: Optional[Dict[int, Optional[float]]] = None,
) -> Tuple[Tensor, Tensor]:
    r"""Generate a set of candidates using a `torch.optim` optimizer.

    Optimizes an acquisition function starting from a set of initial candidates
    using an optimizer from `torch.optim`.

    Args:
        initial_conditions: Starting points for optimization.
        acquisition_function: Acquisition function to be used.
        lower_bounds: Minimum values for each column of initial_conditions.
        upper_bounds: Maximum values for each column of initial_conditions.
        optimizer (Optimizer): The pytorch optimizer to use to perform
            candidate search.
        options: Options used to control the optimization. Includes
            maxiter: Maximum number of iterations
        callback: A callback function accepting the current iteration, loss,
            and gradients as arguments. This function is executed after computing
            the loss and gradients, but before calling the optimizer.
        fixed_features: This is a dictionary of feature indices to values, where
            all generated candidates will have features fixed to these values.
            If the dictionary value is None, then that feature will just be
            fixed to the clamped value and not optimized. Assumes values to be
            compatible with lower_bounds and upper_bounds!

    Returns:
        2-element tuple containing

        - The set of generated candidates.
        - The acquisition value for each t-batch.

    Example:
        >>> qEI = qExpectedImprovement(model, best_f=0.2)
        >>> bounds = torch.tensor([[0., 0.], [1., 2.]])
        >>> Xinit = gen_batch_initial_conditions(
        >>>     qEI, bounds, q=3, num_restarts=25, raw_samples=500
        >>> )
        >>> batch_candidates, batch_acq_values = gen_candidates_torch(
                initial_conditions=Xinit,
                acquisition_function=qEI,
                lower_bounds=bounds[0],
                upper_bounds=bounds[1],
            )
    """
    options = options or {}

    # if there are fixed features we may optimize over a domain of lower dimension
    if fixed_features:
        subproblem = _remove_fixed_features_from_optimization(
            fixed_features=fixed_features,
            acquisition_function=acquisition_function,
            initial_conditions=initial_conditions,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            inequality_constraints=None,
            equality_constraints=None,
        )

        # call the routine with no fixed_features
        clamped_candidates, batch_acquisition = gen_candidates_torch(
            initial_conditions=subproblem.initial_conditions,
            acquisition_function=subproblem.acquisition_function,
            lower_bounds=subproblem.lower_bounds,
            upper_bounds=subproblem.upper_bounds,
            optimizer=optimizer,
            options=options,
            callback=callback,
            fixed_features=None,
        )
        clamped_candidates = subproblem.acquisition_function._construct_X_full(
            clamped_candidates
        )
        return clamped_candidates, batch_acquisition

    _clamp = partial(columnwise_clamp, lower=lower_bounds, upper=upper_bounds)
    clamped_candidates = _clamp(initial_conditions).requires_grad_(True)
    _optimizer = optimizer(params=[clamped_candidates], lr=options.get("lr", 0.025))

    i = 0
    stop = False
    stopping_criterion = ExpMAStoppingCriterion(
        **_filter_kwargs(ExpMAStoppingCriterion, **options)
    )
    while not stop:
        i += 1
        with torch.no_grad():
            X = _clamp(clamped_candidates).requires_grad_(True)

        loss = -acquisition_function(X).sum()
        grad = torch.autograd.grad(loss, X)[0]
        if callback:
            callback(i, loss, grad)

        def assign_grad():
            _optimizer.zero_grad()
            clamped_candidates.grad = grad
            return loss

        _optimizer.step(assign_grad)
        stop = stopping_criterion.evaluate(fvals=loss.detach())

    clamped_candidates = _clamp(clamped_candidates)
    with torch.no_grad():
        batch_acquisition = acquisition_function(clamped_candidates)

    return clamped_candidates, batch_acquisition


def get_best_candidates(batch_candidates: Tensor, batch_values: Tensor) -> Tensor:
    r"""Extract best (q-batch) candidate from batch of candidates

    Args:
        batch_candidates: A `b x q x d` tensor of `b` q-batch candidates, or a
            `b x d` tensor of `b` single-point candidates.
        batch_values: A tensor with `b` elements containing the value of the
            respective candidate (higher is better).

    Returns:
        A tensor of size `q x d` (if q-batch mode) or `d` from batch_candidates
        with the highest associated value.

    Example:
        >>> qEI = qExpectedImprovement(model, best_f=0.2)
        >>> bounds = torch.tensor([[0., 0.], [1., 2.]])
        >>> Xinit = gen_batch_initial_conditions(
        >>>     qEI, bounds, q=3, num_restarts=25, raw_samples=500
        >>> )
        >>> batch_candidates, batch_acq_values = gen_candidates_scipy(
                initial_conditions=Xinit,
                acquisition_function=qEI,
                lower_bounds=bounds[0],
                upper_bounds=bounds[1],
            )
        >>> best_candidates = get_best_candidates(batch_candidates, batch_acq_values)
    """
    best = torch.argmax(batch_values.view(-1), dim=0)
    return batch_candidates[best]
