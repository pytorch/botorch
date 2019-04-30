#!/usr/bin/env python3

r"""
Candidate generation utilities.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import torch
from scipy.optimize import minimize
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

from .optim.parameter_constraints import (
    _arrayify,
    make_scipy_bounds,
    make_scipy_linear_constraints,
)
from .optim.utils import check_convergence, columnwise_clamp, fix_features


def gen_candidates_scipy(
    initial_conditions: Tensor,
    acquisition_function: Module,
    lower_bounds: Optional[Union[float, Tensor]] = None,
    upper_bounds: Optional[Union[float, Tensor]] = None,
    inequality_constraints: Optional[List[Tuple[Tensor, Tensor, float]]] = None,
    equality_constraints: Optional[List[Tuple[Tensor, Tensor, float]]] = None,
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
            `\sum_i (X[indices[i]] * coefficients[i]) >= rhs`
        equality constraints: A list of tuples (indices, coefficients, rhs),
            with each tuple encoding an inequality constraint of the form
            `\sum_i (X[indices[i]] * coefficients[i]) = rhs`
        options: options used to control the optimization including "method"
            and "maxiter"
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
    clamped_candidates = columnwise_clamp(
        initial_conditions, lower_bounds, upper_bounds
    ).requires_grad_(True)

    shapeX = clamped_candidates.shape
    x0 = _arrayify(clamped_candidates.view(-1))
    bounds = make_scipy_bounds(
        X=initial_conditions, lower_bounds=lower_bounds, upper_bounds=upper_bounds
    )
    constraints = make_scipy_linear_constraints(
        shapeX=clamped_candidates.shape,
        inequality_constraints=inequality_constraints,
        equality_constraints=equality_constraints,
    )

    def f(x):
        X = (
            torch.from_numpy(x)
            .to(initial_conditions)
            .view(shapeX)
            .contiguous()
            .requires_grad_(True)
        )
        X_fix = fix_features(X=X, fixed_features=fixed_features)
        loss = -acquisition_function(X_fix).sum()
        loss.backward()
        fval = loss.item()
        gradf = _arrayify(X.grad.view(-1))
        return fval, gradf

    res = minimize(
        f,
        x0,
        method=options.get("method", "SLSQP"),
        jac=True,
        bounds=bounds,
        constraints=constraints,
        options={k: v for k, v in options.items() if k != "method"},
    )
    candidates = fix_features(
        X=torch.from_numpy(res.x)  # pyre-ignore [16]
        .to(initial_conditions)
        .view(shapeX)
        .contiguous(),
        fixed_features=fixed_features,
    )
    batch_acquisition = acquisition_function(candidates)
    return candidates, batch_acquisition


def gen_candidates_torch(
    initial_conditions: Tensor,
    acquisition_function: Callable,
    lower_bounds: Optional[Union[float, Tensor]] = None,
    upper_bounds: Optional[Union[float, Tensor]] = None,
    optimizer: Type[Optimizer] = torch.optim.Adam,
    options: Optional[Dict[str, Union[float, str]]] = None,
    verbose: bool = True,
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
        verbose: If True, provide verbose output.
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
    clamped_candidates = columnwise_clamp(
        initial_conditions, lower_bounds, upper_bounds
    ).requires_grad_(True)
    candidates = fix_features(clamped_candidates, fixed_features)
    bayes_optimizer = optimizer(
        params=[clamped_candidates], lr=options.get("lr", 0.025)
    )
    param_trajectory: Dict[str, List[Tensor]] = {"candidates": []}
    loss_trajectory: List[float] = []
    i = 0
    converged = False
    while not converged:
        i += 1
        loss = -acquisition_function(candidates).sum()
        if verbose:
            print("Iter: {} - Value: {:.3f}".format(i, -loss.item()))
        loss_trajectory.append(loss.item())
        param_trajectory["candidates"].append(candidates.clone())

        def closure():
            bayes_optimizer.zero_grad()
            loss = -acquisition_function(candidates).sum()
            loss.backward()
            return loss

        bayes_optimizer.step(closure)  # pyre-ignore
        clamped_candidates.data = columnwise_clamp(
            clamped_candidates, lower_bounds, upper_bounds
        )
        candidates = fix_features(clamped_candidates, fixed_features)
        converged = check_convergence(
            loss_trajectory=loss_trajectory,
            param_trajectory=param_trajectory,
            options=options,
        )
    batch_acquisition = acquisition_function(candidates)
    return candidates, batch_acquisition


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
    best = torch.max(batch_values.view(-1), dim=0)[1].item()
    return batch_candidates[best]
