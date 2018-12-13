#!/usr/bin/env python3

from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
from botorch.utils import check_convergence, columnwise_clamp, fix_features
from scipy.optimize import Bounds, minimize
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer


def _arrayify(X: Tensor) -> np.ndarray:
    return X.cpu().detach().double().clone().numpy()


def _make_bounds_scipy(
    lower_bounds: Optional[Union[float, Tensor]],
    upper_bounds: Optional[Union[float, Tensor]],
    X: Tensor,
) -> Optional[Bounds]:
    """Creates a scipy Bounds object for optimziation

    Args:
        lower_bounds: Lower bounds on each column (last dimension) of X. If this
            is a single float, then all columns have the same bound.
        upper_bounds: Lower bounds on each column (last dimension) of X. If this
            is a single float, then all columns have the same bound.
        X: `... x d` tensor

    Returns
        A scipy Bounds object if either lower_bounds or upper_bounds is not None,
            and None otherwise.

    """
    if lower_bounds is None and upper_bounds is None:
        return None

    def _expand(bounds: Tensor, X: Tensor) -> Tensor:
        if bounds is None:
            ebounds = torch.full_like(X, np.inf)
        else:
            if not torch.is_tensor(bounds):
                bounds = torch.tensor(bounds)
            ebounds = bounds.expand_as(X)
        return ebounds.cpu().detach().double().contiguous().view(-1).clone().numpy()

    lb = -_expand(lower_bounds, X)
    ub = _expand(upper_bounds, X)
    return Bounds(lb=lb, ub=ub, keep_feasible=True)


def gen_candidates_scipy(
    initial_candidates: Tensor,
    acquisition_function: Module,
    lower_bounds: Optional[Union[float, Tensor]] = None,
    upper_bounds: Optional[Union[float, Tensor]] = None,
    options: Optional[Dict[str, Any]] = None,
    fixed_features: Optional[Dict[int, Optional[float]]] = None,
) -> Tuple[Tensor, Tensor]:
    """Generate a set of candidates via optimization from a given set of
    starting points.

    Args:
        initial_candidates: starting points for optimization
        acquisition_function: acquisition function to be used
        lower_bounds: minimum values for each column of initial_candidates
        upper_bounds: maximum values for each column of initial_candidates
        options:  options used to control the optimization including "method"
            and "maxiter"
        fixed_features:  This is a dictionary of feature indices
            to values, where all generated candidates will have features
            fixed to these values.  If the dictionary value is None, then that
            feature will just be fixed to the clamped value and
            not optimized.  Assumes values to be compatible with
            lower_bounds and upper_bounds!

    Returns:
        Tensor: The set of generated candidates
        Tensor: The acquisition value for each t-batch.

    """
    options = options or {}
    clamped_candidates = columnwise_clamp(
        initial_candidates, lower_bounds, upper_bounds
    ).requires_grad_(True)

    shapeX = clamped_candidates.shape
    x0 = _arrayify(clamped_candidates.view(-1))
    bounds = _make_bounds_scipy(
        lower_bounds=lower_bounds, upper_bounds=upper_bounds, X=initial_candidates
    )

    def f(x):
        X = (
            torch.from_numpy(x)
            .type_as(initial_candidates)
            .view(shapeX)
            .contiguous()
            .requires_grad_(True)
        )
        X = fix_features(X=X, fixed_features=fixed_features)
        loss = -acquisition_function(X).sum()
        loss.backward()
        fval = loss.item()
        gradf = _arrayify(X.grad.view(-1))
        return fval, gradf

    res = minimize(
        f,
        x0,
        method=options.get("method", "L-BFGS-B"),
        jac=True,
        bounds=bounds,
        options=options,
    )

    candidates = (
        torch.from_numpy(res.x).type_as(initial_candidates).view(shapeX).contiguous()
    )
    batch_acquisition = acquisition_function(candidates)
    return candidates, batch_acquisition


def gen_candidates_torch(
    initial_candidates: Tensor,
    acquisition_function: Callable,
    lower_bounds: Optional[Union[float, Tensor]] = None,
    upper_bounds: Optional[Union[float, Tensor]] = None,
    optimizer: Type[Optimizer] = torch.optim.Adam,
    options: Optional[Dict[str, Union[float, str]]] = None,
    max_iter: int = 50,
    verbose: bool = True,
    fixed_features: Optional[Dict[int, Optional[float]]] = None,
) -> Tuple[Tensor, Tensor]:
    """Generate a set of candidates via optimization from a given set of
    starting points.

    Args:
        initial_candidates: starting points for optimization
        acquisition_function: acquisition function to be used
        lower_bounds: minimum values for each column of initial_candidates
        upper_bounds: maximum values for each column of initial_candidates
        optimizer (Optimizer): The pytorch optimizer to use to perform
            candidate search
        options:  options used to control the optimization
        max_iter (int):  maximum number of iterations
        verbose (bool):  whether to provide verbose output
        fixed_features:  This is a dictionary of feature indices
            to values, where all generated candidates will have features
            fixed to these values.  If the dictionary value is None, then that
            feature will just be fixed to the clamped value and
            not optimized.  Assumes values to be compatible with
            lower_bounds and upper_bounds!

    Returns:
        Tensor: The set of generated candidates
        Tensor: The acquisition value for each t-batch.

    """
    options = options or {}
    clamped_candidates = columnwise_clamp(
        initial_candidates, lower_bounds, upper_bounds
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

        bayes_optimizer.step(closure)
        clamped_candidates.data = columnwise_clamp(
            clamped_candidates, lower_bounds, upper_bounds
        )
        candidates = fix_features(clamped_candidates, fixed_features)
        converged = check_convergence(
            loss_trajectory=loss_trajectory,
            param_trajectory=param_trajectory,
            options=options,
            max_iter=max_iter,
        )
    batch_acquisition = acquisition_function(candidates)
    return candidates, batch_acquisition
