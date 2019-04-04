#!/usr/bin/env python3

import time
from collections import OrderedDict
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import numpy as np
from gpytorch.mlls.marginal_log_likelihood import MarginalLogLikelihood
from scipy.optimize import Bounds, minimize
from torch import Tensor
from torch.optim.adam import Adam
from torch.optim.optimizer import Optimizer

from .numpy_converter import TorchAttr, module_to_array, set_params_with_array
from .utils import _get_extra_mll_args, check_convergence


ParameterBounds = Dict[str, Tuple[Optional[float], Optional[float]]]


class OptimizationIteration(NamedTuple):
    itr: int
    fun: float
    time: float


def fit_gpytorch_torch(
    mll: MarginalLogLikelihood,
    optimizer_cls: Optimizer = Adam,
    lr: float = 0.05,
    maxiter: int = 100,
    optimizer_args: Optional[Dict[str, float]] = None,
    disp: bool = True,
    track_iterations: bool = True,
) -> Tuple[MarginalLogLikelihood, List[OptimizationIteration]]:
    """Fit a gpytorch model by maximizing MLL with a torch optimizer.

    The model and likelihood in mll must already be in train mode.

    Note: this method requires that the model has `train_inputs` and `train_targets`.

    Args:
        mll: MarginalLogLikelihood to be maximized.
        optimizer_cls: Torch optimizer to use. Must not need a closure.
            Defaults to Adam.
        lr: Starting learning rate.
        maxiter: Maximum number of iterations.
        optimizer_args: Additional arguments to instantiate optimizer_cls.
        disp: Print information during optimization.
        track_iterations: Track the function values and wall time for each
            iteration.

    Returns:
        mll: mll with parameters optimized in-place.
        iterations: List of OptimizationIteration objects describing each
            iteration. If track_iterations is False, will be an empty list.
    """
    optimizer_args = {} if optimizer_args is None else optimizer_args
    optimizer = optimizer_cls(
        params=[{"params": mll.parameters()}], lr=lr, **optimizer_args
    )

    iterations = []
    t1 = time.time()

    param_trajectory: Dict[str, List[Tensor]] = {
        name: [] for name, param in mll.named_parameters()
    }
    loss_trajectory: List[float] = []
    i = 0
    converged = False
    train_inputs, train_targets = mll.model.train_inputs, mll.model.train_targets
    while not converged:
        optimizer.zero_grad()
        output = mll.model(*train_inputs)
        # we sum here to support batch mode
        args = [output, train_targets] + _get_extra_mll_args(mll)
        loss = -mll(*args).sum()
        loss.backward()
        loss_trajectory.append(loss.item())
        for name, param in mll.named_parameters():
            param_trajectory[name].append(param.detach().clone())
        if disp and (i % 10 == 0 or i == (maxiter - 1)):
            print(f"Iter {i +1}/{maxiter}: {loss.item()}")
        if track_iterations:
            iterations.append(OptimizationIteration(i, loss.item(), time.time() - t1))
        optimizer.step()
        i += 1
        converged = check_convergence(
            loss_trajectory=loss_trajectory,
            param_trajectory=param_trajectory,
            options={"maxiter": maxiter},
        )
    return mll, iterations


def fit_gpytorch_scipy(
    mll: MarginalLogLikelihood,
    bounds: Optional[ParameterBounds] = None,
    method: str = "L-BFGS-B",
    options: Optional[Dict[str, Any]] = None,
    track_iterations: bool = True,
) -> Tuple[MarginalLogLikelihood, List[OptimizationIteration]]:
    """Fit a gpytorch model by maximizing MLL with a scipy optimizer.

    The model and likelihood in mll must already be in train mode.

    Note: this method requires that the model has `train_inputs` and `train_targets`.

    Args:
        mll: MarginalLogLikelihood to be maximized.
        bounds: A ParameterBounds dictionary mapping parameter names to tuples of
            lower and upper bounds.
        method: Solver type, passed along to scipy.minimize.
        options: Dictionary of solver options, passed along to scipy.minimize.
        track_iterations: Track the function values and wall time for each
            iteration.

    Returns:
        mll: mll with parameters optimized in-place.
        iterations: List of OptimizationIteration objects describing each
            iteration. If track_iterations is False, will be an empty list.
    """
    x0, property_dict, bounds = module_to_array(module=mll, bounds=bounds)
    x0 = x0.astype(np.float64)
    if bounds is not None:
        bounds = Bounds(lb=bounds[0], ub=bounds[1], keep_feasible=True)

    xs = []
    ts = []
    t1 = time.time()

    def store_iteration(xk):
        xs.append(xk.copy())
        ts.append(time.time() - t1)

    cb = store_iteration if track_iterations else None

    res = minimize(
        _scipy_objective_and_grad,
        x0,
        args=(mll, property_dict),
        bounds=bounds,
        method=method,
        jac=True,
        options=options,
        callback=cb,
    )
    iterations = []
    if track_iterations:
        for i, xk in enumerate(xs):
            obj, _ = _scipy_objective_and_grad(xk, mll, property_dict)
            iterations.append(OptimizationIteration(i, obj, ts[i]))

    # Set to optimum
    mll = set_params_with_array(mll, res.x, property_dict)
    return mll, iterations


def _scipy_objective_and_grad(
    x: np.ndarray, mll: MarginalLogLikelihood, property_dict: Dict[str, TorchAttr]
) -> Tuple[float, np.ndarray]:
    mll = set_params_with_array(mll, x, property_dict)
    train_inputs, train_targets = mll.model.train_inputs, mll.model.train_targets
    mll.zero_grad()
    output = mll.model(*train_inputs)
    args = [output, train_targets] + _get_extra_mll_args(mll)
    loss = -mll(*args).sum()
    loss.backward()
    param_dict = OrderedDict(mll.named_parameters())
    grad = []
    for p_name in property_dict:
        t = param_dict[p_name].grad
        if t is None:
            # this deals with parameters that do not affect the loss
            grad.append(np.zeros(property_dict[p_name].shape.numel()))
        else:
            grad.append(t.detach().view(-1).cpu().double().clone().numpy())
    mll.zero_grad()
    return loss.item(), np.concatenate(grad)
