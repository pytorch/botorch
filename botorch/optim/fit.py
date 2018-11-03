#!/usr/bin/env python3

import time
from collections import OrderedDict
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import numpy as np
from botorch.optim.converter import TorchAttr, module_to_array, set_params_with_array
from botorch.utils import check_convergence
from gpytorch.mlls.marginal_log_likelihood import MarginalLogLikelihood
from scipy.optimize import minimize
from torch import Tensor
from torch.optim.adam import Adam
from torch.optim.optimizer import Optimizer


class OptimizationIteration(NamedTuple):
    itr: int
    fun: float
    time: float


def fit_torch(
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
        params=[{"params": mll.model.parameters()}], lr=lr, **optimizer_args
    )

    iterations = []
    t1 = time.time()

    param_trajectory: Dict[str, List[Tensor]] = {
        name: [] for name, param in mll.model.named_parameters()
    }
    loss_trajectory: List[float] = []
    i = 0
    converged = False
    while not converged:
        optimizer.zero_grad()
        output = mll.model(mll.model.train_inputs[0])
        # we sum here to support batch mode
        loss = -mll(output, mll.model.train_targets).sum()
        loss.backward()
        loss_trajectory.append(loss.item())
        for name, param in mll.model.named_parameters():
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
            options={},
            max_iter=maxiter,
        )
    return mll, iterations


def fit_scipy(
    mll: MarginalLogLikelihood,
    method: str = "L-BFGS-B",
    options: Optional[Dict[str, Any]] = None,
    track_iterations: bool = True,
):
    """Fit a gpytorch model by maximizing MLL with a scipy optimizer.

    The model and likelihood in mll must already be in train mode.

    Args:
        mll: MarginalLogLikelihood to be maximized.
        method: Solver type, passed along to scipy.minimize.
        options: Dictionary of solver options, passed along to scipy.minimize.
        track_iterations: Track the function values and wall time for each
            iteration.

    Returns:
        mll: mll with parameters optimized in-place.
        iterations: List of OptimizationIteration objects describing each
            iteration. If track_iterations is False, will be an empty list.
    """
    x0, property_dict = module_to_array(mll)
    x0 = x0.astype(np.float64)

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
        method=method,
        jac=True,
        options=options,
        callback=cb,
    )

    iterations = []
    if track_iterations:
        for i, xk in enumerate(xs):
            iterations.append(
                OptimizationIteration(
                    i, _scipy_objective_and_grad(xk, mll, property_dict)[0], ts[i]
                )
            )

    # Set to optimum
    mll = set_params_with_array(mll, res.x, property_dict)
    return mll, iterations


def _scipy_objective_and_grad(
    x: np.ndarray, mll: MarginalLogLikelihood, property_dict: Dict[str, TorchAttr]
) -> Tuple[float, np.ndarray]:
    mll = set_params_with_array(mll, x, property_dict)
    mll.zero_grad()
    output = mll.model(*mll.model.train_inputs)
    loss = -mll(output, mll.model.train_targets).sum()
    loss.backward()
    param_dict = OrderedDict(mll.named_parameters())
    grad = []
    for p_name in property_dict:
        t = param_dict[p_name].grad
        grad.append(t.detach().view(-1).cpu().double().clone().numpy())
    mll.zero_grad()
    return loss.item(), np.concatenate(grad)
