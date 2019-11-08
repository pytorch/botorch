#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Tools for model fitting.
"""

import time
import warnings
from collections import OrderedDict
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np
from gpytorch import settings as gpt_settings
from gpytorch.mlls.marginal_log_likelihood import MarginalLogLikelihood
from scipy.optimize import Bounds, minimize
from torch import Tensor
from torch.optim.adam import Adam
from torch.optim.optimizer import Optimizer

from ..exceptions.warnings import OptimizationWarning
from .numpy_converter import TorchAttr, module_to_array, set_params_with_array
from .utils import ConvergenceCriterion, _filter_kwargs, _get_extra_mll_args


ParameterBounds = Dict[str, Tuple[Optional[float], Optional[float]]]


class OptimizationIteration(NamedTuple):
    itr: int
    fun: float
    time: float


def fit_gpytorch_torch(
    mll: MarginalLogLikelihood,
    bounds: Optional[ParameterBounds] = None,
    optimizer_cls: Optimizer = Adam,
    options: Optional[Dict[str, Any]] = None,
    track_iterations: bool = True,
    approx_mll: bool = True,
) -> Tuple[MarginalLogLikelihood, Dict[str, Union[float, List[OptimizationIteration]]]]:
    r"""Fit a gpytorch model by maximizing MLL with a torch optimizer.

    The model and likelihood in mll must already be in train mode.
    Note: this method requires that the model has `train_inputs` and `train_targets`.

    Args:
        mll: MarginalLogLikelihood to be maximized.
        bounds: A ParameterBounds dictionary mapping parameter names to tuples
            of lower and upper bounds. Bounds specified here take precedence
            over bounds on the same parameters specified in the constraints
            registered with the module.
        optimizer_cls: Torch optimizer to use. Must not require a closure.
        options: options for model fitting. Relevant options will be passed to
            the `optimizer_cls`. Additionally, options can include: "disp"
            to specify whether to display model fitting diagnostics and "maxiter"
            to specify the maximum number of iterations.
        track_iterations: Track the function values and wall time for each
            iteration.
        approx_mll: If True, use gpytorch's approximate MLL computation (
            according to the gpytorch defaults based on the training at size).
            Unlike for the deterministic algorithms used in fit_gpytorch_scipy,
            this is not an issue for stochastic optimizers.

    Returns:
        2-element tuple containing
        - mll with parameters optimized in-place.
        - Dictionary with the following key/values:
        "fopt": Best mll value.
        "wall_time": Wall time of fitting.
        "iterations": List of OptimizationIteration objects with information on each
        iteration. If track_iterations is False, will be empty.

    Example:
        >>> gp = SingleTaskGP(train_X, train_Y)
        >>> mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        >>> mll.train()
        >>> fit_gpytorch_torch(mll)
        >>> mll.eval()
    """
    optim_options = {"maxiter": 100, "disp": True, "lr": 0.05}
    optim_options.update(options or {})
    exclude = optim_options.pop("exclude", None)
    if exclude is not None:
        mll_params = [
            t for p_name, t in mll.named_parameters() if p_name not in exclude
        ]
    else:
        mll_params = list(mll.parameters())
    optimizer = optimizer_cls(
        params=[{"params": mll_params}],
        **_filter_kwargs(optimizer_cls, **optim_options),
    )

    # get bounds specified in model (if any)
    bounds_: ParameterBounds = {}
    if hasattr(mll, "named_parameters_and_constraints"):
        for param_name, _, constraint in mll.named_parameters_and_constraints():
            if constraint is not None and not constraint.enforced:
                bounds_[param_name] = constraint.lower_bound, constraint.upper_bound

    # update with user-supplied bounds (overwrites if already exists)
    if bounds is not None:
        bounds_.update(bounds)

    iterations = []
    t1 = time.time()

    param_trajectory: Dict[str, List[Tensor]] = {
        name: [] for name, param in mll.named_parameters()
    }
    loss_trajectory: List[float] = []
    i = 0
    converged = False
    convergence_criterion = ConvergenceCriterion(
        **_filter_kwargs(ConvergenceCriterion, **optim_options)
    )
    train_inputs, train_targets = mll.model.train_inputs, mll.model.train_targets
    while not converged:
        optimizer.zero_grad()
        with gpt_settings.fast_computations(log_prob=approx_mll):
            output = mll.model(*train_inputs)
            # we sum here to support batch mode
            args = [output, train_targets] + _get_extra_mll_args(mll)
            loss = -mll(*args).sum()
            loss.backward()
        loss_trajectory.append(loss.item())
        for name, param in mll.named_parameters():
            param_trajectory[name].append(param.detach().clone())
        if optim_options["disp"] and (
            (i + 1) % 10 == 0 or i == (optim_options["maxiter"] - 1)
        ):
            print(f"Iter {i + 1}/{optim_options['maxiter']}: {loss.item()}")
        if track_iterations:
            iterations.append(OptimizationIteration(i, loss.item(), time.time() - t1))
        optimizer.step()
        # project onto bounds:
        if bounds_:
            for pname, param in mll.named_parameters():
                if pname in bounds_:
                    param.data = param.data.clamp(*bounds_[pname])
        i += 1
        converged = convergence_criterion.evaluate(fvals=loss.detach())
    info_dict = {
        "fopt": loss_trajectory[-1],
        "wall_time": time.time() - t1,
        "iterations": iterations,
    }
    return mll, info_dict


def fit_gpytorch_scipy(
    mll: MarginalLogLikelihood,
    bounds: Optional[ParameterBounds] = None,
    method: str = "L-BFGS-B",
    options: Optional[Dict[str, Any]] = None,
    track_iterations: bool = True,
    approx_mll: bool = False,
) -> Tuple[MarginalLogLikelihood, Dict[str, Union[float, List[OptimizationIteration]]]]:
    r"""Fit a gpytorch model by maximizing MLL with a scipy optimizer.

    The model and likelihood in mll must already be in train mode.
    This method requires that the model has `train_inputs` and `train_targets`.

    Args:
        mll: MarginalLogLikelihood to be maximized.
        bounds: A dictionary mapping parameter names to tuples of lower and upper
            bounds.
        method: Solver type, passed along to scipy.minimize.
        options: Dictionary of solver options, passed along to scipy.minimize.
        track_iterations: Track the function values and wall time for each
            iteration.
        approx_mll: If True, use gpytorch's approximate MLL computation. This is
            disabled by default since the stochasticity is an issue for
            determistic optimizers). Enabling this is only recommended when
            working with large training data sets (n>2000).

    Returns:
        2-element tuple containing
        - MarginalLogLikelihood with parameters optimized in-place.
        - Dictionary with the following key/values:
        "fopt": Best mll value.
        "wall_time": Wall time of fitting.
        "iterations": List of OptimizationIteration objects with information on each
        iteration. If track_iterations is False, will be empty.

    Example:
        >>> gp = SingleTaskGP(train_X, train_Y)
        >>> mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        >>> mll.train()
        >>> fit_gpytorch_scipy(mll)
        >>> mll.eval()
    """
    options = options or {}
    x0, property_dict, bounds = module_to_array(
        module=mll, bounds=bounds, exclude=options.pop("exclude", None)
    )
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

    with gpt_settings.fast_computations(log_prob=approx_mll):
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
                obj, _ = _scipy_objective_and_grad(
                    x=xk, mll=mll, property_dict=property_dict
                )
                iterations.append(OptimizationIteration(i, obj, ts[i]))
    # Construct info dict
    info_dict = {
        "fopt": float(res.fun),
        "wall_time": time.time() - t1,
        "iterations": iterations,
    }
    if not res.success:
        try:
            # Some res.message are bytes
            msg = res.message.decode("ascii")
        except AttributeError:
            # Others are str
            msg = res.message
        warnings.warn(
            f"Fitting failed with the optimizer reporting '{msg}'", OptimizationWarning
        )
    # Set to optimum
    mll = set_params_with_array(mll, res.x, property_dict)
    return mll, info_dict


def _scipy_objective_and_grad(
    x: np.ndarray, mll: MarginalLogLikelihood, property_dict: Dict[str, TorchAttr]
) -> Tuple[float, np.ndarray]:
    r"""Get objective and gradient in format that scipy expects.

    Args:
        x: The (flattened) input parameters.
        mll: The MarginalLogLikelihood module to evaluate.
        property_dict: The property dictionary required to "unflatten" the input
            parameter vector, as generated by `module_to_array`.

    Returns:
        2-element tuple containing

        - The objective value.
        - The gradient of the objective.
    """
    mll = set_params_with_array(mll, x, property_dict)
    train_inputs, train_targets = mll.model.train_inputs, mll.model.train_targets
    mll.zero_grad()
    try:  # catch linear algebra errors in gpytorch
        output = mll.model(*train_inputs)
        args = [output, train_targets] + _get_extra_mll_args(mll)
        loss = -mll(*args).sum()
    except RuntimeError as e:
        if "singular" in e.args[0]:
            return float("nan"), np.full_like(x, "nan")
        else:
            raise e  # pragma: nocover
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
