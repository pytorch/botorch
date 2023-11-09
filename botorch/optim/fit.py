#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Tools for model fitting."""

from __future__ import annotations

from functools import partial
from typing import Any, Callable, Dict, Optional, Sequence, Set, Tuple, Union
from warnings import warn

from botorch.exceptions.warnings import OptimizationWarning
from botorch.optim.closures import get_loss_closure_with_grads
from botorch.optim.core import (
    OptimizationResult,
    OptimizationStatus,
    scipy_minimize,
    torch_minimize,
)
from botorch.optim.stopping import ExpMAStoppingCriterion
from botorch.optim.utils import get_parameters_and_bounds, TorchAttr
from botorch.utils.types import DEFAULT
from gpytorch.mlls.marginal_log_likelihood import MarginalLogLikelihood
from numpy import ndarray
from torch import Tensor
from torch.nn import Module
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer

TBoundsDict = Dict[str, Tuple[Optional[float], Optional[float]]]
TScipyObjective = Callable[
    [ndarray, MarginalLogLikelihood, Dict[str, TorchAttr]], Tuple[float, ndarray]
]
TModToArray = Callable[
    [Module, Optional[TBoundsDict], Optional[Set[str]]],
    Tuple[ndarray, Dict[str, TorchAttr], Optional[ndarray]],
]
TArrayToMod = Callable[[Module, ndarray, Dict[str, TorchAttr]], Module]


def fit_gpytorch_mll_scipy(
    mll: MarginalLogLikelihood,
    parameters: Optional[Dict[str, Tensor]] = None,
    bounds: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None,
    closure: Optional[Callable[[], Tuple[Tensor, Sequence[Optional[Tensor]]]]] = None,
    closure_kwargs: Optional[Dict[str, Any]] = None,
    method: str = "L-BFGS-B",
    options: Optional[Dict[str, Any]] = None,
    callback: Optional[Callable[[Dict[str, Tensor], OptimizationResult], None]] = None,
    timeout_sec: Optional[float] = None,
) -> OptimizationResult:
    r"""Generic scipy.optimized-based fitting routine for GPyTorch MLLs.

    The model and likelihood in mll must already be in train mode.

    Args:
        mll: MarginalLogLikelihood to be maximized.
        parameters: Optional dictionary of parameters to be optimized. Defaults
            to all parameters of `mll` that require gradients.
        bounds: A dictionary of user-specified bounds for `parameters`. Used to update
            default parameter bounds obtained from `mll`.
        closure: Callable that returns a tensor and an iterable of gradient tensors.
            Responsible for setting the `grad` attributes of `parameters`. If no closure
            is provided, one will be obtained by calling `get_loss_closure_with_grads`.
        closure_kwargs: Keyword arguments passed to `closure`.
        method: Solver type, passed along to scipy.minimize.
        options: Dictionary of solver options, passed along to scipy.minimize.
        callback: Optional callback taking `parameters` and an OptimizationResult as its
            sole arguments.
        timeout_sec: Timeout in seconds after which to terminate the fitting loop
            (note that timing out can result in bad fits!).

    Returns:
        The final OptimizationResult.
    """
    # Resolve `parameters` and update default bounds
    _parameters, _bounds = get_parameters_and_bounds(mll)
    bounds = _bounds if bounds is None else {**_bounds, **bounds}
    if parameters is None:
        parameters = {n: p for n, p in _parameters.items() if p.requires_grad}

    if closure is None:
        closure = get_loss_closure_with_grads(mll, parameters=parameters)

    if closure_kwargs is not None:
        closure = partial(closure, **closure_kwargs)

    result = scipy_minimize(
        closure=closure,
        parameters=parameters,
        bounds=bounds,
        method=method,
        options=options,
        callback=callback,
        timeout_sec=timeout_sec,
    )
    if result.status != OptimizationStatus.SUCCESS:
        warn(
            f"`scipy_minimize` terminated with status {result.status}, displaying"
            f" original message from `scipy.optimize.minimize`: {result.message}",
            OptimizationWarning,
        )

    return result


def fit_gpytorch_mll_torch(
    mll: MarginalLogLikelihood,
    parameters: Optional[Dict[str, Tensor]] = None,
    bounds: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None,
    closure: Optional[Callable[[], Tuple[Tensor, Sequence[Optional[Tensor]]]]] = None,
    closure_kwargs: Optional[Dict[str, Any]] = None,
    step_limit: Optional[int] = None,
    stopping_criterion: Optional[Callable[[Tensor], bool]] = DEFAULT,  # pyre-ignore [9]
    optimizer: Union[Optimizer, Callable[..., Optimizer]] = Adam,
    scheduler: Optional[Union[_LRScheduler, Callable[..., _LRScheduler]]] = None,
    callback: Optional[Callable[[Dict[str, Tensor], OptimizationResult], None]] = None,
    timeout_sec: Optional[float] = None,
) -> OptimizationResult:
    r"""Generic torch.optim-based fitting routine for GPyTorch MLLs.

    Args:
        mll: MarginalLogLikelihood to be maximized.
        parameters: Optional dictionary of parameters to be optimized. Defaults
            to all parameters of `mll` that require gradients.
        bounds: A dictionary of user-specified bounds for `parameters`. Used to update
            default parameter bounds obtained from `mll`.
        closure: Callable that returns a tensor and an iterable of gradient tensors.
            Responsible for setting the `grad` attributes of `parameters`. If no closure
            is provided, one will be obtained by calling `get_loss_closure_with_grads`.
        closure_kwargs: Keyword arguments passed to `closure`.
        step_limit: Optional upper bound on the number of optimization steps.
        stopping_criterion: A StoppingCriterion for the optimization loop.
        optimizer: A `torch.optim.Optimizer` instance or a factory that takes
            a list of parameters and returns an `Optimizer` instance.
        scheduler: A `torch.optim.lr_scheduler._LRScheduler` instance or a factory
            that takes an `Optimizer` instance and returns an `_LRSchedule`.
        callback: Optional callback taking `parameters` and an OptimizationResult as its
            sole arguments.
        timeout_sec: Timeout in seconds after which to terminate the fitting loop
            (note that timing out can result in bad fits!).

    Returns:
        The final OptimizationResult.
    """
    if stopping_criterion == DEFAULT:
        stopping_criterion = ExpMAStoppingCriterion()

    # Resolve `parameters` and update default bounds
    param_dict, bounds_dict = get_parameters_and_bounds(mll)
    if parameters is None:
        parameters = {n: p for n, p in param_dict.items() if p.requires_grad}

    if closure is None:
        closure = get_loss_closure_with_grads(mll, parameters)

    if closure_kwargs is not None:
        closure = partial(closure, **closure_kwargs)

    return torch_minimize(
        closure=closure,
        parameters=parameters,
        bounds=bounds_dict if bounds is None else {**bounds_dict, **bounds},
        optimizer=optimizer,
        scheduler=scheduler,
        step_limit=step_limit,
        stopping_criterion=stopping_criterion,
        callback=callback,
        timeout_sec=timeout_sec,
    )
