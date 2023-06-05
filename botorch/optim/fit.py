#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Tools for model fitting."""

from __future__ import annotations

from functools import partial
from itertools import filterfalse
from time import monotonic
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Pattern,
    Sequence,
    Set,
    Tuple,
    Union,
)
from warnings import warn

from botorch.exceptions.warnings import OptimizationWarning
from botorch.optim.closures import get_loss_closure_with_grads
from botorch.optim.core import (
    OptimizationResult,
    OptimizationStatus,
    scipy_minimize,
    torch_minimize,
)
from botorch.optim.numpy_converter import (
    _scipy_objective_and_grad,
    module_to_array,
    set_params_with_array,
)
from botorch.optim.stopping import ExpMAStoppingCriterion
from botorch.optim.utils import (
    _filter_kwargs,
    _get_extra_mll_args,
    get_name_filter,
    get_parameters_and_bounds,
    TorchAttr,
)
from botorch.optim.utils.model_utils import get_parameters
from botorch.utils.types import DEFAULT
from gpytorch.mlls.marginal_log_likelihood import MarginalLogLikelihood
from gpytorch.settings import fast_computations
from numpy import ndarray
from scipy.optimize import Bounds, minimize
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


def fit_gpytorch_scipy(
    mll: MarginalLogLikelihood,
    bounds: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None,
    method: str = "L-BFGS-B",
    options: Optional[Dict[str, Any]] = None,
    track_iterations: bool = False,
    approx_mll: bool = False,
    scipy_objective: TScipyObjective = _scipy_objective_and_grad,
    module_to_array_func: TModToArray = module_to_array,
    module_from_array_func: TArrayToMod = set_params_with_array,
    **kwargs: Any,
) -> Tuple[MarginalLogLikelihood, Dict[str, Union[float, List[OptimizationResult]]]]:
    r"""Legacy method for scipy-based fitting of gpytorch models.

    The model and likelihood in mll must already be in train mode. This method requires
    that the model has `train_inputs` and `train_targets`.

    Args:
        mll: MarginalLogLikelihood to be maximized.
        bounds: A dictionary mapping parameter names to tuples of lower and upper
            bounds.
        method: Solver type, passed along to scipy.optimize.minimize.
        options: Dictionary of solver options, passed along to scipy.optimize.minimize.
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
        "iterations": List of OptimizationResult objects with information on each
        iteration. If track_iterations is False, will be empty.
        "OptimizeResult": The result returned by `scipy.optim.minimize`.
    """
    warn(
        "`fit_gpytorch_scipy` is marked for deprecation, consider using "
        "`scipy_minimize` or its model fitting helper `fit_gpytorch_mll_scipy`.",
        DeprecationWarning,
    )
    start_time = monotonic()
    iterations: List[OptimizationResult] = []

    options = {} if options is None else options.copy()
    exclude: Iterator[Union[Pattern, str]] = options.pop("exclude", None)
    if exclude:
        exclude, _ = zip(  # get the qualified names of excluded parameters
            *filterfalse(get_name_filter(exclude), mll.named_parameters())
        )

    x0, property_dict, bounds = module_to_array_func(
        module=mll, exclude=exclude, bounds=bounds
    )
    if bounds is not None:
        bounds = Bounds(lb=bounds[0], ub=bounds[1], keep_feasible=True)

    def wrapper(x: ndarray) -> Tuple[float, ndarray]:
        with fast_computations(log_prob=approx_mll):
            return scipy_objective(x=x, mll=mll, property_dict=property_dict)

    def store_iteration(xk):
        iterations.append(
            OptimizationResult(
                step=len(iterations),
                fval=float(wrapper(xk)[0]),
                status=OptimizationStatus.RUNNING,
                runtime=monotonic() - start_time,
            )
        )

    result = minimize(
        wrapper,
        x0,
        bounds=bounds,
        method=method,
        jac=True,
        options=options,
        callback=store_iteration if track_iterations else None,
    )

    info_dict = {
        "fopt": float(result.fun),
        "wall_time": monotonic() - start_time,
        "iterations": iterations,
        "OptimizeResult": result,
    }
    if not result.success:
        try:
            # Some result.message are bytes
            msg = result.message.decode("ascii")
        except AttributeError:
            # Others are str
            msg = result.message
        warn(
            f"Fitting failed with the optimizer reporting '{msg}'", OptimizationWarning
        )

    # Set to optimum
    mll = module_from_array_func(mll, result.x, property_dict)
    return mll, info_dict


def fit_gpytorch_torch(
    mll: MarginalLogLikelihood,
    bounds: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None,
    optimizer_cls: Optimizer = Adam,
    options: Optional[Dict[str, Any]] = None,
    track_iterations: bool = False,
    approx_mll: bool = False,
) -> Tuple[MarginalLogLikelihood, Dict[str, Union[float, List[OptimizationResult]]]]:
    r"""Legacy method for torch-based fitting of gpytorch models.

    The model and likelihood in mll must already be in train mode.
    Note: this method requires that the model has `train_inputs` and `train_targets`.

    Args:
        mll: MarginalLogLikelihood to be maximized.
        bounds: An optional dictionary mapping parameter names to tuples
            of lower and upper bounds. Bounds specified here take precedence
            over bounds on the same parameters specified in the constraints
            registered with the module.
        optimizer_cls: Torch optimizer to use. Must not require a closure.
        options: options for model fitting. Relevant options will be passed to
            the `optimizer_cls`. Additionally, options can include: "disp"
            to specify whether to display model fitting diagnostics and "maxiter"
            to specify the maximum number of iterations.

    Returns:
        2-element tuple containing
        - mll with parameters optimized in-place.
        - Dictionary with the following key/values:
        "fopt": Best mll value.
        "wall_time": Wall time of fitting.
        "iterations": List of OptimizationResult objects with information on each
        iteration. If track_iterations is False, will be empty.

    Example:
        >>> gp = SingleTaskGP(train_X, train_Y)
        >>> mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        >>> mll.train()
        >>> fit_gpytorch_torch(mll)
        >>> mll.eval()
    """
    warn(
        "`fit_gpytorch_torch` is marked for deprecation, consider using "
        "`torch_minimize` or its model fitting helper `fit_gpytorch_mll_torch`.",
        DeprecationWarning,
    )
    _options = {"maxiter": 100, "disp": True, "lr": 0.05}
    _options.update(options or {})
    exclude = _options.pop("exclude", None)
    parameters = get_parameters(
        mll,
        requires_grad=True,
        name_filter=None if exclude is None else get_name_filter(exclude),
    )

    optimizer = optimizer_cls(
        params=list(parameters.values()), **_filter_kwargs(optimizer_cls, **_options)
    )
    iterations: List[OptimizationResult] = []
    stopping_criterion = ExpMAStoppingCriterion(
        **_filter_kwargs(ExpMAStoppingCriterion, **_options)
    )

    def closure() -> Tuple[Tensor, Tuple[Tensor, ...]]:
        optimizer.zero_grad()
        with fast_computations(log_prob=approx_mll):
            out = mll.model(*mll.model.train_inputs)
            loss = -mll(out, mll.model.train_targets, *_get_extra_mll_args(mll)).sum()
            loss.backward()

        return loss, tuple(param.grad for param in parameters.values())

    def store_iteration(parameters: Dict[str, Tensor], result: OptimizationResult):
        iterations.append(result)

    result = fit_gpytorch_mll_torch(
        mll=mll,
        closure=closure,
        bounds=bounds,
        parameters=parameters,
        optimizer=optimizer,
        stopping_criterion=stopping_criterion,
        callback=store_iteration if track_iterations else None,
    )
    return mll, {
        "fopt": result.fval,
        "wall_time": result.runtime,
        "iterations": iterations,
    }
