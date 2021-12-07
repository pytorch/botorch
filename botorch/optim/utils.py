#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Utilities for optimization.
"""

from __future__ import annotations

import warnings
from collections import OrderedDict
from inspect import signature
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.exceptions.errors import BotorchError
from botorch.exceptions.warnings import BotorchWarning
from botorch.models.gpytorch import ModelListGPyTorchModel, GPyTorchModel
from botorch.optim.numpy_converter import TorchAttr, set_params_with_array
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.mlls.marginal_log_likelihood import MarginalLogLikelihood
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from gpytorch.utils.errors import NanError, NotPSDError
from torch import Tensor


def sample_all_priors(model: GPyTorchModel) -> None:
    r"""Sample from hyperparameter priors (in-place).

    Args:
        model: A GPyTorchModel.
    """
    for _, module, prior, closure, setting_closure in model.named_priors():
        if setting_closure is None:
            raise RuntimeError(
                "Must provide inverse transform to be able to sample from prior."
            )
        try:
            setting_closure(module, prior.sample(closure(module).shape))
        except NotImplementedError:
            warnings.warn(
                f"`rsample` not implemented for {type(prior)}. Skipping.",
                BotorchWarning,
            )


def columnwise_clamp(
    X: Tensor,
    lower: Optional[Union[float, Tensor]] = None,
    upper: Optional[Union[float, Tensor]] = None,
    raise_on_violation: bool = False,
) -> Tensor:
    r"""Clamp values of a Tensor in column-wise fashion (with support for t-batches).

    This function is useful in conjunction with optimizers from the torch.optim
    package, which don't natively handle constraints. If you apply this after
    a gradient step you can be fancy and call it "projected gradient descent".
    This funtion is also useful for post-processing candidates generated by the
    scipy optimizer that satisfy bounds only up to numerical accuracy.

    Args:
        X: The `b x n x d` input tensor. If 2-dimensional, `b` is assumed to be 1.
        lower: The column-wise lower bounds. If scalar, apply bound to all columns.
        upper: The column-wise upper bounds. If scalar, apply bound to all columns.
        raise_on_violation: If `True`, raise an exception when the elments in `X`
            are out of the specified bounds (up to numerical accuracy). This is
            useful for post-processing candidates generated by optimizers that
            satisfy imposed bounds only up to numerical accuracy.

    Returns:
        The clamped tensor.
    """
    min_bounds = _expand_bounds(lower, X)
    max_bounds = _expand_bounds(upper, X)
    if min_bounds is not None and max_bounds is not None:
        if torch.any(min_bounds > max_bounds):
            raise ValueError("Minimum values must be <= maximum values")
    Xout = X
    if min_bounds is not None:
        Xout = Xout.max(min_bounds)
    if max_bounds is not None:
        Xout = Xout.min(max_bounds)
    if raise_on_violation and not torch.allclose(Xout, X):
        raise BotorchError("Original value(s) are out of bounds.")
    return Xout


def fix_features(
    X: Tensor, fixed_features: Optional[Dict[int, Optional[float]]] = None
) -> Tensor:
    r"""Fix feature values in a Tensor.

    The fixed features will have zero gradient in downstream calculations.

    Args:
        X: input Tensor with shape `... x p`, where `p` is the number of features
        fixed_features: A dictionary with keys as column indices and values
            equal to what the feature should be set to in `X`. If the value is
            None, that column is just considered fixed. Keys should be in the
            range `[0, p - 1]`.

    Returns:
        The tensor X with fixed features.
    """
    if fixed_features is None:
        return X
    else:
        return torch.cat(
            [
                X[..., i].unsqueeze(-1)
                if i not in fixed_features
                else _fix_feature(X[..., i].unsqueeze(-1), fixed_features[i])
                for i in range(X.shape[-1])
            ],
            dim=-1,
        )


def _fix_feature(Z: Tensor, value: Optional[float]) -> Tensor:
    r"""Helper function returns a Tensor like `Z` filled with `value` if provided."""
    if value is None:
        return Z.detach()
    return torch.full_like(Z, value)


def _expand_bounds(
    bounds: Optional[Union[float, Tensor]], X: Tensor
) -> Optional[Tensor]:
    r"""Expands a tensor representing bounds.

    Expand the dimension of bounds if necessary such that the dimension of bounds
    is the same as the dimension of `X`.

    Args:
        bounds: a bound (either upper or lower) of each entry of `X`. If this is a
            single float, then all entries have the same bound. Different sizes of
            tensors can be used to specify custom bounds. E.g., a `d`-dim tensor can
            be used to specify bounds for each column (last dimension) of `X`, or a
            tensor with same shape as `X` can be used to specify a different bound
            for each entry of `X`.
        X: `... x d` tensor

    Returns:
        A tensor of bounds expanded to the size of `X` if bounds is not None,
        and None if bounds is None.
    """
    if bounds is not None:
        if not torch.is_tensor(bounds):
            bounds = torch.tensor(bounds)
        try:
            ebounds = bounds.expand_as(X)
        except RuntimeError:
            raise RuntimeError("Bounds must be broadcastable to X!")
        return ebounds.to(dtype=X.dtype, device=X.device)
    else:
        return None


def _get_extra_mll_args(
    mll: MarginalLogLikelihood,
) -> Union[List[Tensor], List[List[Tensor]]]:
    r"""Obtain extra arguments for MarginalLogLikelihood objects.

    Get extra arguments (beyond the model output and training targets) required
    for the particular type of MarginalLogLikelihood for a forward pass.

    Args:
        mll: The MarginalLogLikelihood module.

    Returns:
        Extra arguments for the MarginalLogLikelihood.
        Returns an empty list if the mll type is unknown.
    """
    if isinstance(mll, ExactMarginalLogLikelihood):
        return list(mll.model.train_inputs)
    elif isinstance(mll, SumMarginalLogLikelihood):
        return [list(x) for x in mll.model.train_inputs]
    return []


def _filter_kwargs(function: Callable, **kwargs: Any) -> Any:
    r"""Filter out kwargs that are not applicable for a given function.
    Return a copy of given kwargs dict with only the required kwargs."""
    return {k: v for k, v in kwargs.items() if k in signature(function).parameters}


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
        return _handle_numerical_errors(error=e, x=x)
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


def _handle_numerical_errors(
    error: RuntimeError, x: np.ndarray
) -> Tuple[float, np.ndarray]:
    if isinstance(error, NotPSDError):
        raise error
    error_message = error.args[0] if len(error.args) > 0 else ""
    if (
        isinstance(error, NanError)
        or "singular" in error_message  # old pytorch message
        or "input is not positive-definite" in error_message  # since pytorch #63864
    ):
        return float("nan"), np.full_like(x, "nan")
    raise error  # pragma: nocover


def get_X_baseline(acq_function: AcquisitionFunction) -> Optional[Tensor]:
    r"""Extract X_baseline from an acquisition function.

    This tries to find the baseline set of points. First, this checks if the
    acquisition function has an `X_baseline` attribute. If it does not,
    then this method attempts to use the model's `train_inputs` as `X_baseline`.

    Args:
        acq_function: The acquisition function.

    Returns
        An optional `n x d`-dim tensor of baseline points. This is None if no
            baseline points are found.
    """
    try:
        X = acq_function.X_baseline
        # if there are no baseline points, use training points
        if X.shape[0] == 0:
            raise BotorchError
    except (BotorchError, AttributeError):
        try:
            # for entropy MOO methods
            model = acq_function.mo_model
        except AttributeError:
            try:
                # some acquisition functions do not have a model attribute
                # e.g. FixedFeatureAcquisitionFunction
                model = acq_function.model
            except AttributeError:
                warnings.warn("Failed to extract X_baseline.", BotorchWarning)
                return
        try:
            # make sure input transforms are not applied
            model.train()
            if isinstance(model, ModelListGPyTorchModel):
                X = model.models[0].train_inputs[0]
            else:
                X = model.train_inputs[0]
        except (BotorchError, AttributeError):
            warnings.warn("Failed to extract X_baseline.", BotorchWarning)
            return
    # just use one batch
    while X.ndim > 2:
        X = X[0]
    return X
