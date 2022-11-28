#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
A converter that simplifies using numpy-based optimizers with generic torch
`nn.Module` classes. This enables using a `scipy.optim.minimize` optimizer
for optimizing module parameters.
"""

from __future__ import annotations

from collections import OrderedDict
from math import inf
from numbers import Number
from typing import Dict, List, Optional, Set, Tuple
from warnings import warn

import numpy as np
import torch
from botorch.optim.utils import (
    _get_extra_mll_args,
    _handle_numerical_errors,
    get_name_filter,
    get_parameters_and_bounds,
    TorchAttr,
)
from gpytorch.mlls import MarginalLogLikelihood
from torch.nn import Module


def module_to_array(
    module: Module,
    bounds: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None,
    exclude: Optional[Set[str]] = None,
) -> Tuple[np.ndarray, Dict[str, TorchAttr], Optional[np.ndarray]]:
    r"""Extract named parameters from a module into a numpy array.

    Only extracts parameters with requires_grad, since it is meant for optimizing.

    Args:
        module: A module with parameters. May specify parameter constraints in
            a `named_parameters_and_constraints` method.
        bounds: A dictionary mapping parameter names t lower and upper bounds.
            of lower and upper bounds. Bounds specified here take precedence
            over bounds on the same parameters specified in the constraints
            registered with the module.
        exclude: A list of parameter names that are to be excluded from extraction.

    Returns:
        3-element tuple containing
        - The parameter values as a numpy array.
        - An ordered dictionary with the name and tensor attributes of each
        parameter.
        - A `2 x n_params` numpy array with lower and upper bounds if at least
        one constraint is finite, and None otherwise.

    Example:
        >>> mll = ExactMarginalLogLikelihood(model.likelihood, model)
        >>> parameter_array, property_dict, bounds_out = module_to_array(mll)
    """
    warn(
        "`module_to_array` is marked for deprecation, consider using "
        "`get_parameters_and_bounds`, `get_parameters_as_ndarray_1d`, or "
        "`get_bounds_as_ndarray` instead.",
        DeprecationWarning,
    )
    param_dict, bounds_dict = get_parameters_and_bounds(
        module=module,
        name_filter=None if exclude is None else get_name_filter(exclude),
        requires_grad=True,
    )
    if bounds is not None:
        bounds_dict.update(bounds)

    # Record tensor metadata and read parameter values to the tape
    param_tape: List[Number] = []
    property_dict = OrderedDict()
    with torch.no_grad():
        for name, param in param_dict.items():
            property_dict[name] = TorchAttr(param.shape, param.dtype, param.device)
            param_tape.extend(param.view(-1).cpu().double().tolist())

    # Extract lower and upper bounds
    start = 0
    bounds_np = None
    params_np = np.asarray(param_tape)
    for name, param in param_dict.items():
        numel = param.numel()
        if name in bounds_dict:
            for row, bound in enumerate(bounds_dict[name]):
                if bound is None:
                    continue

                if torch.is_tensor(bound):
                    if (bound == (2 * row - 1) * inf).all():
                        continue
                    bound = bound.detach().cpu()

                elif bound == (2 * row - 1) * inf:
                    continue

                if bounds_np is None:
                    bounds_np = np.full((2, len(params_np)), ((-inf,), (inf,)))

                bounds_np[row, start : start + numel] = bound
        start += numel

    return params_np, property_dict, bounds_np


def set_params_with_array(
    module: Module, x: np.ndarray, property_dict: Dict[str, TorchAttr]
) -> Module:
    r"""Set module parameters with values from numpy array.

    Args:
        module: Module with parameters to be set
        x: Numpy array with parameter values
        property_dict: Dictionary of parameter names and torch attributes as
            returned by module_to_array.

    Returns:
        Module: module with parameters updated in-place.

    Example:
        >>> mll = ExactMarginalLogLikelihood(model.likelihood, model)
        >>> parameter_array, property_dict, bounds_out = module_to_array(mll)
        >>> parameter_array += 0.1  # perturb parameters (for example only)
        >>> mll = set_params_with_array(mll, parameter_array,  property_dict)
    """
    warn(
        "`_set_params_with_array` is marked for deprecation, consider using "
        "`set_parameters_from_ndarray_1d` instead.",
        DeprecationWarning,
    )
    param_dict = OrderedDict(module.named_parameters())
    start_idx = 0
    for p_name, attrs in property_dict.items():
        # Construct the new tensor
        if len(attrs.shape) == 0:  # deal with scalar tensors
            end_idx = start_idx + 1
            new_data = torch.tensor(
                x[start_idx], dtype=attrs.dtype, device=attrs.device
            )
        else:
            end_idx = start_idx + np.prod(attrs.shape)
            new_data = torch.tensor(
                x[start_idx:end_idx], dtype=attrs.dtype, device=attrs.device
            ).view(*attrs.shape)
        start_idx = end_idx
        # Update corresponding parameter in-place. Disable autograd to update.
        param_dict[p_name].requires_grad_(False)
        param_dict[p_name].copy_(new_data)
        param_dict[p_name].requires_grad_(True)
    return module


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
    warn("`_scipy_objective_and_grad` is marked for deprecation.", DeprecationWarning)
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

    i = 0
    param_dict = OrderedDict(mll.named_parameters())
    grad = np.zeros(sum([tattr.shape.numel() for tattr in property_dict.values()]))
    for p_name in property_dict:
        t = param_dict[p_name]
        size = t.numel()
        if t.requires_grad and t.grad is not None:
            grad[i : i + size] = t.grad.detach().view(-1).cpu().double().clone().numpy()
        i += size

    mll.zero_grad()
    return loss.item(), grad
