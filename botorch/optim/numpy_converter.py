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
from re import Pattern
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    Union,
)

import numpy as np
import torch
from torch.nn import Module, Parameter

ParameterBounds = Dict[str, Tuple[Optional[float], Optional[float]]]


class TorchAttr(NamedTuple):
    shape: torch.Size
    dtype: torch.dtype
    device: torch.device


def create_name_filter(
    patterns: Iterator[Union[Pattern, str]]
) -> Callable[[Union[str, Tuple[str, Any, ...]]], bool]:
    r"""Returns a binary function that filters strings (or iterables whose first
    element is a string) according to a bank of excluded patterns. Typically, used
    in conjunction with generators such as `module.named_parameters()`.

    Args:
        patterns: A collection of regular expressions or strings that
            define the set of names to be excluded.

    Returns:
        A binary function indicating whether or not an item should be filtered.
    """
    names = set()
    _patterns = set()
    for pattern in patterns:
        if isinstance(pattern, str):
            names.add(pattern)
        elif isinstance(pattern, Pattern):
            _patterns.add(pattern)
        else:
            raise TypeError

    def name_filter(item: Union[str, Tuple[str, Any, ...]]) -> bool:
        name = item if isinstance(item, str) else next(iter(item))
        if name in names:
            return False

        for pattern in _patterns:
            if pattern.search(name):
                return False

        return True

    return name_filter


def get_parameters_and_bounds(
    module: Module,
    name_filter: Optional[Callable[[str], bool]] = None,
    requires_grad: Optional[bool] = None,
    default_bounds: Tuple[float, float] = (-float("inf"), float("inf")),
) -> Tuple[Dict[str, Parameter], Dict[str, ParameterBounds]]:
    r"""Helper method for extracting parameters and feasible ranges thereof.

    Args:
        module: The target module from which parameters are to be extracted.
        name_filter: Optional Boolean function used to filter parameters by name.
        requires_grad: Optional Boolean used to filter parameters based on whether
            or not their require_grad attribute matches the user provided value.
        default_bounds: Default lower and upper bounds for constrained parameters
            with `None` typed bounds.

    Returns:
        0: Dictionary mapping names to Parameters.
        1: Dictionary mapping names of constrained parameters to ParameterBounds.
    """
    if hasattr(module, "named_parameters_and_constraints"):
        bounds = {}
        params = {}
        for name, param, constraint in module.named_parameters_and_constraints():
            if (requires_grad is None or (param.requires_grad == requires_grad)) and (
                name_filter is None or name_filter(name)
            ):
                params[name] = param
                if constraint is None:
                    continue

                bounds[name] = tuple(
                    default if bound is None else constraint.inverse_transform(bound)
                    for (bound, default) in zip(constraint, default_bounds)
                )
    else:
        bounds = {}
        params = {
            name: param
            for name, param in module.named_parameters()
            if name_filter is None or name_filter(name)
        }

    return params, bounds


def module_to_array(
    module: Module,
    bounds: Optional[ParameterBounds] = None,
    exclude: Optional[Set[str]] = None,
) -> Tuple[np.ndarray, Dict[str, TorchAttr], Optional[np.ndarray]]:
    r"""Extract named parameters from a module into a numpy array.

    Only extracts parameters with requires_grad, since it is meant for optimizing.

    Args:
        module: A module with parameters. May specify parameter constraints in
            a `named_parameters_and_constraints` method.
        bounds: A ParameterBounds dictionary mapping parameter names to tuples
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
    param_dict, bounds_dict = get_parameters_and_bounds(
        module=module,
        name_filter=None if exclude is None else create_name_filter(exclude),
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
