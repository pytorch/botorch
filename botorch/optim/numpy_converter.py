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
from typing import Dict, List, NamedTuple, Optional, Set, Tuple

import numpy as np
import torch
from torch.nn import Module


ParameterBounds = Dict[str, Tuple[Optional[float], Optional[float]]]


class TorchAttr(NamedTuple):
    shape: torch.Size
    dtype: torch.dtype
    device: torch.device


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
    x: List[np.ndarray] = []
    lower: List[np.ndarray] = []
    upper: List[np.ndarray] = []
    property_dict = OrderedDict()
    exclude = set() if exclude is None else exclude

    # get bounds specified in model (if any)
    bounds_: ParameterBounds = {}
    if hasattr(module, "named_parameters_and_constraints"):
        for param_name, _, constraint in module.named_parameters_and_constraints():
            if constraint is not None and not constraint.enforced:
                bounds_[param_name] = constraint.lower_bound, constraint.upper_bound

    # update with user-supplied bounds (overwrites if already exists)
    if bounds is not None:
        bounds_.update(bounds)

    for p_name, t in module.named_parameters():
        if p_name not in exclude and t.requires_grad:
            property_dict[p_name] = TorchAttr(
                shape=t.shape, dtype=t.dtype, device=t.device
            )
            x.append(t.detach().view(-1).cpu().double().clone().numpy())
            # construct bounds
            if bounds_:
                l_, u_ = bounds_.get(p_name, (-inf, inf))
                if torch.is_tensor(l_):
                    l_ = l_.cpu().detach()
                if torch.is_tensor(u_):
                    u_ = u_.cpu().detach()
                # check for Nones here b/c it may be passed in manually in bounds
                lower.append(np.full(t.nelement(), l_ if l_ is not None else -inf))
                upper.append(np.full(t.nelement(), u_ if u_ is not None else inf))

    x_out = np.concatenate(x)
    bounds_out = None
    if bounds_:
        if not all(np.isinf(b).all() for lu in (lower, upper) for b in lu):
            bounds_out = np.stack([np.concatenate(lower), np.concatenate(upper)])
    return x_out, property_dict, bounds_out


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
