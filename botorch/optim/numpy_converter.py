#!/usr/bin/env python3

from collections import OrderedDict
from math import inf
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import torch
from torch.nn import Module


ParameterBounds = Dict[str, Tuple[Optional[float], Optional[float]]]


class TorchAttr(NamedTuple):
    shape: torch.Size
    dtype: torch.dtype
    device: torch.device


def module_to_array(
    module: Module, bounds: Optional[ParameterBounds] = None
) -> Tuple[np.ndarray, Dict[str, TorchAttr], Optional[np.ndarray]]:
    """Extract named parameters from a module into a numpy array.

    Only extracts parameters with requires_grad, since it is meant for optimizing.

    Args:
        module: A module with parameters. May specify parameter constraints in
            a `parameter_bounds` attribute.
        bounds: A ParameterBounds dictionary mapping parameter names to tuples of
            lower and upper bounds. Bounds specified here take precedence over
            bounds specified in the `parameter_bounds` attribute of the module.

    Returns:
        A numpy array with parameter values
        An ordered dictionary with the name and tensor attributes of each parameter.
        A `2 x n_params` numpy array with lower and upper bounds if at least one
            constraint is finite, and None otherwise

    """
    x: List[np.ndarray] = []
    lower: List[np.ndarray] = []
    upper: List[np.ndarray] = []
    property_dict = OrderedDict()

    # extract parameter bounds from module.model.parameter_bounds and
    # module.likelihood.parameter_bounds (if present)
    model_bounds = getattr(getattr(module, "model", None), "parameter_bounds", {})
    bounds_ = {".".join(["model", key]): val for key, val in model_bounds.items()}
    likelihood_bounds = getattr(
        getattr(module, "likelihood", None), "parameter_bounds", {}
    )
    bounds_.update(
        {".".join(["likelihood", key]): val for key, val in likelihood_bounds.items()}
    )
    # update with user-supplied bounds
    if bounds is not None:
        bounds_.update(bounds)

    for p_name, t in module.named_parameters():
        if t.requires_grad:
            property_dict[p_name] = TorchAttr(
                shape=t.shape, dtype=t.dtype, device=t.device
            )
            x.append(t.detach().view(-1).cpu().double().clone().numpy())
            # construct bounds
            if bounds_:
                l, u = bounds_.get(p_name, (-inf, inf))
                # check for Nones here b/c it may be passed in manually in bounds
                lower.append(np.full(t.nelement(), l if l is not None else -inf))
                upper.append(np.full(t.nelement(), u if u is not None else inf))

    x_out = np.concatenate(x)
    bounds_out = None
    if bounds_:
        if not all(np.isinf(b).all() for lu in (lower, upper) for b in lu):
            bounds_out = np.stack([np.concatenate(lower), np.concatenate(upper)])
    return x_out, property_dict, bounds_out


def set_params_with_array(
    module: Module, x: np.ndarray, property_dict: Dict[str, TorchAttr]
) -> Module:
    """Set module parameters with values from numpy array.

    Args:
        module: Module with parameters to be set
        x: Numpy array with parameter values
        property_dict: Dictionary of parameter names and torch attributes as
            returned by module_to_array.

    Returns:
        module with parameters updated in-place.
    """
    param_dict = OrderedDict(module.named_parameters())
    start_idx = 0
    for p_name, attrs in property_dict.items():
        # Construct the new tensor
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
