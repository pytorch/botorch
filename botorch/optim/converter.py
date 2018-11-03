#!/usr/bin/env python3

from collections import OrderedDict
from typing import Dict, List, NamedTuple, Tuple

import numpy as np
import torch
from gpytorch.module import Module


class TorchAttr(NamedTuple):
    shape: torch.Size
    dtype: torch.dtype
    device: torch.device


def module_to_array(module: Module) -> Tuple[np.ndarray, Dict[str, TorchAttr]]:
    """Extract named parameters from a module into a numpy array.

    Only extracts parameters with requires_grad, since it is meant for
    optimizing.

    Args:
        module: A module with parameters

    Returns:
        A numpy array with parameter values
        An ordered dictionary with the name and tensor attributes of each
            parameter.
    """
    x: List[np.ndarray] = []
    property_dict = OrderedDict()
    for p_name, t in OrderedDict(module.named_parameters()).items():
        if t.requires_grad:
            property_dict[p_name] = TorchAttr(
                shape=t.shape, dtype=t.dtype, device=t.device
            )
            x.append(t.detach().view(-1).cpu().double().clone().numpy())
    return np.concatenate(x), property_dict


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
