#!/usr/bin/env python3

from typing import Dict, List, NamedTuple

import numpy as np
import torch
from torch import Tensor


class TorchAttr(NamedTuple):
    shape: torch.Size
    dtype: torch.dtype
    device: torch.device


class NumpyParameters(NamedTuple):
    x: np.array
    property_dict: Dict[str, TorchAttr]  # assumed ordered


def numpy_to_state_dict(numpy_params: NumpyParameters) -> Dict[str, Tensor]:
    """Unflatten a numpy array and metadata into a pytorch state dict."""
    x = numpy_params.x
    start_idx = 0
    state_dict: Dict[str, Tensor] = {}
    for p, attrs in numpy_params.property_dict.items():
        end_idx = start_idx + np.prod(attrs.shape)
        state_dict[p] = torch.tensor(
            x[start_idx:end_idx], dtype=attrs.dtype, device=attrs.device
        ).view(*attrs.shape)
        start_idx = end_idx
    return state_dict


def state_dict_to_numpy(state_dict: Dict[str, Tensor]) -> NumpyParameters:
    """Flatten a pytorch state dict into a numpy array and metadata."""
    x: List[np.ndarray] = []
    pdict: Dict[str, TorchAttr] = {}
    for p, t in state_dict.items():
        pdict[p] = TorchAttr(shape=t.shape, dtype=t.dtype, device=t.device)
        x.append(t.detach().view(-1).double().numpy())
    return NumpyParameters(x=np.concatenate(x), property_dict=pdict)
