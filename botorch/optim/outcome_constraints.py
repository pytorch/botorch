#!/usr/bin/env python3

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor


ParameterBounds = Dict[str, Tuple[Optional[float], Optional[float]]]


def soft_eval_constraint(lhs: Tensor, eta: float = 1e-3) -> Tensor:
    """Element-wise evaluation of a constraint in a 'soft' fashion

    `value(x) = 1 / (1 + exp(x / eta))`

    Args:
        lhs (Tensor): The left hand side of the constraint `lhs <= 0`.
        eta (float): The temperature parameter of the softmax function. As eta
            grows larger, this approximates the Heaviside step function.

    Returns:
        Tensor: element-wise 'soft' feasibility indicator of the same shape as lhs.
            For each element x, value(x) -> 0 as x becomes positive, and value(x) -> 1
            as x becomes negative.

    """
    if eta <= 0:
        raise ValueError("eta must be positive")
    return torch.sigmoid(-lhs / eta)
