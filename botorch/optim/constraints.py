#!/usr/bin/env python3

from torch import Tensor
from torch.nn.functional import sigmoid


def soft_eval_constraint(lhs: Tensor, eta: float = 1e-3):
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
    return sigmoid(-lhs / eta)
