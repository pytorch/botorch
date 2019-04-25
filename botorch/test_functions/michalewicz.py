#!/usr/bin/env python3

import math

import torch
from torch import Tensor


GLOBAL_MAXIMUM = 9.6601517
GLOBAL_MAXIMIZER = [
    2.202906,
    1.570796,
    1.284992,
    1.923058,
    1.720470,
    1.570796,
    1.454414,
    1.756087,
    1.655717,
    1.570796,
]


def neg_michalewicz(X: Tensor) -> Tensor:
    r"""Negative 10-dim Michalewicz test function.

    10-dim function (usually evaluated on hypercube [0, pi]^10):

        `M(x) = sum_{i=1}^10 sin(x_i) (sin(i x_i^2 / pi)^20)`

    Args:
        X: A Tensor of size `10` or `k x 10` (`k` batch evaluations).

    Returns:
        `-M(X)`, the negative value of the Michalewicz function.
    """
    batch = X.ndimension() > 1
    X = X if batch else X.unsqueeze(0)
    a = 1 + torch.arange(10, device=X.device, dtype=X.dtype)
    result = torch.sum(torch.sin(X) * torch.sin(a * X ** 2 / math.pi) ** 20, dim=-1)
    return result if batch else result.squeeze(0)
