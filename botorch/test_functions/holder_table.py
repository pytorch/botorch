#!/usr/bin/env python3

import math

import torch
from torch import Tensor


GLOBAL_MAXIMIZERS = [
    [8.0550, 9.6646],
    [-8.0550, -9.6646],
    [-8.0550, 9.6646],
    [8.0550, -9.6646],
]
GLOBAL_MAXIMUM = 19.2085


def neg_holder_table(X: Tensor) -> Tensor:
    r"""Negative Holder Table synthetic test function.

    Two-dimensional function (typically evaluated on `[0, 10] x [0, 10]`):

        `H(x) = - | sin(x_1) * cos(x_2) * exp(| 1 - ||x|| / pi | ) |`

    H has 4 global minima with `H(z_i) = -19.2085` at

        `z_1 = ( 8.05502,  9.66459)`
        `z_2 = (-8.05502, -9.66459)`
        `z_3 = (-8.05502,  9.66459)`
        `z_4 = ( 8.05502, -9.66459)`

    Args:
        X: A Tensor of size `2` or `k x 2` (`k` batch evaluations).

    Returns:
        `-H(X)`, the negative value of the standard Holder Table function.
    """
    batch = X.ndimension() > 1
    X = X if batch else X.unsqueeze(0)
    term = torch.abs(1 - torch.norm(X, dim=1) / math.pi)
    H = -torch.abs(torch.sin(X[:, 0]) * torch.cos(X[:, 1]) * torch.exp(term))
    result = -H
    return result if batch else result.squeeze(0)
