#!/usr/bin/env python3

import math

import torch
from torch import Tensor


GLOBAL_MAXIMIZERS = [[-math.pi, 12.275], [math.pi, 2.275], [9.42478, 2.475]]
GLOBAL_MAXIMUM = -0.397887


def neg_branin(X: Tensor) -> Tensor:
    r"""Negative Branin test function.

    Two-dimensional function (usually evaluated on `[-5, 10] x [0, 15]`):

        `B(x) = (x2 - b x_1^2 + c x_1 - r)^2 + 10 (1-t) cos(x_1) + 10`

    B has 3 minimizers for its global minimum at

        `z_1 = (-pi, 12.275), z_2 = (pi, 2.275), z_3 = (9.42478, 2.475)`

    with `B(z_i) = -0.397887`

    Args:
        X: A Tensor of size `2` or `k x 2` (`k` batch evaluations).

    Returns:
        `-B(X)`, the negative value of the standard Branin function.
    """
    batch = X.ndimension() > 1
    X = X if batch else X.unsqueeze(0)
    t1 = X[:, 1] - 5.1 / (4 * math.pi ** 2) * X[:, 0] ** 2 + 5 / math.pi * X[:, 0] - 6
    t2 = 10 * (1 - 1 / (8 * math.pi)) * torch.cos(X[:, 0])
    B = t1 ** 2 + t2 + 10
    result = -B
    return result if batch else result.squeeze(0)
