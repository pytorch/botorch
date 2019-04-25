#!/usr/bin/env python3

import torch
from torch import Tensor


GLOBAL_MAXIMUM = 959.6407
GLOBAL_MAXIMIZER = [512, 404.2319]


def neg_eggholder(X: Tensor) -> Tensor:
    r"""Negative Eggholder test function.

    Two-dimensional function (usually evaluated on `[-512, 512]^2`):

        `E(x) = (x_2 + 47) sin(R1(x)) - x_1 * sin(R2(x))`
        `R1(x) = sqrt(|x_2 + x_1 / 2 + 47|)`
        `R2(x) = sqrt(|x_1 - (x_2 + 47)|)`

    Args:
        X: A Tensor of size `2` or `k x 2` (`k` batch evaluations).

    Returns:
        `-E(X)`, the negative value of the Eggholder function.
    """
    batch = X.ndimension() > 1
    X = X if batch else X.unsqueeze(0)
    a = (X[:, 1] + X[:, 0] / 2 + 47).abs().sqrt()
    b = (X[:, 0] - (X[:, 1] + 47)).abs().sqrt()
    result = (X[:, 1] + 47) * torch.sin(a) + X[:, 0] * torch.sin(b)
    return result if batch else result.squeeze(0)
