#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import math

import torch
from torch import Tensor


A = 20
B = 0.2
C = 2 * math.pi

GLOBAL_MAXIMIZER = [0.0]  # [0.0] * d
GLOBAL_MAXIMUM = 0.0


def neg_ackley(X: Tensor) -> Tensor:
    r"""Negative Ackley test function.

    d-dimensional function (usually evaluated on `[-32.768, 32.768]^d`):

        `f(x) = -A exp(-B sqrt(1/d sum_{i=1}^d x_i^2))
                    - exp(1/d sum_{i=1}^d cos(c x_i))
                    + A + exp(1)`

    f has one minimizer for its global minimum at

        `z_1 = (0, 0, ..., 0)`

    with `f(z_1) = 0`

    Args:
        X: A Tensor of size `d` or `k x d` (`k` batch evaluations).

    Returns:
        `-f(X)`, the negative value of the standard Ackley function.
    """

    batch = X.ndimension() > 1
    X = X if batch else X.unsqueeze(0)
    norm = torch.norm(X, p=2, dim=-1)
    first = -A * torch.exp(-B * norm / math.sqrt(X.size(-1)))
    second = -torch.exp(torch.cos(C * X).mean(dim=-1))
    result = -(first + second + A + math.e)
    return result if batch else result.squeeze(0)
