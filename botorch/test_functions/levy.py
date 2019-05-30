#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import math

import torch


GLOBAL_MAXIMIZER = [1.0]  # [1.0] * d
GLOBAL_MAXIMUM = 0.0


def neg_levy(X):
    r"""Levy test function.

    d-dimensional function (usually evaluated on `[-10, 10]^d`):

        `f(x) = sin^2(pi w_1) + sum_{i=1}^{d-1} (w_i-1)^2 (1 + 10 sin^2(pi w_i + 1))
                              + (w_d - 1)^2 (1 + sin^2(2 pi w_d))'
        where w_i = 1 + (x_i - 1) / 4 for all i.

    f has one minimizer for its global minimum at

        `z_1 = (1, 1, ..., 1)`

    with `f(z_1) = 0`

    Args:
        X: A Tensor of size `d` or `k x d` (`k` batch evaluations).

    Returns:
        `-f(X)`, the negative value of the Levy function.
    """

    batch = X.ndimension() > 1
    X = X if batch else X.unsqueeze(0)
    W = 1.0 + 0.25 * X - 0.25
    first = (torch.sin(math.pi * W[..., 0])) ** 2
    second = (
        (W[..., :-1] - 1.0) ** 2
        * (1.0 + 10 * (torch.sin(math.pi * W[..., :-1] + 1.0)) ** 2)
    ).sum(dim=-1)
    third = (W[..., -1] - 1.0) ** 2 * (1.0 + (torch.sin(2 * math.pi * W[..., -1])) ** 2)
    result = -1.0 * (first + second + third)
    return result if batch else result.squeeze(0)
