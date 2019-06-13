#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import math

import torch
from torch import Tensor


# This function has infinitely many global maximizers
GLOBAL_MAXIMUM = -0.397887


def neg_aug_branin(X: Tensor) -> Tensor:
    r"""Negative augmented-Branin test function.

    The last dimension of X is the fidelity.
    3-dimensional function with domain `[-5, 10] x [0, 15] * [0,1]`:

        B(x) = (x_2 - (b - 0.1 * (1 - x_3))x_1^2 + c x_1 - r)^2 +
            10 (1-t) cos(x_1) + 10

    Here `b`, `c`, `r` and `t` are constants where `b = 5.1 / (4 * math.pi ** 2)`
    `c = 5 / math.pi`, `r = 6`, `t = 1 / (8 * math.pi)`.
    B has infinitely many minimizers with `x_1 = -pi, pi, 3pi`

    and `B_min = 0.397887`

    Args:
        X: A Tensor of size `3` or `k x 3` (`k` batch evaluations).

    Returns:
        `-B(X)`, the negative value of the augmented-Branin function.
    """
    batch = X.ndimension() > 1
    X = X if batch else X.unsqueeze(0)
    t1 = (
        X[:, 1]
        - (5.1 / (4 * math.pi ** 2) - 0.1 * (1 - X[:, 2])) * X[:, 0] ** 2
        + 5 / math.pi * X[:, 0]
        - 6
    )
    t2 = 10 * (1 - 1 / (8 * math.pi)) * torch.cos(X[:, 0])
    B = t1 ** 2 + t2 + 10
    result = -B
    return result if batch else result.squeeze(0)
