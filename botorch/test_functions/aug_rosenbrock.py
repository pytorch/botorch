#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from torch import Tensor


# This function has infinitely many global maximizers.
GLOBAL_MAXIMUM = 0.0


def neg_aug_rosenbrock(X: Tensor):
    r"""Augmented Rosenbrock test function.

    4-dimensional function, the last two dimensions are for fidelity parameters,
    X is usually in [-5,10]^2 * [0,1]^2:

        f(x) = 100 (x_2 - x_1^2 + 0.1 * (1 - x_3))^2 + (x_1 - 1 + 0.1 * (1 - x_4)^2)^2

    f has infinitely many minimizers, with `f_min = 0.0`

    Args:
        X: A Tensor of size `4` or `k x 4` (`k` batch evaluations).
    Returns:
        `-f(X)`, the negative value of the augmented Rosenbrock function.
    """
    batch = X.ndimension() > 1
    X = X if batch else X.unsqueeze(0)
    result = (
        -100 * (X[:, 1] - X[:, 0] ** 2 + 0.1 * (1 - X[:, 2])) ** 2
        + (X[:, 0] - 1 + 0.1 * (1 - X[:, 3]) ** 2) ** 2
    )
    return result if batch else result.squeeze(0)
