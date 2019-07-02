#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from torch import Tensor


GLOBAL_MAXIMIZER = [1.0]
GLOBAL_MAXIMUM = 0.0


def neg_aug_rosenbrock(X: Tensor):
    r"""Augmented-Rosenbrock test function.

    d-dimensional function (usually evaluated on `[-5, 10]^(d-2) * [0, 1]^2`),
    the last two dimensions are the fidelity parameters:

        `f(x) = sum_{i=1}^{d-1} (100 (x_{i+1} - x_i^2 + 0.1 * (1-x_{d-1}))^2
                + (x_i - 1 + 0.1 * (1 - x_d)^2)^2)'

    f has one minimizer for its global minimum at

        `z_1 = (1, 1, ..., 1)`

    with `f(z_i) = 0.0`

    Args:
        X: A Tensor of size `d` or `k x d` (`k` batch evaluations).

    Returns:
        `-f(X)`, the negative value of the Augmented-Rosenbrock function.
    """
    batch = X.ndimension() > 1
    X = X if batch else X.unsqueeze(0)
    X_curr = X[..., :-3]
    X_next = X[..., 1:-2]
    result = (
        -(
            100 * (X_next - X_curr ** 2 + 0.1 * (1 - X[..., -2].unsqueeze(-1))) ** 2
            + (X_curr - 1 + 0.1 * (1 - X[..., -1].unsqueeze(-1)) ** 2) ** 2
        )
    ).sum(dim=-1)
    return result if batch else result.squeeze(0)
