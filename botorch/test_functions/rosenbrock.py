#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


GLOBAL_MAXIMIZER = [1.0]  # [1.0] * d
GLOBAL_MAXIMUM = 0.0


def neg_rosenbrock(X):
    r"""Rosenbrock test function.

    d-dimensional function (usually evaluated on `[-5, 10]^d`):

        `f(x) = sum_{i=1}^{d-1} (100 (x_{i+1} - x_i^2)^2 + (x_i - 1)^2)'

    f has one minimizer for its global minimum at

        `z_1 = (1, 1, ..., 1)`

    with `f(z_i) = 0.0`

    Args:
        X: A Tensor of size `d` or `k x d` (`k` batch evaluations).

    Returns:
        `-f(X)`, the negative value of the Rosenbrock function.
    """
    batch = X.ndimension() > 1
    X = X if batch else X.unsqueeze(0)
    X_curr = X[..., :-1]
    X_next = X[..., 1:]
    result = -(100 * (X_next - X_curr ** 2) ** 2 + (X_curr - 1) ** 2).sum(dim=-1)
    return result if batch else result.squeeze(0)
