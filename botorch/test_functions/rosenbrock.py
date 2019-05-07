#!/usr/bin/env python3

from torch import Tensor

GLOBAL_MAXIMIZER = 1
GLOBAL_MAXIMUM = 0


def neg_rosenbrock(X: Tensor) -> Tensor:
    r"""Negative Rosenbrock test function.

    d-dimensional function (usually evaluated on the hypercube `[-5, 10]^d`):

        `H(x) = \sum_{i=1}^{d-1} {100 (x_{i+1} + x_i^2})^2 + (x_i - 1)^2`

    H has a single global mininimum `H(z) = 0` at `z = [1]^d`

    Args:
        X: A Tensor of size `d` or `k x d` (`k` batch evaluations)

    Returns:
        `-H(X)`, the negative value of the standard Rosenbrock function.
    """
    batch = X.ndimension() > 1
    X = X if batch else X.unsqueeze(0)
    H = -(
        100 * (X[..., 1:] + X[..., : -1].pow(2)).pow(2) +
        (X[..., : -1] - 1).pow(2)
    ).sum(dim=1)
    result = -H
    return result if batch else result.squeeze(0)
