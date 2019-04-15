#!/usr/bin/env python3

from torch import Tensor


GLOBAL_MAXIMIZER = -2.903534
GLOBAL_MAXIMUM = 39.166166


def neg_styblinski_tang(X: Tensor) -> Tensor:
    r"""Negative Styblinski-Tang test function.

    d-dimensional function (usually evaluated on the hypercube `[-5, 5]^d`):

        `H(x) = 0.5 * sum_{i=1}^d (x_i^4 - 16 * x_i^2 + 5 * x_i)`

    H has a single global mininimum `H(z) = -39.166166 * d` at `z = [-2.903534]^d`

    Args:
        X: A Tensor of size `d` or `k x d` (`k` batch evaluations)

    Returns:
        `-H(X)`, the negative value of the standard Styblinski-Tang function.
    """
    batch = X.ndimension() > 1
    X = X if batch else X.unsqueeze(0)
    H = 0.5 * (X ** 4 - 16 * X ** 2 + 5 * X).sum(dim=1)
    result = -H
    return result if batch else result.squeeze(0)
