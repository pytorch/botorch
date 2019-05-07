#!/usr/bin/env python3

import math
from torch import Tensor


GLOBAL_MAXIMIZER = 0.0
GLOBAL_MAXIMUM = 0.0


def neg_ackley(X: Tensor) -> Tensor:
    r"""Negative Ackley test function.

    d-dimensional function (usually evaluated on the hypercube `[-32, 32]^d`):

        `H(x) = -a \exp(-b \sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2})) -\exp(\frac{1}{d} \sum_{i=1}^d \cos(c x_i^2)) + a + \exp(1)`

    H has a single global mininimum `H(z) = 0` at `z = [0]^d`

    Args:
        X: A Tensor of size `d` or `k x d` (`k` batch evaluations)

    Returns:
        `-H(X)`, the negative value of the standard Ackley function.
    """
    a, b, c = 20, 0.2, 2 * math.pi
    batch = X.ndimension() > 1
    d = X.size
    X = X if batch else X.unsqueeze(0)

    result = (
        -a * (-b * X.pow(2).mean(dim=1).sqrt()).exp() -
        (c * X).cos().mean(dim=1).exp() +
        a + math.exp(1)
    )
    return result if batch else result.squeeze(0)
