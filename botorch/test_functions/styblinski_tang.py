#!/usr/bin/env python3

from torch import Tensor


GLOBAL_MINIMIZER = -2.903534
GLOBAL_MINIMUM = -39.166166


def styblinski_tang(X: Tensor):
    """Styblinski-Tang synthetic test function supporting batch evaluation.

    H(x) = 0.5 * sum_{i=1}^d (x_i^4 - 16 * x_i^2 + 5 * x_i)

    H is usually evaluated on the hypercube [-5, 5]^d

    H has a single global mininimum H(z) = -39.166166 * d at z = -2.903534

    Args:
        X (Tensor): A Tensor of size d or k x d (k batch evaluations)

    """
    batch = X.ndimension() > 1
    X = X if batch else X.unsqueeze(0)
    result = 0.5 * (X ** 4 - 16 * X ** 2 + 5 * X).sum(dim=1)
    return result if batch else result.squeeze(0)
