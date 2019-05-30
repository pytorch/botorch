#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch


GLOBAL_MAXIMIZER = [
    4.00074720382690,
    3.999509811401367,
    4.000747203826904,
    3.999509811401367,
]
GLOBAL_MAXIMUM = 10.5364

A = torch.tensor(
    [
        [4.0, 4.0, 4.0, 4.0],
        [1.0, 1.0, 1.0, 1.0],
        [8.0, 8.0, 8.0, 8.0],
        [6.0, 6.0, 6.0, 6.0],
        [3.0, 7.0, 3.0, 7.0],
        [2.0, 9.0, 2.0, 9.0],
        [5.0, 3.0, 5.0, 3.0],
        [8.0, 1.0, 8.0, 1.0],
        [6.0, 2.0, 6.0, 2.0],
        [7.0, 3.6, 7.0, 3.6],
    ]
)

C = torch.tensor([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])


def neg_shekel(X):
    r"""Shekel test function.

    4-dimensional function (usually evaluated on `[0, 10]^4`):

        `f(x) = -sum_{i=1}^10 (sum_{j=1}^4 (x_j - A_{ji})^2 + C_i)^{-1}`

    f has one minimizer for its global minimum at

        `z_1 = (4, 4, 4, 4)`

    with `f(z_1) = -10.5363`

    Args:
        X: A Tensor of size `4` or `k x 4` (`k` batch evaluations).

    Returns:
        `-f(X)`, the negative value of the Levy function.
    """

    inner = ((X.unsqueeze(dim=-2) - A.to(X)) ** 2).sum(dim=-1)
    return (1 / (inner + C.to(X))).sum(dim=-1)
