#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
from torch import Tensor

from .hartmann6 import ALPHA, A, P


GLOBAL_MAXIMIZER = [0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573, 1]


def neg_aug_hartmann6(X: Tensor) -> Tensor:
    r"""Negative augmented Hartmann6 test function.

    The last dimension of X is the fidelity parameter.
    7-dimensional function (typically evaluated on `[0, 1]^7`):

        H(x) = -(ALPHA_1 - 0.1 * (1-x_7)) * exp(- sum_{j=1}^6 A_1j (x_j - P_1j) ** 2) -
            sum_{i=2}^4 ALPHA_i exp( - sum_{j=1}^6 A_ij (x_j - P_ij) ** 2)

    H has unique global minimizer
    x = [0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573, 1]

    with `H_min = -3.32237`

    Args:
        X: A Tensor of size `7` or `k x 7` (k batch evaluations).

    Returns:
        `-H(X)`, the negative value of the augmented Hartmann6 function.
    """
    batch = X.ndimension() > 1
    X = X if batch else X.unsqueeze(0)
    inner_sum = torch.sum(
        X.new(A) * (X[:, :6].unsqueeze(1) - 0.0001 * X.new(P)) ** 2, dim=2
    )
    alpha1 = ALPHA[0] - 0.1 * (1 - X[:, 6])
    H = (
        -torch.sum(X.new(ALPHA)[1:] * torch.exp(-inner_sum)[:, 1:], dim=1)
        - alpha1 * torch.exp(-inner_sum)[:, 0]
    )
    result = -H
    return result if batch else result.squeeze(0)
