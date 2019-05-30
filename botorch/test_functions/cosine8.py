#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import math

import torch


GLOBAL_MAXIMIZER = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
GLOBAL_MAXIMUM = 0.8


def cosine8(X):
    r"""8d Cosine Mixture test function.

    8-dimensional function (usually evaluated on `[-1, 1]^8`):

        `f(x) = 0.1 sum_{i=1}^8 cos(5 pi x_i) - sum_{i=1}^8 x_i^2'

    f has one maximizer for its global maximum at

        `z_1 = (0, 0, ..., 0)`

    with `f(z_1) = 0.8`

    Args:
        X: A Tensor of size `8` or `k x 8` (`k` batch evaluations).

    Returns:
        `f(X)`, the value of the 8d Cosine Mixture function.
    """

    batch = X.ndimension() > 1
    X = X if batch else X.unsqueeze(0)
    result = 0.1 * (torch.cos(5.0 * math.pi * X)).sum(dim=-1) - (X ** 2).sum(dim=-1)
    return result if batch else result.squeeze(0)
