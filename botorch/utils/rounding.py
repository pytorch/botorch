#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch
from torch import Tensor


def approximate_round(X: Tensor, tau: float = 1e-3) -> Tensor:
    r"""Diffentiable approximate rounding function.

    This method is a piecewise approximation of a rounding function where
    each piece is a hyperbolic tangent function.

    Args:
        X: The tensor to round to the nearest integer (element-wise).
        tau: A temperature hyperparameter.

    Returns:
        The approximately rounded input tensor.
    """
    offset = X.floor()
    scaled_remainder = (X - offset - 0.5) / tau
    rounding_component = (torch.tanh(scaled_remainder) + 1) / 2
    return offset + rounding_component
