#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from torch import Tensor


def is_non_dominated(Y: Tensor) -> Tensor:
    r"""Computes the non-dominated front.

    Note: this assumes maximization.

    Args:
        Y: a `(batch_shape) x n x m`-dim tensor of outcomes.

    Returns:
        A `(batch_shape) x n`-dim boolean tensor indicating whether
        each point is non-dominated.
    """
    expanded_shape = Y.shape[:-2] + Y.shape[-2:-1] + Y.shape[-2:]
    Y1 = Y.unsqueeze(-3).expand(expanded_shape)
    Y2 = Y.unsqueeze(-2).expand(expanded_shape)
    dominates = (Y1 >= Y2).all(dim=-1) & (Y1 > Y2).any(dim=-1)
    return ~(dominates.any(dim=-1))
