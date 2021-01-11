#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch
from torch import Tensor


def is_non_dominated(Y: Tensor, deduplicate: bool = True) -> Tensor:
    r"""Computes the non-dominated front.

    Note: this assumes maximization.

    Args:
        Y: A `(batch_shape) x n x m`-dim tensor of outcomes.
        deduplicate: A boolean indicating whether to only return
            unique points on the pareto frontier.

    Returns:
        A `(batch_shape) x n`-dim boolean tensor indicating whether
        each point is non-dominated.
    """
    Y1 = Y.unsqueeze(-3)
    Y2 = Y.unsqueeze(-2)
    dominates = (Y1 >= Y2).all(dim=-1) & (Y1 > Y2).any(dim=-1)
    nd_mask = ~(dominates.any(dim=-1))
    if deduplicate:
        # remove duplicates
        # find index of first occurrence  of each unique element
        indices = (Y1 == Y2).all(dim=-1).long().argmax(dim=-1)
        keep = torch.zeros_like(nd_mask)
        keep.scatter_(dim=-1, index=indices, value=1.0)
        return nd_mask & keep
    return nd_mask
