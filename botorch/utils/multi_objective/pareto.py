#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch
from torch import Tensor

# maximum tensor size for simple pareto computation
MAX_BYTES = 5e6


def is_non_dominated(Y: Tensor, deduplicate: bool = True) -> Tensor:
    r"""Computes the non-dominated front.

    Note: this assumes maximization.

    For small `n`, this method uses a highly parallel methodology
    that compares all pairs of points in Y. However, this is memory
    intensive and slow for large `n`. For large `n` (or if Y is larger
    than 5MB), this method will dispatch to a loop-based approach
    that is faster and has a lower memory footprint.

    Args:
        Y: A `(batch_shape) x n x m`-dim tensor of outcomes.
        deduplicate: A boolean indicating whether to only return
            unique points on the pareto frontier.

    Returns:
        A `(batch_shape) x n`-dim boolean tensor indicating whether
        each point is non-dominated.
    """
    n = Y.shape[-2]
    if n == 0:
        return torch.zeros(Y.shape[:-1], dtype=torch.bool, device=Y.device)
    el_size = 64 if Y.dtype == torch.double else 32
    if n > 1000 or n**2 * Y.shape[:-2].numel() * el_size / 8 > MAX_BYTES:
        return _is_non_dominated_loop(Y)

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


def _is_non_dominated_loop(Y: Tensor, maximize: bool = True) -> Tensor:
    r"""Determine which points are non-dominated.

    Compared to `is_non_dominated`, this method is significantly
    faster for large `n` on a CPU and will significant reduce memory
    overhead. However, `is_non_dominated` is faster for smaller problems.

    Args:
        Y: A `(batch_shape) x n x m` Tensor of outcomes.
        maximize: A boolean indicating if the goal is maximization.

    Returns:
        A `(batch_shape) x n`-dim Tensor of booleans indicating whether each point is
            non-dominated.
    """
    is_efficient = torch.ones(*Y.shape[:-1], dtype=bool, device=Y.device)
    for i in range(Y.shape[-2]):
        i_is_efficient = is_efficient[..., i]
        if i_is_efficient.any():
            vals = Y[..., i : i + 1, :]
            if maximize:
                update = (Y > vals).any(dim=-1)
            else:
                update = (Y < vals).any(dim=-1)
            # If an element in Y[..., i, :] is efficient, mark it as efficient
            update[..., i] = i_is_efficient.clone()
            # Only include batches where  Y[..., i, :] is efficient
            # Create a copy
            is_efficient2 = is_efficient.clone()
            if Y.ndim > 2:
                # Set all elements in all batches where Y[..., i, :] is not
                # efficient to False
                is_efficient2[~i_is_efficient] = False
            # Only include elements from in_efficient from the batches
            # where Y[..., i, :] is efficient
            is_efficient[is_efficient2] = update[is_efficient2]
    return is_efficient
