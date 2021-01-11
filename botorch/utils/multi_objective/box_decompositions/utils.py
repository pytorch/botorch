#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Utilities for box decomposition algorithms."""

from typing import Optional

import torch
from botorch.exceptions.errors import BotorchTensorDimensionError, UnsupportedError
from botorch.utils.multi_objective.pareto import is_non_dominated
from torch import Size, Tensor


def _expand_ref_point(ref_point: Tensor, batch_shape: Size) -> Tensor:
    r"""Expand reference point to the proper batch_shape.

    Args:
        ref_point: A `(batch_shape) x m`-dim tensor containing the reference
            point.
        batch_shape: The batch shape.

    Returns:
        A `batch_shape x m`-dim tensor containing the expanded reference point
    """
    if ref_point.shape[:-1] != batch_shape:
        if ref_point.ndim > 1:
            raise BotorchTensorDimensionError(
                "Expected ref_point to be a `batch_shape x m` or `m`-dim tensor, "
                f"but got {ref_point.shape}."
            )
        ref_point = ref_point.view(
            *(1 for _ in batch_shape), ref_point.shape[-1]
        ).expand(batch_shape + ref_point.shape[-1:])
    return ref_point


def _pad_batch_pareto_frontier(
    Y: Tensor,
    ref_point: Tensor,
    is_pareto: bool = False,
    feasibility_mask: Optional[Tensor] = None,
) -> Tensor:
    r"""Get a batch Pareto frontier by padding the pareto frontier with repeated points.

    This assumes maximization.

    Args:
        Y: A `(batch_shape) x n x m`-dim tensor of points
        ref_point: a `(batch_shape) x m`-dim tensor containing the reference point
        is_pareto: a boolean indicating whether the points in Y are already
            non-dominated.
        feasibility_mask: A `(batch_shape) x n`-dim tensor of booleans indicating
            whether each point is feasible.

    Returns:
        A `(batch_shape) x max_num_pareto x m`-dim tensor of padded Pareto
            frontiers.
    """
    tkwargs = {"dtype": Y.dtype, "device": Y.device}
    ref_point = ref_point.unsqueeze(-2)
    batch_shape = Y.shape[:-2]
    if len(batch_shape) > 1:
        raise UnsupportedError(
            "_pad_batch_pareto_frontier only supports a single "
            f"batch dimension, but got {len(batch_shape)} "
            "batch dimensions."
        )
    if feasibility_mask is not None:
        # set infeasible points to be the reference point (corresponding to the batch)
        Y = torch.where(feasibility_mask.unsqueeze(-1), Y, ref_point)
    if not is_pareto:
        pareto_mask = is_non_dominated(Y)
    else:
        pareto_mask = torch.ones(Y.shape[:-1], dtype=torch.bool, device=Y.device)
    better_than_ref = (Y > ref_point).all(dim=-1)
    # is_non_dominated assumes maximization
    # TODO: filter out points that are worse than the reference point first here
    pareto_mask = pareto_mask & better_than_ref
    if len(batch_shape) == 0:
        return Y[pareto_mask]
    # Note: in the batch case, the Pareto frontier is padded by repeating
    # a Pareto point. This ensures that the padded box-decomposition has
    # the same number of points, which enables fast batch operations.
    max_n_pareto = pareto_mask.sum(dim=-1).max().item()
    pareto_Y = torch.empty(*batch_shape, max_n_pareto, Y.shape[-1], **tkwargs)
    for i, pareto_i in enumerate(pareto_mask):
        pareto_i = Y[i, pareto_mask[i]]
        n_pareto = pareto_i.shape[0]
        if n_pareto > 0:
            pareto_Y[i, :n_pareto] = pareto_i
            # pad pareto_Y, so that all batches have the same size Pareto set
            pareto_Y[i, n_pareto:] = pareto_i[-1]
        else:
            # if there are no pareto points in this batch, use the reference
            # point
            pareto_Y[i, :] = ref_point[i]
    return pareto_Y
