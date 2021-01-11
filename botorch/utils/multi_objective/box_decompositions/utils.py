#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from botorch.exceptions.errors import BotorchTensorDimensionError
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
