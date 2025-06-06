#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Utilities for interfacing Numpy and Torch."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from itertools import tee

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor


torch_to_numpy_dtype_dict = {
    torch.bool: bool,
    torch.uint8: np.uint8,
    torch.int8: np.int8,
    torch.int16: np.int16,
    torch.int32: np.int32,
    torch.int64: np.int64,
    torch.float16: np.float16,
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.complex64: np.complex64,
    torch.complex128: np.complex128,
}


def as_ndarray(
    values: Tensor, dtype: np.dtype | None = None, inplace: bool = True
) -> npt.NDArray:
    r"""Helper for going from torch.Tensor to numpy.ndarray.

    Args:
        values: Tensor to be converted to ndarray.
        dtype: Optional numpy.dtype for the converted tensor.
        inplace: Boolean indicating whether memory should be shared if possible.

    Returns:
        An ndarray with the same data as `values`.
    """
    with torch.no_grad():
        out = values.cpu()  # maybe transfer to cpu

        # Determine whether or not to `clone`
        if (
            # cond 1: are we not in `inplace` mode?
            not inplace
            # cond 2: did we already copy when calling `cpu` above?
            and out.device == values.device
            # cond 3: will we copy when calling `astype` below?
            and (dtype is None or out.dtype == torch_to_numpy_dtype_dict[dtype])
        ):
            out = out.clone()

        # Convert to ndarray and maybe cast to `dtype`
        out = out.numpy()
        return out.astype(dtype, copy=False)


def get_tensors_as_ndarray_1d(
    tensors: Iterator[Tensor] | dict[str, Tensor],
    out: npt.NDArray | None = None,
    dtype: np.dtype | str | None = None,
    as_array: Callable[[Tensor], npt.NDArray] = as_ndarray,
) -> npt.NDArray:
    # Create a pair of iterators, one for setup and one for data transfer
    named_tensors_iter, named_tensors_iter2 = tee(
        iter(tensors.items()) if isinstance(tensors, dict) else enumerate(tensors), 2
    )

    # Use `named_tensors_iter` to get size of `out` and `dtype` when None
    try:
        name, tnsr = next(named_tensors_iter)
    except StopIteration:
        raise RuntimeError(f"Argument `tensors` with type {type(tensors)} is empty.")
    size = tnsr.numel() + sum(tnsr.numel() for _, tnsr in named_tensors_iter)
    dtype = torch_to_numpy_dtype_dict[tnsr.dtype] if dtype is None else dtype

    # Preallocate or validate `out`
    if out is None:  # use first tensor as a reference when `dtype` is None
        out = np.empty([size], dtype=dtype)
    elif out.ndim != 1:
        raise ValueError(f"Expected a vector for `out`, but out.shape={out.shape}.")
    elif out.size != size:
        raise ValueError(
            f"Size of `parameters` ({size}) does not match size of `out` ({out.size})."
        )

    # Use `named_tensors_iter2` to transfer data from `tensors` to `out`
    index = 0
    for name, tnsr in named_tensors_iter2:
        try:
            size = tnsr.numel()
            out[index : index + size] = as_array(tnsr.view(-1))
            index += size
        except Exception as e:
            raise RuntimeError(
                "`get_tensors_as_ndarray_1d` failed while copying values from "
                f"tensor {name}; rethrowing original exception."
            ) from e

    return out


def set_tensors_from_ndarray_1d(
    tensors: Iterator[Tensor] | dict[str, Tensor],
    array: npt.NDArray,
) -> None:
    r"""Sets the values of one more tensors based off of a vector of assignments."""
    named_tensors_iter = (
        iter(tensors.items()) if isinstance(tensors, dict) else enumerate(tensors)
    )
    with torch.no_grad():
        index = 0
        for name, tnsr in named_tensors_iter:
            try:
                size = tnsr.numel()
                vals = array[index : index + size] if tnsr.ndim else array[index]
                tnsr.copy_(
                    torch.as_tensor(vals, device=tnsr.device, dtype=tnsr.dtype).view(
                        tnsr.shape
                    )
                )
                index += size
            except Exception as e:
                raise RuntimeError(
                    "`set_tensors_from_ndarray_1d` failed while copying values to "
                    f"tensor {name}; rethrowing original exception."
                ) from e


def get_bounds_as_ndarray(
    parameters: dict[str, Tensor],
    bounds: dict[str, tuple[float | Tensor | None, float | Tensor | None]],
) -> npt.NDArray | None:
    r"""Helper method for converting bounds into an ndarray.

    Args:
        parameters: A dictionary of parameters.
        bounds: A dictionary of (optional) lower and upper bounds.

    Returns:
        An ndarray of bounds.
    """
    inf = float("inf")
    full_size = sum(param.numel() for param in parameters.values())
    out = np.full((full_size, 2), (-inf, inf))
    index = 0
    for name, param in parameters.items():
        size = param.numel()
        if name in bounds:
            lower, upper = bounds[name]
            lower = -inf if lower is None else lower
            upper = inf if upper is None else upper
            if isinstance(lower, Tensor):
                lower = lower.cpu()
            if isinstance(upper, Tensor):
                upper = upper.cpu()
            out[index : index + size, 0] = lower
            out[index : index + size, 1] = upper
        index = index + size
    # If all bounds are +/- inf, return None.
    if np.isinf(out).all():
        out = None
    return out
