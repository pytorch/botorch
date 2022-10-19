#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from functools import lru_cache
from math import pi
from numbers import Number
from typing import Any, Callable, Iterable, Iterator, Optional, Tuple, Union

import torch
from numpy.polynomial.legendre import leggauss as numpy_leggauss
from torch import BoolTensor, LongTensor, Tensor

CaseNd = Tuple[Callable[[], BoolTensor], Callable[[BoolTensor], Tensor]]

_inv_sqrt_2pi = (2 * pi) ** -0.5
_neg_inv_sqrt2 = -(2**-0.5)
STANDARDIZED_RANGE: Tuple[float, float] = (-1e6, 1e6)


def case_dispatcher(
    out: Tensor,
    cases: Iterable[CaseNd] = (),
    default: Callable[[BoolTensor], Tensor] = None,
) -> Tensor:
    r"""Basic implementation of a tensorized switching case statement.

    Args:
        out: Tensor to which case outcomes are written.
        cases: Iterable of function pairs (pred, func), where `mask=pred()` specifies
            whether `func` is applicable for each entry in `out`. Note that cases are
            resolved first-come, first-serve.
        default: Optional `func` to which all unclaimed entries of `out` are dispatched.
    """
    active = None
    for closure, func in cases:
        pred = closure()
        if not pred.any():
            continue

        mask = pred if (active is None) else pred & active
        if not mask.any():
            continue

        if mask.all():  # where possible, use Ellipsis to avoid indexing
            out[...] = func(...)
            return out

        out[mask] = func(mask)
        if active is None:
            active = ~mask
        else:
            active[mask] = False

        if not active.any():
            break

    if default is not None:
        if active is None:
            out[...] = default(...)
        elif active.any():
            out[active] = default(active)

    return out


@lru_cache(maxsize=None)
def get_constants(
    values: Union[Number, Iterator[Number]],
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Union[Tensor, Tuple[Tensor, ...]]:
    r"""Returns scalar-valued Tensors containing each of the given constants.
    Used to expedite tensor operations involving scalar arithmetic. Note that
    the returned Tensors should not be modified in-place."""
    if isinstance(values, Number):
        return torch.full((), values, dtype=dtype, device=device)

    return tuple(torch.full((), val, dtype=dtype, device=device) for val in values)


def get_constants_like(
    values: Union[Number, Iterator[Number]],
    ref: Tensor,
) -> Union[Tensor, Iterator[Tensor]]:
    return get_constants(values, device=ref.device, dtype=ref.dtype)


def gen_positional_indices(
    shape: torch.Size,
    dim: int,
    device: Optional[torch.device] = None,
) -> Iterator[torch.LongTensor]:
    ndim = len(shape)
    _dim = ndim + dim if dim < 0 else dim
    if _dim >= ndim or _dim < 0:
        raise ValueError(f"dim={dim} invalid for shape {shape}.")

    cumsize = shape[_dim + 1 :].numel()
    for i, s in enumerate(reversed(shape[: _dim + 1])):
        yield torch.arange(0, s * cumsize, cumsize, device=device)[(...,) + i * (None,)]
        cumsize *= s


def build_positional_indices(
    shape: torch.Size,
    dim: int,
    device: Optional[torch.device] = None,
) -> LongTensor:
    return sum(gen_positional_indices(shape=shape, dim=dim, device=device))


@lru_cache(maxsize=None)
def leggauss(deg: int, **tkwargs: Any) -> Tuple[Tensor, Tensor]:
    x, w = numpy_leggauss(deg)
    return torch.as_tensor(x, **tkwargs), torch.as_tensor(w, **tkwargs)


def ndtr(x: Tensor) -> Tensor:
    r"""Standard normal CDF."""
    half, ninv_sqrt2 = get_constants_like((0.5, _neg_inv_sqrt2), x)
    return half * torch.erfc(ninv_sqrt2 * x)


def phi(x: Tensor) -> Tensor:
    r"""Standard normal PDF."""
    inv_sqrt_2pi, neg_half = get_constants_like((_inv_sqrt_2pi, -0.5), x)
    return inv_sqrt_2pi * (neg_half * x.square()).exp()


def swap_along_dim_(
    values: Tensor,
    i: Union[int, LongTensor],
    j: Union[int, LongTensor],
    dim: int,
    buffer: Optional[Tensor] = None,
) -> Tensor:
    r"""Swaps Tensor slices in-place along dimension `dim`.

    When passed as Tensors, `i` (and `j`) should be `dim`-dimensional tensors
    with the same shape as `values.shape[:dim]`. The xception to this rule occurs
    when `dim=0`, in which case `i` (and `j`) should be (at most) one-dimensional
    when passed as a Tensor.

    Args:
        values: Tensor whose values are to be swapped.
        i: Indices for slices along dimension `dim`.
        j: Indices for slices along dimension `dim`.
        dim: The dimension of `values` along which to swap slices.
        buffer: Optional buffer used internally to store copied values.

    Returns:
        The original `values` tensor.
    """
    dim = values.ndim + dim if dim < 0 else dim
    if dim and (
        (isinstance(i, Tensor) and i.ndim) or (isinstance(j, Tensor) and j.ndim)
    ):
        # Handle n-dimensional batches of heterogeneous swaps via linear indexing
        if isinstance(i, Tensor) and i.shape != values.shape[:dim]:
            raise ValueError("Batch shapes of `i` and `values` do not match.")

        if isinstance(j, Tensor) and j.shape != values.shape[:dim]:
            raise ValueError("Batch shapes of `j` and `values` do not match.")

        pidx = build_positional_indices(
            shape=values.shape[: dim + 1], dim=-2, device=values.device
        )

        swap_along_dim_(
            values.view(-1, *values.shape[dim + 1 :]),
            i=(pidx + i).view(-1),
            j=(pidx + j).view(-1),
            dim=0,
            buffer=buffer,
        )

    else:
        # Base cases: homogeneous swaps and 1-dimenensional heterogeneous swaps
        if isinstance(i, Tensor) and i.ndim > 1:
            raise ValueError("Tensor `i` must be at most 1-dimensional when `dim=0`.")

        if isinstance(j, Tensor) and j.ndim > 1:
            raise ValueError("Tensor `j` must be at most 1-dimensional when `dim=0`.")

        if dim:
            ctx = tuple(slice(None) for _ in range(dim))
            i = ctx + (i,)
            j = ctx + (j,)

        if buffer is None:
            buffer = values[i].clone()
        else:
            buffer.copy_(values[i])

        values[i] = values[j]
        values[j] = buffer

    return values
