#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math

import torch
from botorch.utils.constants import get_constants_like
from torch import finfo, Tensor


# Unary ops
def exp(x: Tensor, **kwargs) -> Tensor:
    info = finfo(x.dtype)
    maxexp = get_constants_like(math.log(info.max) - 1e-4, x)
    return torch.exp(x.clip(max=maxexp), **kwargs)


def log(x: Tensor, **kwargs) -> Tensor:
    info = finfo(x.dtype)
    return torch.log(x.clip(min=info.tiny), **kwargs)


# Binary ops
def add(a: Tensor, b: Tensor, **kwargs) -> Tensor:
    _0 = get_constants_like(0, a)
    case = a.isinf() & b.isinf() & (a != b)
    return torch.where(case, _0, a + b)


def sub(a: Tensor, b: Tensor) -> Tensor:
    _0 = get_constants_like(0, a)
    case = (a.isinf() & b.isinf()) & (a == b)
    return torch.where(case, _0, a - b)


def div(a: Tensor, b: Tensor) -> Tensor:
    _0, _1 = get_constants_like(values=(0, 1), ref=a)
    case = ((a == _0) & (b == _0)) | (a.isinf() & a.isinf())
    return torch.where(case, torch.where(a != b, -_1, _1), a / torch.where(case, _1, b))


def mul(a: Tensor, b: Tensor) -> Tensor:
    _0 = get_constants_like(values=0, ref=a)
    case = (a.isinf() & (b == _0)) | (b.isinf() & (a == _0))
    return torch.where(case, _0, a * torch.where(case, _0, b))
