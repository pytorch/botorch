#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Smooth spectral matrix functions with analytic, reusable higher derivatives."""

from typing import Any

import torch


def _symmetric(value: Any) -> Any:
    return (value + value.mT) * 0.5


class _SymmetricSylvester(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, coefficient: Any, rhs: Any) -> Any:
        eigenvalues, vectors = torch.linalg.eigh(coefficient)
        if bool((eigenvalues <= 0).any()):
            raise ValueError("The Sylvester coefficient must be positive definite.")
        transformed = vectors.mT @ rhs @ vectors
        result = (
            vectors
            @ (transformed / (eigenvalues.unsqueeze(-1) + eigenvalues.unsqueeze(-2)))
            @ vectors.mT
        )
        ctx.save_for_backward(coefficient, result)
        return result

    @staticmethod
    def backward(ctx: Any, gradient: Any) -> tuple[Any, Any]:
        coefficient, result = ctx.saved_tensors
        # This same implicit rule is differentiable again. Saved outputs retain
        # their Function edge, so no eigenvector derivative is used at any order.
        dual = _SymmetricSylvester.apply(coefficient, _symmetric(gradient))
        return -(dual @ result + result @ dual), dual


class _SymmetricAbsolute(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, value: Any) -> Any:
        eigenvalues, vectors = torch.linalg.eigh(value)
        tolerance = (
            torch.finfo(value.dtype).eps
            * eigenvalues.abs().max().clamp_min(1)
            * value.shape[-1]
        )
        if bool((eigenvalues.abs() <= tolerance).any()):
            raise ValueError(
                "PSD-floor derivatives are undefined at the clipping threshold."
            )
        result = (vectors * eigenvalues.abs().unsqueeze(-2)) @ vectors.mT
        ctx.save_for_backward(value, result)
        return result

    @staticmethod
    def backward(ctx: Any, gradient: Any) -> tuple[Any]:
        value, absolute = ctx.saved_tensors
        # |A|^2=A^2: |A| d|A| + d|A| |A| = A dA + dA A.
        dual = _SymmetricSylvester.apply(absolute, _symmetric(gradient))
        return (value @ dual + dual @ value,)


def smooth_psd_floor(value: Any, floor: float) -> Any:
    """Clip a mixed spectrum without differentiating its eigenvectors.

    The matrix absolute value uses an O(n^3) Sylvester solve, including repeated
    eigenvalues on either side of the threshold. Only the clipping threshold
    itself is nonsmooth. Promote float32 internally so tiny floor eigenspaces
    remain resolved alongside ordinary covariance scales.
    """
    working = value.to(torch.double) if value.dtype == torch.float32 else value
    identity = torch.eye(value.shape[-1], device=value.device, dtype=working.dtype)
    shifted = working - floor * identity
    absolute = _SymmetricAbsolute.apply(shifted)
    return ((working + floor * identity + absolute) * 0.5).to(value.dtype)
