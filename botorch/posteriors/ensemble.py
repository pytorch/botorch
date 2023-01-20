#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Ensemble posteriors. Used in conjunction with ensemble
models.
"""

from __future__ import annotations

from typing import Optional

import torch
from botorch.posteriors.posterior import Posterior
from torch import Tensor


class EnsemblePosterior(Posterior):
    r"""Ensemble posterior."""

    def __init__(self, values: Tensor) -> None:
        r"""
        Args:
            values: Values of the samples produced by this posterior as
                a (b) x q x m x s tensor where m is the output size of the
                model and s is the ensemble size.
        """
        if values.ndim < 3:
            raise ValueError("Values has to be at least three-dimensional.")
        self.values = values

    @property
    def size(self) -> int:
        r"""The size of the ensemble"""
        return self.values.shape[-1]

    @property
    def device(self) -> torch.device:
        r"""The torch device of the posterior."""
        return self.values.device

    @property
    def dtype(self) -> torch.dtype:
        r"""The torch dtype of the posterior."""
        return self.values.dtype

    @property
    def mean(self) -> Tensor:
        r"""The mean of the posterior as a `(b) x n x m`-dim Tensor."""
        return self.values.mean(dim=-1)

    @property
    def variance(self) -> Tensor:
        r"""The variance of the posterior as a `(b) x n x m`-dim Tensor.

        As this is a deterministic posterior, this is a tensor of zeros.
        """
        return self.values.var(dim=-1)

    def _extended_shape(
        self, sample_shape: torch.Size = torch.Size()  # noqa: B008
    ) -> torch.Size:
        r"""Returns the shape of the samples produced by the posterior with
        the given `sample_shape`.
        """
        return sample_shape + self.values.shape

    def rsample(
        self,
        sample_shape: Optional[torch.Size] = None,
    ) -> Tensor:
        r"""Sample from the posterior (with gradients).

        For the deterministic posterior, this just returns the values expanded
        to the requested shape.

        Args:
            sample_shape: A `torch.Size` object specifying the sample shape. To
                draw `n` samples, set to `torch.Size([n])`. To draw `b` batches
                of `n` samples each, set to `torch.Size([b, n])`.

        Returns:
            Samples from the posterior, a tensor of shape
            `self._extended_shape(sample_shape=sample_shape)`.
        """
        if sample_shape is None:
            sample_shape = torch.Size([1])
        # sample_indices = torch.arange(0, sample_shape[0]).to(dtype=torch.int64)
        # return self.values[..., sample_indices]
        return self.values.expand(self._extended_shape(sample_shape))
