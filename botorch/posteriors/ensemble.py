#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Ensemble posteriors. Used in conjunction with ensemble models.
"""

from __future__ import annotations

from typing import Optional

import torch
from botorch.posteriors.posterior import Posterior
from torch import Tensor

r"""
Posterior module to be used with ensemble models.
"""


class EnsemblePosterior(Posterior):
    r"""Ensemble posterior."""

    def __init__(self, values: Tensor) -> None:
        r"""
        Args:
            values: Values of the samples produced by this posterior as
                a `(b) x q x m x s` tensor where `m` is the output size of the
                model and `s` is the ensemble size.
        """
        if values.ndim < 3:
            raise ValueError("Values has to be at least three-dimensional.")
        if values.shape[-1] < 2:
            raise ValueError("Ensemble size has to be at least two.")
        self.values = values

    @property
    def size(self) -> int:
        r"""The size of the ensemble"""
        return self.values.shape[-1]

    @property
    def weights(self) -> Tensor:
        r"""The weights of the individual models in the ensemble.
        Equally weighted by default."""
        return torch.ones(self.size) / self.size

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

        Computed as the sample variance across the ensemble outputs.
        """
        return self.values.var(dim=-1)

    def _extended_shape(
        self, sample_shape: torch.Size = torch.Size()  # noqa: B008
    ) -> torch.Size:
        r"""Returns the shape of the samples produced by the posterior with
        the given `sample_shape`.
        """
        return sample_shape + self.values.shape[:-1]

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
        # get indices as base_samples
        base_samples = torch.multinomial(
            self.weights,
            num_samples=sample_shape.numel(),
            replacement=True,
        ).reshape(sample_shape)
        return self.rsample_from_base_samples(
            sample_shape=sample_shape, base_samples=base_samples
        )

    def rsample_from_base_samples(
        self, sample_shape: torch.Size, base_samples: Tensor
    ) -> Tensor:
        r"""_summary_

        Args:
            sample_shape (torch.Size): _description_
            base_samples (Tensor): _description_

        Raises:
            ValueError: _description_

        Returns:
            Tensor: _description_
        """
        if base_samples.shape != sample_shape:
            raise ValueError("Base samples to not match sample shape.")
        # move sample axis to front
        values = self.values.movedim(-1, 0)
        # sample from the first dimension of values
        return values[base_samples, ...]
