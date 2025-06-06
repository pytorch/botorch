#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Ensemble posteriors. Used in conjunction with ensemble models.
"""

from __future__ import annotations

import torch
from botorch.posteriors.posterior import Posterior
from torch import Tensor


class EnsemblePosterior(Posterior):
    r"""Ensemble posterior, that should be used for ensemble models that compute
    eagerly a finite number of samples per X value as for example a deep ensemble
    or a random forest."""

    def __init__(self, values: Tensor) -> None:
        r"""
        Args:
            values: Values of the samples produced by this posterior as
                a `(b) x s x q x m` tensor where `m` is the output size of the
                model and `s` is the ensemble size.
        """
        if values.ndim < 3:
            raise ValueError("Values has to be at least three-dimensional.")
        self.values = values

    @property
    def ensemble_size(self) -> int:
        r"""The size of the ensemble"""
        return self.values.shape[-3]

    @property
    def weights(self) -> Tensor:
        r"""The weights of the individual models in the ensemble.
        Equally weighted by default."""
        return torch.ones(self.ensemble_size) / self.ensemble_size

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
        return self.values.mean(dim=-3)

    @property
    def variance(self) -> Tensor:
        r"""The variance of the posterior as a `(b) x n x m`-dim Tensor.

        Computed as the sample variance across the ensemble outputs.
        """
        if self.ensemble_size == 1:
            return torch.zeros_like(self.values.squeeze(-3))
        return self.values.var(dim=-3)

    def _extended_shape(
        self,
        sample_shape: torch.Size = torch.Size(),  # noqa: B008
    ) -> torch.Size:
        r"""Returns the shape of the samples produced by the posterior with
        the given `sample_shape`.
        """
        return sample_shape + self.values.shape[:-3] + self.values.shape[-2:]

    def rsample(
        self,
        sample_shape: torch.Size | None = None,
    ) -> Tensor:
        r"""Sample from the posterior (with gradients).

        Based on the sample shape, base samples are generated and passed to
        `rsample_from_base_samples`.

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
        base_samples = (
            torch.multinomial(
                self.weights,
                num_samples=sample_shape.numel(),
                replacement=True,
            )
            .reshape(sample_shape)
            .to(device=self.device)
        )
        return self.rsample_from_base_samples(
            sample_shape=sample_shape, base_samples=base_samples
        )

    def rsample_from_base_samples(
        self, sample_shape: torch.Size, base_samples: Tensor
    ) -> Tensor:
        r"""Sample from the posterior (with gradients) using base samples.

        This is intended to be used with a sampler that produces the corresponding base
        samples, and enables acquisition optimization via Sample Average Approximation.

        Args:
            sample_shape: A `torch.Size` object specifying the sample shape. To
                draw `n` samples, set to `torch.Size([n])`. To draw `b` batches
                of `n` samples each, set to `torch.Size([b, n])`.
            base_samples: A Tensor of indices as base samples of shape
                `sample_shape`, typically obtained from `IndexSampler`.
                This is used for deterministic optimization. The predictions of
                the ensemble corresponding to the indices are then sampled.


        Returns:
            Samples from the posterior, a tensor of shape
            `self._extended_shape(sample_shape=sample_shape)`.
        """
        if base_samples.shape != sample_shape:
            raise ValueError("Base samples do not match sample shape.")
        # move sample axis to front
        values = self.values.movedim(-3, 0)
        # sample from the first dimension of values
        return values[base_samples, ...]
