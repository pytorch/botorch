#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Sampler to be used with `EnsemblePosteriors` to enable
deterministic optimization of acquisition functions with ensemble models.
"""

import torch
from botorch.posteriors import Posterior
from botorch.sampling.base import MCSampler
from botorch.utils.sampling import manual_seed
from torch import Tensor


class IndexSampler(MCSampler):
    r"""A sampler that calls `posterior.rsample_from_base_samples` to
    generate the samples via index base samples."""

    def forward(self, posterior: Posterior) -> Tensor:
        r"""Draws MC samples from the posterior.

        Args:
            posterior: The ensemble posterior to sample from.

        Returns:
            The samples drawn from the posterior.
        """
        self._construct_base_samples(posterior=posterior)
        samples = posterior.rsample_from_base_samples(
            sample_shape=self.sample_shape, base_samples=self.base_samples
        )
        return samples

    def _construct_base_samples(self, posterior: Posterior) -> None:
        r"""Constructs base samples as indices to sample with them from
        the Posterior

        Args:
            posterior: The ensemble posterior to construct the base samples
                for.
        """
        if self.base_samples is None or self.base_samples.shape != self.sample_shape:
            with manual_seed(seed=self.seed):
                base_samples = torch.multinomial(
                    posterior.weights,
                    num_samples=self.sample_shape.numel(),
                    replacement=True,
                ).reshape(self.sample_shape)
            self.register_buffer("base_samples", base_samples)
        if self.base_samples.device != posterior.device:
            self.to(device=posterior.device)  # pragma: nocover

    def _update_base_samples(
        self, posterior: Posterior, base_sampler: MCSampler
    ) -> None:
        r"""Null operation just needed for compatibility with
        `CachedCholeskyAcquisitionFunction`."""
        pass
