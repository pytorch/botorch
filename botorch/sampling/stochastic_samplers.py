#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Samplers to enable use cases that are not base sample driven, such as
stochastic optimization of acquisition functions.
"""

from __future__ import annotations

import torch
from botorch.posteriors import Posterior
from botorch.posteriors.deterministic import DeterministicPosterior
from botorch.sampling.base import MCSampler
from torch import Tensor


class ForkedRNGSampler(MCSampler):
    r"""A sampler using `torch.fork_rng` to enable replicable sampling
    from a posterior that does not support base samples.

    NOTE: This approach is not a one-to-one replacement for base sample
    driven sampling. The main missing piece in this approach is that its
    outputs are not replicable across the batch dimensions. As a result,
    when an acquisition function is batch evaluated with repeated candidates,
    each candidate will produce a different acquisition value, which is not
    compatible with Sample Average Approximation.
    """

    def forward(self, posterior: Posterior) -> Tensor:
        r"""Draws MC samples from the posterior in a `fork_rng` context.

        Args:
            posterior: The posterior to sample from.

        Returns:
            The samples drawn from the posterior.
        """
        with torch.random.fork_rng():
            torch.manual_seed(self.seed)
            return posterior.rsample(sample_shape=self.sample_shape)


class StochasticSampler(MCSampler):
    r"""A sampler that simply calls `posterior.rsample` to generate the
    samples. This should only be used for stochastic optimization of the
    acquisition functions, e.g., via `gen_candidates_torch`. This should
    not be used with `optimize_acqf`, which uses deterministic optimizers
    under the hood.

    NOTE: This ignores the `seed` option.
    """

    def forward(self, posterior: Posterior) -> Tensor:
        r"""Draws MC samples from the posterior.

        Args:
            posterior: The posterior to sample from.

        Returns:
            The samples drawn from the posterior.
        """
        return posterior.rsample(sample_shape=self.sample_shape)

    def _update_base_samples(
        self, posterior: Posterior, base_sampler: StochasticSampler
    ) -> None:
        r"""Update the sampler to use the original base samples for X_baseline.

        This is used in CachedCholeskyAcquisitionFunctions to ensure consistency.
        This is a no-op for DeterministicPosterior and errors out otherwise.

        Args:
            posterior: The posterior for which the base samples are constructed.
            base_sampler: The base sampler to retrieve the base samples from.
        """
        if not isinstance(posterior, DeterministicPosterior):
            super()._update_base_samples(posterior=posterior, base_sampler=base_sampler)
