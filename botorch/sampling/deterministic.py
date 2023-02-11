#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
A dummy sampler for use with deterministic models.
"""

from __future__ import annotations

from botorch.posteriors.deterministic import DeterministicPosterior
from botorch.sampling.stochastic_samplers import StochasticSampler


class DeterministicSampler(StochasticSampler):
    r"""A sampler that simply calls `posterior.rsample`, intended to be used with
    `DeterministicModel` & `DeterministicPosterior`.

    [DEPRECATED] - Use `IndexSampler` in conjunction with `EnsemblePosterior`
    instead of `DeterministicSampler` with `DeterministicPosterior`.

    This is effectively signals that `StochasticSampler` is safe to use with
    deterministic models since their output is deterministic by definition.
    """

    def _update_base_samples(
        self, posterior: DeterministicPosterior, base_sampler: DeterministicSampler
    ) -> None:
        r"""This is a no-op since there are no base samples to update.

        Args:
            posterior: The posterior for which the base samples are constructed.
            base_sampler: The base sampler to retrieve the base samples from.
        """
        return
