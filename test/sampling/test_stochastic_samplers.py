#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest import mock

import torch
from botorch.posteriors.torch import TorchPosterior
from botorch.sampling.stochastic_samplers import ForkedRNGSampler, StochasticSampler
from botorch.utils.testing import BotorchTestCase, MockPosterior
from torch.distributions.exponential import Exponential


class TestForkedRNGSampler(BotorchTestCase):
    def test_forked_rng_sampler(self):
        posterior = TorchPosterior(Exponential(rate=torch.rand(1, 2)))
        sampler = ForkedRNGSampler(sample_shape=torch.Size([2]), seed=0)
        with mock.patch.object(
            posterior.distribution, "rsample", wraps=posterior.distribution.rsample
        ) as mock_rsample:
            samples = sampler(posterior)
        mock_rsample.assert_called_once_with(sample_shape=torch.Size([2]))
        with torch.random.fork_rng():
            torch.manual_seed(0)
            expected = posterior.rsample(sample_shape=torch.Size([2]))
        self.assertAllClose(samples, expected)


class TestStochasticSampler(BotorchTestCase):
    def test_stochastic_sampler(self):
        # Basic usage.
        samples = torch.rand(1, 2)
        posterior = MockPosterior(samples=samples)
        sampler = StochasticSampler(sample_shape=torch.Size([2]))
        self.assertTrue(torch.equal(samples.repeat(2, 1, 1), sampler(posterior)))

        # Test _update_base_samples.
        with self.assertRaisesRegex(NotImplementedError, "_update_base_samples"):
            sampler._update_base_samples(posterior=posterior, base_sampler=sampler)
