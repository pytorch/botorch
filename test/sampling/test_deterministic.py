#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from botorch.sampling.deterministic import DeterministicSampler
from botorch.utils.testing import BotorchTestCase, MockPosterior


class TestDeterministicSampler(BotorchTestCase):
    def test_deterministic_sampler(self):
        # Basic usage.
        samples = torch.rand(1, 2)
        posterior = MockPosterior(samples=samples)
        sampler = DeterministicSampler(sample_shape=torch.Size([2]))
        self.assertTrue(torch.equal(samples.repeat(2, 1, 1), sampler(posterior)))

        # Test _update_base_samples.
        sampler._update_base_samples(
            posterior=posterior,
            base_sampler=sampler,
        )
