#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from botorch.exceptions.errors import InputDataError
from botorch.sampling.base import MCSampler
from botorch.utils.testing import BotorchTestCase, MockPosterior


class NonAbstractSampler(MCSampler):
    def forward(self, posterior):
        raise NotImplementedError


class OtherSampler(MCSampler):
    def forward(self, posterior):
        raise NotImplementedError


class TestBaseMCSampler(BotorchTestCase):
    def test_MCSampler_abstract_raises(self):
        with self.assertRaises(TypeError):
            MCSampler()

    def test_init(self):
        with self.assertRaises(TypeError):
            NonAbstractSampler()
        # Current args.
        sampler = NonAbstractSampler(sample_shape=torch.Size([4]), seed=1234)
        self.assertEqual(sampler.sample_shape, torch.Size([4]))
        self.assertEqual(sampler.seed, 1234)
        self.assertIsNone(sampler.base_samples)
        # Default seed.
        sampler = NonAbstractSampler(sample_shape=torch.Size([4]))
        self.assertIsInstance(sampler.seed, int)
        # Error handling.
        with self.assertRaisesRegex(InputDataError, "sample_shape"):
            NonAbstractSampler(4.5)

    def test_batch_range(self):
        posterior = MockPosterior()
        sampler = NonAbstractSampler(sample_shape=torch.Size([4]))
        # Default: read from the posterior.
        self.assertEqual(
            sampler._get_batch_range(posterior=posterior), posterior.batch_range
        )
        # Overwrite.
        sampler.batch_range_override = (0, -5)
        self.assertEqual(sampler._get_batch_range(posterior=posterior), (0, -5))

    def test_get_collapsed_shape(self):
        posterior = MockPosterior(base_shape=torch.Size([4, 3, 2]))
        sampler = NonAbstractSampler(sample_shape=torch.Size([4]))
        self.assertEqual(
            sampler._get_collapsed_shape(posterior=posterior), torch.Size([4, 1, 3, 2])
        )
        posterior = MockPosterior(
            base_shape=torch.Size([3, 4, 3, 2]), batch_range=(0, 0)
        )
        self.assertEqual(
            sampler._get_collapsed_shape(posterior=posterior),
            torch.Size([4, 3, 4, 3, 2]),
        )
        posterior = MockPosterior(
            base_shape=torch.Size([3, 4, 3, 2]), batch_range=(0, -1)
        )
        self.assertEqual(
            sampler._get_collapsed_shape(posterior=posterior),
            torch.Size([4, 1, 1, 1, 2]),
        )

    def test_get_extended_base_sample_shape(self):
        sampler = NonAbstractSampler(sample_shape=torch.Size([4]))
        posterior = MockPosterior(base_shape=torch.Size([3, 2]))
        self.assertEqual(
            sampler._get_extended_base_sample_shape(posterior=posterior),
            torch.Size([4, 3, 2]),
        )
        posterior = MockPosterior(base_shape=torch.Size([3, 5, 3, 2]))
        bss = sampler._get_extended_base_sample_shape(posterior=posterior)
        self.assertEqual(bss, torch.Size([4, 3, 5, 3, 2]))

    def test_update_base_samples(self):
        sampler = NonAbstractSampler(sample_shape=torch.Size([4]))
        with self.assertRaisesRegex(NotImplementedError, "update_base"):
            sampler._update_base_samples(
                posterior=MockPosterior(), base_sampler=sampler
            )

    def test_instance_check(self):
        sampler = NonAbstractSampler(sample_shape=torch.Size([4]))
        # Same type:
        sampler._instance_check(sampler)
        # Different type:
        other = OtherSampler(sample_shape=torch.Size([4]))
        with self.assertRaisesRegex(RuntimeError, "an instance of"):
            sampler._instance_check(base_sampler=other)
