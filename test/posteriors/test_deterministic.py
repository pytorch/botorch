#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools

import torch
from botorch.posteriors.deterministic import DeterministicPosterior
from botorch.utils.testing import BotorchTestCase


class TestDeterministicPosterior(BotorchTestCase):
    def test_DeterministicPosterior(self):
        for shape, dtype in itertools.product(
            ((3, 2), (2, 3, 1)), (torch.float, torch.double)
        ):
            values = torch.randn(*shape, device=self.device, dtype=dtype)
            p = DeterministicPosterior(values)
            self.assertEqual(p.device, self.device)
            self.assertEqual(p.dtype, dtype)
            self.assertEqual(p.event_shape, values.shape)
            self.assertTrue(torch.equal(p.mean, values))
            self.assertTrue(torch.equal(p.variance, torch.zeros_like(values)))
            # test sampling
            samples = p.rsample()
            self.assertTrue(torch.equal(samples, values.unsqueeze(0)))
            samples = p.rsample(torch.Size([2]))
            self.assertTrue(torch.equal(samples, values.expand(2, *values.shape)))
            base_samples = torch.randn(2, *shape, device=self.device, dtype=dtype)
            samples = p.rsample(torch.Size([2]), base_samples)
            self.assertTrue(torch.equal(samples, values.expand(2, *values.shape)))
            with self.assertRaises(RuntimeError):
                samples = p.rsample(
                    torch.Size([2]), base_samples.expand(3, *base_samples.shape)
                )
