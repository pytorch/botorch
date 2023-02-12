#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools

from warnings import catch_warnings

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
            with catch_warnings(record=True) as ws:
                p = DeterministicPosterior(values)
                self.assertTrue(
                    any("marked for deprecation" in str(w.message) for w in ws)
                )
            self.assertEqual(p.device.type, self.device.type)
            self.assertEqual(p.dtype, dtype)
            self.assertEqual(p._extended_shape(), values.shape)
            with self.assertRaises(NotImplementedError):
                p.base_sample_shape
            self.assertTrue(torch.equal(p.mean, values))
            self.assertTrue(torch.equal(p.variance, torch.zeros_like(values)))
            # test sampling
            samples = p.rsample()
            self.assertTrue(torch.equal(samples, values.unsqueeze(0)))
            samples = p.rsample(torch.Size([2]))
            self.assertTrue(torch.equal(samples, values.expand(2, *values.shape)))
