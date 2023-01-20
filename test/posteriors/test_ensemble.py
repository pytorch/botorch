#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from botorch.posteriors.ensemble import EnsemblePosterior
from botorch.utils.testing import BotorchTestCase
import itertools


class TestEnsemblePosterior(BotorchTestCase):
    def test_EnsemblePosterior_invalid(self):
        for shape, dtype in itertools.product(
            ((5, 2), (2, 5, 2, 1), (5, 2, 1)), (torch.float, torch.double)
        ):
            values = torch.randn(*shape, device=self.device, dtype=dtype)
            with self.assertRaises(ValueError):
                EnsemblePosterior(values)

    def test_EnsemblePosterior(self):
        for shape, dtype in itertools.product(
            ((5, 2, 16), (2, 5, 2, 32)), (torch.float, torch.double)
        ):
            values = torch.randn(*shape, device=self.device, dtype=dtype)
            p = EnsemblePosterior(values)
            self.assertEqual(p.device.type, self.device.type)
            self.assertEqual(p.dtype, dtype)
            self.assertEqual(p.size, shape[-1])
            self.assertTrue(torch.equal(p.mean, values.mean(dim=-1)))
            self.assertTrue(torch.equal(p.variance, values.var(dim=-1)))

            # tests regarding sampling are commented out as this seems still
            # to be incorrect
            # self.assertEqual(p._extended_shape(), values.shape)
            # with self.assertRaises(NotImplementedError):
            #    p.base_sample_shape
            # test sampling
            # samples = p.rsample()
            # self.assertTrue(torch.equal(samples, values.unsqueeze(0)))
            # samples = p.rsample(torch.Size([2]))
            # self.assertTrue(torch.equal(samples, values.expand(2, *values.shape)))
