#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools

import torch
from botorch.posteriors.ensemble import EnsemblePosterior
from botorch.utils.testing import BotorchTestCase


class TestEnsemblePosterior(BotorchTestCase):
    def test_EnsemblePosterior_invalid(self):
        for shape, dtype in itertools.product(
            ((5, 2), (5, 1)), (torch.float, torch.double)
        ):
            values = torch.randn(*shape, device=self.device, dtype=dtype)
            with self.assertRaises(ValueError):
                EnsemblePosterior(values)

    def testEnsemblePosteriorAsDeterministic(self):
        for shape, dtype in itertools.product(
            ((3, 2, 1), (2, 3, 1, 1)), (torch.float, torch.double)
        ):
            values = torch.randn(*shape, device=self.device, dtype=dtype)
            p = EnsemblePosterior(values)
            self.assertEqual(p.size, 1)
            self.assertEqual(p.device.type, self.device.type)
            self.assertEqual(p.dtype, dtype)
            # self.assertEqual(p._extended_shape(), values.shape)
            with self.assertRaises(NotImplementedError):
                p.base_sample_shape
            self.assertTrue(torch.equal(p.mean, values[..., -1]))
            self.assertTrue(torch.equal(p.variance, torch.zeros_like(values[..., -1])))
            # test sampling
            samples = p.rsample()
            self.assertTrue(torch.equal(samples, values[..., -1].unsqueeze(0)))
            samples = p.rsample(torch.Size([2]))
            self.assertEqual(samples.shape, p._extended_shape(torch.Size((2,))))

    def test_EnsemblePosterior(self):
        for shape, dtype in itertools.product(
            ((5, 2, 16), (2, 5, 2, 16)), (torch.float, torch.double)
        ):
            values = torch.randn(*shape, device=self.device, dtype=dtype)
            p = EnsemblePosterior(values)
            self.assertEqual(p.device.type, self.device.type)
            self.assertEqual(p.dtype, dtype)
            self.assertEqual(p.size, shape[-1])
            self.assertAllClose(p.weights, torch.tensor([1.0 / p.size] * p.size))
            # test mean and variance
            self.assertTrue(torch.equal(p.mean, values.mean(dim=-1)))
            self.assertTrue(torch.equal(p.variance, values.var(dim=-1)))
            # test extended shape
            self.assertEqual(
                p._extended_shape(torch.Size((128,))),
                torch.Size((128,)) + p.values.shape[:-1],
            )
            # test rsample
            samples = p.rsample(torch.Size((1024,)))
            self.assertEqual(samples.shape, p._extended_shape(torch.Size((1024,))))
            # test rsample from base samples
            # test that produced samples are correct
            samples = p.rsample_from_base_samples(
                sample_shape=torch.Size((16,)), base_samples=torch.arange(16)
            )
            self.assertEqual(samples.shape, p._extended_shape(torch.Size((16,))))
            self.assertAllClose(p.mean, samples.mean(dim=0))
            self.assertAllClose(p.variance, samples.var(dim=0))
            # test error on base_samples, sample_shape mismatch
            with self.assertRaises(ValueError):
                p.rsample_from_base_samples(
                    sample_shape=torch.Size((17,)), base_samples=torch.arange(16)
                )
