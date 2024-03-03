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
            with self.assertRaisesRegex(
                ValueError,
                "Values has to be at least three-dimensional",
            ):
                EnsemblePosterior(values)

    def test_EnsemblePosterior_as_Deterministic(self):
        for shape, dtype in itertools.product(
            ((1, 3, 2), (2, 1, 3, 2)), (torch.float, torch.double)
        ):
            values = torch.randn(*shape, device=self.device, dtype=dtype)
            p = EnsemblePosterior(values)
            self.assertEqual(p.ensemble_size, 1)
            self.assertEqual(p.device.type, self.device.type)
            self.assertEqual(p.dtype, dtype)
            self.assertEqual(
                p._extended_shape(torch.Size((1,))),
                torch.Size((1, 3, 2)) if len(shape) == 3 else torch.Size((1, 2, 3, 2)),
            )
            self.assertEqual(p.weights, torch.ones(1))
            with self.assertRaises(NotImplementedError):
                p.base_sample_shape
            self.assertTrue(torch.equal(p.mean, values.squeeze(-3)))
            self.assertTrue(
                torch.equal(p.variance, torch.zeros_like(values.squeeze(-3)))
            )
            # test sampling
            samples = p.rsample()
            self.assertTrue(torch.equal(samples, values.squeeze(-3).unsqueeze(0)))
            samples = p.rsample(torch.Size([2]))
            self.assertEqual(samples.shape, p._extended_shape(torch.Size([2])))

    def test_EnsemblePosterior(self):
        for shape, dtype in itertools.product(
            ((16, 5, 2), (2, 16, 5, 2)), (torch.float, torch.double)
        ):
            values = torch.randn(*shape, device=self.device, dtype=dtype)
            p = EnsemblePosterior(values)
            self.assertEqual(p.device.type, self.device.type)
            self.assertEqual(p.dtype, dtype)
            self.assertEqual(p.ensemble_size, 16)
            self.assertAllClose(
                p.weights, torch.tensor([1.0 / p.ensemble_size] * p.ensemble_size)
            )
            # test mean and variance
            self.assertTrue(torch.equal(p.mean, values.mean(dim=-3)))
            self.assertTrue(torch.equal(p.variance, values.var(dim=-3)))
            # test extended shape
            self.assertEqual(
                p._extended_shape(torch.Size((128,))),
                (
                    torch.Size((128, 5, 2))
                    if len(shape) == 3
                    else torch.Size((128, 2, 5, 2))
                ),
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
