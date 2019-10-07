#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
from botorch.posteriors.deterministic import DeterministicPosterior
from botorch.utils.testing import BotorchTestCase


class TestDeterministicPosterior(BotorchTestCase):
    def test_DeterministicPosterior(self):
        for dtype in (torch.float, torch.double):
            for shape in ((3, 2), (2, 3, 1)):
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
