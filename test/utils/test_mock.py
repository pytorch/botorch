#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import torch
from botorch.utils.mock import MockModel, MockPosterior

from ..botorch_test_case import BotorchTestCase


class TestMock(BotorchTestCase):
    def test_MockPosterior(self):
        # test basic logic
        mp = MockPosterior()
        self.assertEqual(mp.device.type, "cpu")
        self.assertEqual(mp.dtype, torch.float32)
        self.assertEqual(mp.event_shape, torch.Size())
        self.assertEqual(
            MockPosterior(variance=torch.rand(2)).event_shape, torch.Size([2])
        )
        # test passing in tensors
        mean = torch.rand(2)
        variance = torch.eye(2)
        samples = torch.rand(1, 2)
        mp = MockPosterior(mean=mean, variance=variance, samples=samples)
        self.assertEqual(mp.device.type, "cpu")
        self.assertEqual(mp.dtype, torch.float32)
        self.assertTrue(torch.equal(mp.mean, mean))
        self.assertTrue(torch.equal(mp.variance, variance))
        self.assertTrue(torch.all(mp.sample() == samples.unsqueeze(0)))
        self.assertTrue(
            torch.all(mp.sample(torch.Size([2])) == samples.repeat(2, 1, 1))
        )
        with self.assertRaises(RuntimeError):
            mp.sample(sample_shape=torch.Size([2]), base_samples=torch.rand(3))

    def test_MockModel(self):
        mp = MockPosterior()
        mm = MockModel(mp)
        X = torch.empty(0)
        self.assertEqual(mm.posterior(X), mp)
        self.assertEqual(mm.num_outputs, 0)
        mm.state_dict()
        mm.load_state_dict()
