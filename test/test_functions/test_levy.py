#! /usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
from botorch.test_functions.levy import GLOBAL_MAXIMIZER, GLOBAL_MAXIMUM, neg_levy
from botorch.utils.testing import BotorchTestCase


DIMENSION = 4


class TestNegLevy(BotorchTestCase):
    def test_single_eval_neg_levy(self):
        for dtype in (torch.float, torch.double):
            X = torch.zeros(DIMENSION, device=self.device, dtype=dtype)
            res = neg_levy(X)
            self.assertEqual(res.dtype, dtype)
            self.assertEqual(res.device.type, self.device.type)
            self.assertEqual(res.shape, torch.Size())

    def test_batch_eval_neg_levy(self):
        for dtype in (torch.float, torch.double):
            X = torch.zeros(2, DIMENSION, device=self.device, dtype=dtype)
            res = neg_levy(X)
            self.assertEqual(res.dtype, dtype)
            self.assertEqual(res.device.type, self.device.type)
            self.assertEqual(res.shape, torch.Size([2]))

    def test_neg_levy_global_maximum(self):
        for dtype in (torch.float, torch.double):
            X = torch.tensor(
                GLOBAL_MAXIMIZER * DIMENSION,
                device=self.device,
                dtype=dtype,
                requires_grad=True,
            )
            res = neg_levy(X)
            self.assertAlmostEqual(res.item(), GLOBAL_MAXIMUM, places=4)
            grad = torch.autograd.grad(res, X)[0]
            self.assertLess(grad.abs().max().item(), 1e-4)
