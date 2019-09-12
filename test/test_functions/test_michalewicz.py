#! /usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
from botorch.test_functions.michalewicz import (
    GLOBAL_MAXIMIZER,
    GLOBAL_MAXIMUM,
    neg_michalewicz,
)
from botorch.utils.testing import BotorchTestCase


class TestNegMichalewicz(BotorchTestCase):
    def test_single_eval_neg_michalewicz(self):
        for dtype in (torch.float, torch.double):
            X = torch.zeros(10, device=self.device, dtype=dtype)
            res = neg_michalewicz(X)
            self.assertEqual(res.dtype, dtype)
            self.assertEqual(res.device.type, self.device.type)
            self.assertEqual(res.shape, torch.Size())

    def test_batch_eval_neg_michalewicz(self):
        for dtype in (torch.float, torch.double):
            X = torch.zeros(2, 10, device=self.device, dtype=dtype)
            res = neg_michalewicz(X)
            self.assertEqual(res.dtype, dtype)
            self.assertEqual(res.device.type, self.device.type)
            self.assertEqual(res.shape, torch.Size([2]))

    def test_neg_michalewicz_global_maximum(self):
        for dtype in (torch.float, torch.double):
            X = torch.tensor(
                GLOBAL_MAXIMIZER, device=self.device, dtype=dtype, requires_grad=True
            )
            res = neg_michalewicz(X)
            self.assertAlmostEqual(res.item(), GLOBAL_MAXIMUM, places=4)
            grad = torch.autograd.grad(res, X)[0]
            self.assertLess(grad.abs().max().item(), 1e-3)
