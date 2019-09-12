#! /usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
from botorch.test_functions.branin import GLOBAL_MAXIMIZERS, GLOBAL_MAXIMUM, neg_branin
from botorch.utils.testing import BotorchTestCase


class TestNegBranin(BotorchTestCase):
    def test_single_eval_neg_branin(self):
        for dtype in (torch.float, torch.double):
            X = torch.zeros(2, device=self.device, dtype=dtype)
            res = neg_branin(X)
            self.assertEqual(res.dtype, dtype)
            self.assertEqual(res.device.type, self.device.type)
            self.assertEqual(res.shape, torch.Size())

    def test_batch_eval_neg_branin(self):
        for dtype in (torch.float, torch.double):
            X = torch.zeros(2, 2, device=self.device, dtype=dtype)
            res = neg_branin(X)
            self.assertEqual(res.dtype, dtype)
            self.assertEqual(res.device.type, self.device.type)
            self.assertEqual(res.shape, torch.Size([2]))

    def test_neg_branin_global_maxima(self):
        for dtype in (torch.float, torch.double):
            X = torch.tensor(
                GLOBAL_MAXIMIZERS, device=self.device, dtype=dtype, requires_grad=True
            )
            res = neg_branin(X)
            for r in res:
                self.assertAlmostEqual(r.item(), GLOBAL_MAXIMUM, places=4)
            grad = torch.autograd.grad(res.sum(), X)[0]
            self.assertLess(grad.abs().max().item(), 1e-4)
