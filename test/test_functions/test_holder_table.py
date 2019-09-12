#! /usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
from botorch.test_functions.holder_table import (
    GLOBAL_MAXIMIZERS,
    GLOBAL_MAXIMUM,
    neg_holder_table,
)
from botorch.utils.testing import BotorchTestCase


class TestNegHolderTable(BotorchTestCase):
    def test_single_eval_neg_holder_table(self):
        for dtype in (torch.float, torch.double):
            X = torch.zeros(2, device=self.device, dtype=dtype)
            res = neg_holder_table(X)
            self.assertEqual(res.dtype, dtype)
            self.assertEqual(res.device.type, self.device.type)
            self.assertEqual(res.shape, torch.Size())
            self.assertTrue(res.abs().item() < 1e-6)

    def test_batch_eval_neg_holder_table(self):
        for dtype in (torch.float, torch.double):
            X = torch.zeros(2, 2, device=self.device, dtype=dtype)
            res = neg_holder_table(X)
            self.assertEqual(res.dtype, dtype)
            self.assertEqual(res.device.type, self.device.type)
            self.assertEqual(res.shape, torch.Size([2]))
            self.assertTrue(res.abs().sum().item() < 1e-6)

    def test_neg_holder_table_global_maxima(self):
        for dtype in (torch.float, torch.double):
            X = torch.tensor(
                GLOBAL_MAXIMIZERS, device=self.device, dtype=dtype, requires_grad=True
            )
            res = neg_holder_table(X)
            grad = torch.autograd.grad([*res], X)[0]
            self.assertLess((res - GLOBAL_MAXIMUM).abs().max().item(), 1e-5)
            self.assertLess(grad.abs().max().item(), 1e-3)
