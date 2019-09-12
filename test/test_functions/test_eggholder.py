#! /usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
from botorch.test_functions.eggholder import (
    GLOBAL_MAXIMIZER,
    GLOBAL_MAXIMUM,
    neg_eggholder,
)
from botorch.utils.testing import BotorchTestCase


class TestNegEggholder(BotorchTestCase):
    def test_single_eval_neg_eggholder(self):
        for dtype in (torch.float, torch.double):
            X = torch.zeros(2, device=self.device, dtype=dtype)
            res = neg_eggholder(X)
            self.assertEqual(res.dtype, dtype)
            self.assertEqual(res.device.type, self.device.type)
            self.assertEqual(res.shape, torch.Size())

    def test_batch_eval_neg_eggholder(self):
        for dtype in (torch.float, torch.double):
            X = torch.zeros(2, 2, device=self.device, dtype=dtype)
            res = neg_eggholder(X)
            self.assertEqual(res.dtype, dtype)
            self.assertEqual(res.device.type, self.device.type)
            self.assertEqual(res.shape, torch.Size([2]))

    def test_neg_eggholder_global_maximum(self):
        for dtype in (torch.float, torch.double):
            X = torch.tensor(
                GLOBAL_MAXIMIZER, device=self.device, dtype=dtype, requires_grad=True
            )
            res = neg_eggholder(X)
            self.assertAlmostEqual(res.item(), GLOBAL_MAXIMUM, places=4)
            grad = torch.autograd.grad(res, X)[0]
            self.assertGreater(grad.abs().max().item(), 3.0)
