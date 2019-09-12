#! /usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import math

import torch
from botorch.test_functions.aug_branin import GLOBAL_MAXIMUM, neg_aug_branin
from botorch.utils.testing import BotorchTestCase


class TestNegAugBranin(BotorchTestCase):
    def test_single_eval_neg_aug_branin(self):
        for dtype in (torch.float, torch.double):
            X = torch.zeros(3, device=self.device, dtype=dtype)
            res = neg_aug_branin(X)
            self.assertEqual(res.dtype, dtype)
            self.assertEqual(res.device.type, self.device.type)
            self.assertEqual(res.shape, torch.Size())

    def test_batch_eval_neg_aug_branin(self):
        for dtype in (torch.float, torch.double):
            X = torch.zeros(2, 3, device=self.device, dtype=dtype)
            res = neg_aug_branin(X)
            self.assertEqual(res.dtype, dtype)
            self.assertEqual(res.device.type, self.device.type)
            self.assertEqual(res.shape, torch.Size([2]))

    def test_neg_aug_branin_global_maxima(self):
        for dtype in (torch.float, torch.double):
            X = torch.tensor(
                [
                    [-math.pi, 12.275, 1],
                    [math.pi, 1.3867356039019576, 0.1],
                    [math.pi, 1.781519779945532, 0.5],
                    [math.pi, 2.1763039559891064, 0.9],
                ],
                device=self.device,
                dtype=dtype,
                requires_grad=True,
            )
            res = neg_aug_branin(X)
            for r in res:
                self.assertAlmostEqual(r.item(), GLOBAL_MAXIMUM, places=4)
            grad = torch.autograd.grad(res.sum(), X)[0]
            self.assertLess(grad.abs().max().item(), 1e-4)
