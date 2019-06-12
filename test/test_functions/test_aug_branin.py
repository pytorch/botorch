#! /usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import math
import unittest

import torch
from botorch.test_functions.aug_branin import GLOBAL_MAXIMUM, neg_aug_branin


class TestNegAugBranin(unittest.TestCase):
    def test_single_eval_neg_aug_branin(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            X = torch.zeros(3, device=device, dtype=dtype)
            res = neg_aug_branin(X)
            self.assertEqual(res.dtype, dtype)
            self.assertEqual(res.device.type, device.type)
            self.assertEqual(res.shape, torch.Size())

    def test_single_eval_neg_aug_branin_cuda(self):
        if torch.cuda.is_available():
            self.test_single_eval_neg_aug_branin(cuda=True)

    def test_batch_eval_neg_aug_branin(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            X = torch.zeros(2, 3, device=device, dtype=dtype)
            res = neg_aug_branin(X)
            self.assertEqual(res.dtype, dtype)
            self.assertEqual(res.device.type, device.type)
            self.assertEqual(res.shape, torch.Size([2]))

    def test_batch_eval_neg_aug_branin_cuda(self):
        if torch.cuda.is_available():
            self.test_batch_eval_neg_aug_branin(cuda=True)

    def test_neg_aug_branin_global_maxima(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            X = torch.tensor(
                [
                    [-math.pi, 12.275, 1],
                    [math.pi, 1.3867356039019576, 0.1],
                    [math.pi, 1.781519779945532, 0.5],
                    [math.pi, 2.1763039559891064, 0.9],
                ],
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
            res = neg_aug_branin(X)
            for r in res:
                self.assertAlmostEqual(r.item(), GLOBAL_MAXIMUM, places=4)
            grad = torch.autograd.grad(res.sum(), X)[0]
            self.assertLess(grad.abs().max().item(), 1e-4)

    def test_neg_aug_branin_global_maxima_cuda(self):
        if torch.cuda.is_available():
            self.test_neg_aug_branin_global_maxima(cuda=True)
