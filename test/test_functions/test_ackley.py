#! /usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

import torch
from botorch.test_functions.ackley import GLOBAL_MAXIMIZER, GLOBAL_MAXIMUM, neg_ackley


DIMENSION = 5


class TestNegAckley(unittest.TestCase):
    def test_single_eval_neg_ackley(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            X = torch.zeros(DIMENSION, device=device, dtype=dtype)
            res = neg_ackley(X)
            self.assertEqual(res.dtype, dtype)
            self.assertEqual(res.device.type, device.type)
            self.assertEqual(res.shape, torch.Size())

    def test_single_eval_neg_ackley_cuda(self):
        if torch.cuda.is_available():
            self.test_single_eval_neg_ackley(cuda=True)

    def test_batch_eval_neg_ackley(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            X = torch.zeros(2, DIMENSION, device=device, dtype=dtype)
            res = neg_ackley(X)
            self.assertEqual(res.dtype, dtype)
            self.assertEqual(res.device.type, device.type)
            self.assertEqual(res.shape, torch.Size([2]))

    def test_batch_eval_neg_ackley_cuda(self):
        if torch.cuda.is_available():
            self.test_batch_eval_neg_ackley(cuda=True)

    def test_neg_ackley_global_maximum(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            X = torch.tensor(
                GLOBAL_MAXIMIZER * DIMENSION,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
            res = neg_ackley(X)
            self.assertAlmostEqual(res.item(), GLOBAL_MAXIMUM, places=4)
            grad = torch.autograd.grad(res, X)[0]
            self.assertLess(grad.abs().max().item(), 1e-4)

    def test_neg_ackley_global_maximum_cuda(self):
        if torch.cuda.is_available():
            self.test_neg_ackley_global_maximum(cuda=True)
