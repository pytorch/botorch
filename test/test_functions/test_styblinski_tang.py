#! /usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import torch
from botorch.test_functions.styblinski_tang import (
    GLOBAL_MAXIMIZER,
    GLOBAL_MAXIMUM,
    neg_styblinski_tang,
)

from ..botorch_test_case import BotorchTestCase


class TestNegStyblinskiTang(BotorchTestCase):
    def test_single_eval_neg_styblinski_tang(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            X = torch.zeros(3, device=device, dtype=dtype)
            res = neg_styblinski_tang(X)
            self.assertEqual(res.dtype, dtype)
            self.assertEqual(res.device.type, device.type)
            self.assertEqual(res.shape, torch.Size())
            self.assertTrue(torch.all(res == 0))

    def test_single_eval_neg_styblinski_tang_cuda(self):
        if torch.cuda.is_available():
            self.test_single_eval_neg_styblinski_tang(cuda=True)

    def test_batch_eval_neg_styblinski_tang(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            X = torch.zeros(2, 3, device=device, dtype=dtype)
            res = neg_styblinski_tang(X)
            self.assertEqual(res.dtype, dtype)
            self.assertEqual(res.device.type, device.type)
            self.assertEqual(res.shape, torch.Size([2]))
            self.assertTrue(torch.all(res == 0))

    def test_batch_eval_neg_styblinski_tang_cuda(self):
        if torch.cuda.is_available():
            self.test_batch_eval_neg_styblinski_tang(cuda=True)

    def test_neg_styblinski_tang_global_maximum(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            X = torch.full(
                (3,), GLOBAL_MAXIMIZER, device=device, dtype=dtype, requires_grad=True
            )
            res = neg_styblinski_tang(X)
            self.assertAlmostEqual(res.item(), 3 * GLOBAL_MAXIMUM, places=4)
            grad = torch.autograd.grad(res, X)[0]
            self.assertLess(grad.abs().max().item(), 1e-5)

    def test_neg_styblinski_tang_global_maximum_cuda(self):
        if torch.cuda.is_available():
            self.test_neg_styblinski_tang_global_maximum(cuda=True)
