#! /usr/bin/env python3

import unittest

import torch
from botorch.test_functions.holder_table import (
    GLOBAL_MAXIMIZERS,
    GLOBAL_MAXIMUM,
    neg_holder_table,
)


class TestNegHolderTable(unittest.TestCase):
    def test_single_eval_neg_holder_table(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            X = torch.zeros(2, device=device, dtype=dtype)
            res = neg_holder_table(X)
            self.assertEqual(res.dtype, dtype)
            self.assertEqual(res.device.type, device.type)
            self.assertEqual(res.shape, torch.Size())
            self.assertTrue(res.abs().item() < 1e-6)

    def test_single_eval_neg_holder_table_cuda(self):
        if torch.cuda.is_available():
            self.test_single_eval_neg_holder_table(cuda=True)

    def test_batch_eval_neg_holder_table(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            X = torch.zeros(2, 2, device=device, dtype=dtype)
            res = neg_holder_table(X)
            self.assertEqual(res.dtype, dtype)
            self.assertEqual(res.device.type, device.type)
            self.assertEqual(res.shape, torch.Size([2]))
            self.assertTrue(res.abs().sum().item() < 1e-6)

    def test_batch_eval_neg_holder_table_cuda(self):
        if torch.cuda.is_available():
            self.test_batch_eval_neg_holder_table(cuda=True)

    def test_neg_holder_table_global_maxima(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            X = torch.tensor(
                GLOBAL_MAXIMIZERS, device=device, dtype=dtype, requires_grad=True
            )
            res = neg_holder_table(X)
            torch.autograd.backward([*res])
            self.assertTrue(torch.max((res - GLOBAL_MAXIMUM).abs()) < 1e-5)
            self.assertLess(X.grad.abs().max().item(), 1e-3)

    def test_neg_holder_table_global_maxima_cuda(self):
        if torch.cuda.is_available():
            self.test_neg_holder_table_global_maxima(cuda=True)
