#! /usr/bin/env python3

import unittest

import torch
from botorch.test_functions.branin import GLOBAL_MAXIMIZERS, GLOBAL_MAXIMUM, neg_branin


class TestNegBranin(unittest.TestCase):
    def test_single_eval_neg_branin(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            X = torch.zeros(2, device=device, dtype=dtype)
            res = neg_branin(X)
            self.assertEqual(res.dtype, dtype)
            self.assertEqual(res.device.type, device.type)
            self.assertEqual(res.shape, torch.Size())

    def test_single_eval_neg_branin_cuda(self):
        if torch.cuda.is_available():
            self.test_single_eval_neg_branin(cuda=True)

    def test_batch_eval_neg_branin(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            X = torch.zeros(2, 2, device=device, dtype=dtype)
            res = neg_branin(X)
            self.assertEqual(res.dtype, dtype)
            self.assertEqual(res.device.type, device.type)
            self.assertEqual(res.shape, torch.Size([2]))

    def test_batch_eval_neg_branin_cuda(self):
        if torch.cuda.is_available():
            self.test_batch_eval_neg_branin(cuda=True)

    def test_neg_branin_global_maxima(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            X = torch.tensor(
                GLOBAL_MAXIMIZERS, device=device, dtype=dtype, requires_grad=True
            )
            res = neg_branin(X)
            res.sum().backward()
            for r in res:
                self.assertAlmostEqual(r.item(), GLOBAL_MAXIMUM, places=4)
            self.assertLess(X.grad.abs().max().item(), 1e-4)

    def test_neg_branin_global_maxima_cuda(self):
        if torch.cuda.is_available():
            self.test_neg_branin_global_maxima(cuda=True)
