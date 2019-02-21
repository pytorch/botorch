#! /usr/bin/env python3

import unittest

import torch
from botorch.test_functions.branin import GLOBAL_MAXIMIZERS, GLOBAL_MAXIMUM, neg_branin


class TestNegBranin(unittest.TestCase):
    def test_single_eval_neg_branin(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        X = torch.zeros(2, device=device)
        res = neg_branin(X)
        self.assertTrue(res.dtype, torch.float)
        self.assertEqual(res.shape, torch.Size([]))
        res_double = neg_branin(X.double())
        self.assertTrue(res_double.dtype, torch.double)
        self.assertEqual(res_double.shape, torch.Size([]))

    def test_single_eval_neg_branin_cuda(self):
        if torch.cuda.is_available():
            self.test_single_eval_neg_branin(cuda=True)

    def test_batch_eval_neg_branin(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        X = torch.zeros(2, 2, device=device)
        res = neg_branin(X)
        self.assertTrue(res.dtype, torch.float)
        self.assertEqual(res.shape, torch.Size([2]))
        res_double = neg_branin(X.double())
        self.assertTrue(res_double.dtype, torch.double)
        self.assertEqual(res_double.shape, torch.Size([2]))

    def test_batch_eval_neg_branin_cuda(self):
        if torch.cuda.is_available():
            self.test_batch_eval_neg_branin(cuda=True)

    def test_neg_branin_gobal_minima(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        X = torch.tensor(GLOBAL_MAXIMIZERS, device=device, requires_grad=True)
        res = neg_branin(X)
        res.sum().backward()
        for r in res:
            self.assertAlmostEqual(r.item(), GLOBAL_MAXIMUM, places=4)
        self.assertLess(X.grad.abs().max().item(), 1e-4)

    def test_neg_branin_gobal_minima_cuda(self):
        if torch.cuda.is_available():
            self.test_neg_branin_gobal_minima(cuda=True)
