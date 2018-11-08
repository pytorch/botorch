#! /usr/bin/env python3

import unittest

import torch
from botorch.test_functions.branin import GLOBAL_MINIMIZERS, GLOBAL_MINIMUM, branin


class TestBranin(unittest.TestCase):
    def test_single_eval_branin(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        X = torch.zeros(2, device=device)
        res = branin(X)
        self.assertTrue(res.dtype, torch.float)
        self.assertEqual(res.shape, torch.Size([]))
        res_double = branin(X.double())
        self.assertTrue(res_double.dtype, torch.double)
        self.assertEqual(res_double.shape, torch.Size([]))

    def test_single_eval_branin_cuda(self):
        if torch.cuda.is_available():
            self.test_single_eval_branin(cuda=True)

    def test_batch_eval_branin(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        X = torch.zeros(2, 2, device=device)
        res = branin(X)
        self.assertTrue(res.dtype, torch.float)
        self.assertEqual(res.shape, torch.Size([2]))
        res_double = branin(X.double())
        self.assertTrue(res_double.dtype, torch.double)
        self.assertEqual(res_double.shape, torch.Size([2]))

    def test_batch_eval_branin_cuda(self):
        if torch.cuda.is_available():
            self.test_batch_eval_branin(cuda=True)

    def test_branin_gobal_minima(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        X = torch.tensor(GLOBAL_MINIMIZERS, device=device, requires_grad=True)
        res = branin(X)
        res.sum().backward()
        for r in res:
            self.assertAlmostEqual(r.item(), GLOBAL_MINIMUM, places=4)
        self.assertLess(X.grad.abs().max().item(), 1e-4)

    def test_branin_gobal_minima_cuda(self):
        if torch.cuda.is_available():
            self.test_branin_gobal_minima(cuda=True)


if __name__ == "__main__":
    unittest.main()
