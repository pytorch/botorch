#! /usr/bin/env python3

import unittest

import torch
from botorch.test_functions.styblinski_tang import (
    GLOBAL_MINIMIZER,
    GLOBAL_MINIMUM,
    styblinski_tang,
)


class TestStyblinskiTang(unittest.TestCase):
    def test_single_eval_styblinski_tang(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        X = torch.zeros(3, device=device)
        res = styblinski_tang(X)
        self.assertTrue(res.dtype, torch.float)
        self.assertEqual(res.shape, torch.Size([]))
        self.assertTrue(torch.all(res == 0))
        res_double = styblinski_tang(X.double())
        self.assertTrue(res_double.dtype, torch.double)
        self.assertEqual(res_double.shape, torch.Size([]))
        self.assertTrue(torch.all(res_double == 0))

    def test_single_eval_styblinski_tang_cuda(self):
        if torch.cuda.is_available():
            self.test_single_eval_styblinski_tang(cuda=True)

    def test_batch_eval_styblinski_tang(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        X = torch.zeros(2, 3, device=device)
        res = styblinski_tang(X)
        self.assertTrue(res.dtype, torch.float)
        self.assertEqual(res.shape, torch.Size([2]))
        self.assertTrue(torch.all(res == 0))
        res_double = styblinski_tang(X.double())
        self.assertTrue(res_double.dtype, torch.double)
        self.assertEqual(res_double.shape, torch.Size([2]))
        self.assertTrue(torch.all(res_double == 0))

    def test_batch_eval_styblinski_tang_cuda(self):
        if torch.cuda.is_available():
            self.test_batch_eval_styblinski_tang(cuda=True)

    def test_styblinski_tang_gobal_minimum(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        X = torch.full((3,), GLOBAL_MINIMIZER, device=device, requires_grad=True)
        res = styblinski_tang(X)
        res.backward()
        self.assertAlmostEqual(res.item(), 3 * GLOBAL_MINIMUM, places=4)
        self.assertLess(X.grad.abs().max().item(), 1e-5)

    def test_styblinski_tang_gobal_minimum_cuda(self):
        if torch.cuda.is_available():
            self.test_styblinski_tang_gobal_minimum(cuda=True)


if __name__ == "__main__":
    unittest.main()
