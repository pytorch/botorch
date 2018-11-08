#! /usr/bin/env python3

import unittest

import torch
from botorch.test_functions.styblinski_tang import (
    GLOBAL_MAXIMIZER,
    GLOBAL_MAXIMUM,
    neg_styblinski_tang,
)


class TestNegStyblinskiTang(unittest.TestCase):
    def test_single_eval_neg_styblinski_tang(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        X = torch.zeros(3, device=device)
        res = neg_styblinski_tang(X)
        self.assertTrue(res.dtype, torch.float)
        self.assertEqual(res.shape, torch.Size([]))
        self.assertTrue(torch.all(res == 0))
        res_double = neg_styblinski_tang(X.double())
        self.assertTrue(res_double.dtype, torch.double)
        self.assertEqual(res_double.shape, torch.Size([]))
        self.assertTrue(torch.all(res_double == 0))

    def test_single_eval_neg_styblinski_tang_cuda(self):
        if torch.cuda.is_available():
            self.test_single_eval_neg_styblinski_tang(cuda=True)

    def test_batch_eval_neg_styblinski_tang(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        X = torch.zeros(2, 3, device=device)
        res = neg_styblinski_tang(X)
        self.assertTrue(res.dtype, torch.float)
        self.assertEqual(res.shape, torch.Size([2]))
        self.assertTrue(torch.all(res == 0))
        res_double = neg_styblinski_tang(X.double())
        self.assertTrue(res_double.dtype, torch.double)
        self.assertEqual(res_double.shape, torch.Size([2]))
        self.assertTrue(torch.all(res_double == 0))

    def test_batch_eval_neg_styblinski_tang_cuda(self):
        if torch.cuda.is_available():
            self.test_batch_eval_neg_styblinski_tang(cuda=True)

    def test_neg_styblinski_tang_gobal_minimum(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        X = torch.full((3,), GLOBAL_MAXIMIZER, device=device, requires_grad=True)
        res = neg_styblinski_tang(X)
        res.backward()
        self.assertAlmostEqual(res.item(), 3 * GLOBAL_MAXIMUM, places=4)
        self.assertLess(X.grad.abs().max().item(), 1e-5)

    def test_neg_styblinski_tang_gobal_minimum_cuda(self):
        if torch.cuda.is_available():
            self.test_neg_styblinski_tang_gobal_minimum(cuda=True)


if __name__ == "__main__":
    unittest.main()
