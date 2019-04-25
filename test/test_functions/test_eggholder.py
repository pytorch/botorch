#! /usr/bin/env python3

import unittest

import torch
from botorch.test_functions.eggholder import (
    GLOBAL_MAXIMIZER,
    GLOBAL_MAXIMUM,
    neg_eggholder,
)


class TestNegEggholder(unittest.TestCase):
    def test_single_eval_neg_eggholder(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            X = torch.zeros(2, device=device, dtype=dtype)
            res = neg_eggholder(X)
            self.assertEqual(res.dtype, dtype)
            self.assertEqual(res.device.type, device.type)
            self.assertEqual(res.shape, torch.Size())

    def test_single_eval_neg_eggholder_cuda(self):
        if torch.cuda.is_available():
            self.test_single_eval_neg_eggholder(cuda=True)

    def test_batch_eval_neg_eggholder(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            X = torch.zeros(2, 2, device=device, dtype=dtype)
            res = neg_eggholder(X)
            self.assertEqual(res.dtype, dtype)
            self.assertEqual(res.device.type, device.type)
            self.assertEqual(res.shape, torch.Size([2]))

    def test_batch_eval_neg_eggholder_cuda(self):
        if torch.cuda.is_available():
            self.test_batch_eval_neg_eggholder(cuda=True)

    def test_neg_eggholder_global_maximum(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            X = torch.tensor(
                GLOBAL_MAXIMIZER, device=device, dtype=dtype, requires_grad=True
            )
            res = neg_eggholder(X)
            res.backward()
            self.assertAlmostEqual(res.item(), GLOBAL_MAXIMUM, places=4)
            self.assertGreater(X.grad.abs().max().item(), 3.0)

    def test_neg_eggholder_global_maximum_cuda(self):
        if torch.cuda.is_available():
            self.test_neg_eggholder_global_maximum(cuda=False)
