#!/usr/bin/env python3

import unittest

import torch

from .utils import approx_equal


class TestTestUtils(unittest.TestCase):
    def test_approx_equal(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        eps = 1e-4
        t1 = torch.rand(2, 3, device=device)
        t2 = t1 + 0.99 * eps * (1 - 2 * torch.randint_like(t1, 2))
        self.assertTrue(approx_equal(t1, t2, eps))
        t3 = t1 + 1.01 * eps * (1 - 2 * torch.randint_like(t1, 2))
        self.assertFalse(approx_equal(t1, t3, eps))
        with self.assertRaises(RuntimeError):
            approx_equal(t1, torch.rand(2, 4), eps)

    def test_approx_equal_cuda(self):
        if torch.cuda.is_available():
            self.test_approx_equal(cuda=True)
