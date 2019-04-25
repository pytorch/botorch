#! /usr/bin/env python3

import unittest
import warnings

import torch
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.optim import initialize_q_batch, initialize_q_batch_nonneg


class TestInitializeQBatch(unittest.TestCase):
    def test_initialize_q_batch_nonneg(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            # basic test
            X = torch.rand(5, 3, 4, device=device, dtype=dtype)
            Y = torch.rand(5, device=device, dtype=dtype)
            ics = initialize_q_batch_nonneg(X=X, Y=Y, n=2)
            self.assertEqual(ics.shape, torch.Size([2, 3, 4]))
            self.assertEqual(ics.device, X.device)
            self.assertEqual(ics.dtype, X.dtype)
            # ensure nothing happens if we want all samples
            ics = initialize_q_batch_nonneg(X=X, Y=Y, n=5)
            self.assertTrue(torch.equal(X, ics))
            # make sure things work with constant inputs
            Y = torch.ones(5, device=device, dtype=dtype)
            ics = initialize_q_batch_nonneg(X=X, Y=Y, n=2)
            self.assertEqual(ics.shape, torch.Size([2, 3, 4]))
            self.assertEqual(ics.device, X.device)
            self.assertEqual(ics.dtype, X.dtype)
            # ensure raises correct warning
            Y = torch.zeros(5, device=device, dtype=dtype)
            with warnings.catch_warnings(record=True) as w:
                ics = initialize_q_batch_nonneg(X=X, Y=Y, n=2)
                self.assertEqual(len(w), 1)
                self.assertTrue(issubclass(w[-1].category, BadInitialCandidatesWarning))
            self.assertEqual(ics.shape, torch.Size([2, 3, 4]))
            with self.assertRaises(RuntimeError):
                initialize_q_batch_nonneg(X=X, Y=Y, n=10)
            # test less than `n` positive acquisition values
            Y = torch.arange(5, device=device, dtype=dtype) - 3
            ics = initialize_q_batch_nonneg(X=X, Y=Y, n=2)
            self.assertEqual(ics.shape, torch.Size([2, 3, 4]))
            self.assertEqual(ics.device, X.device)
            self.assertEqual(ics.dtype, X.dtype)
            # check that we chose the point with the positive acquisition value
            self.assertTrue(torch.equal(ics[0], X[-1]) or torch.equal(ics[1], X[-1]))
            # test less than `n` alpha_pos values
            Y = torch.arange(5, device=device, dtype=dtype)
            ics = initialize_q_batch_nonneg(X=X, Y=Y, n=2, alpha=1.0)
            self.assertEqual(ics.shape, torch.Size([2, 3, 4]))
            self.assertEqual(ics.device, X.device)
            self.assertEqual(ics.dtype, X.dtype)

    def test_initialize_q_batch_nonneg_cuda(self):
        if torch.cuda.is_available():
            self.test_initialize_q_batch_nonneg(cuda=True)

    def test_initialize_q_batch(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            # basic test
            X = torch.rand(5, 3, 4, device=device, dtype=dtype)
            Y = torch.rand(5, device=device, dtype=dtype)
            ics = initialize_q_batch(X=X, Y=Y, n=2)
            self.assertEqual(ics.shape, torch.Size([2, 3, 4]))
            self.assertEqual(ics.device, X.device)
            self.assertEqual(ics.dtype, X.dtype)
            # ensure nothing happens if we want all samples
            ics = initialize_q_batch(X=X, Y=Y, n=5)
            self.assertTrue(torch.equal(X, ics))
            # ensure raises correct warning
            Y = torch.zeros(5, device=device, dtype=dtype)
            with warnings.catch_warnings(record=True) as w:
                ics = initialize_q_batch(X=X, Y=Y, n=2)
                self.assertEqual(len(w), 1)
                self.assertTrue(issubclass(w[-1].category, BadInitialCandidatesWarning))
            self.assertEqual(ics.shape, torch.Size([2, 3, 4]))
            with self.assertRaises(RuntimeError):
                initialize_q_batch(X=X, Y=Y, n=10)

    def test_initialize_q_batch_cuda(self):
        if torch.cuda.is_available():
            self.test_initialize_q_batch(cuda=True)
