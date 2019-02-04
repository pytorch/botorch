#! /usr/bin/env python3

import unittest

import torch
from botorch.exceptions import BadInitialCandidatesError
from botorch.optim import initialize_q_batch_simple


class TestSimpleQBatchInitialization(unittest.TestCase):
    def test_initialize_q_batch_simple(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            # basic test
            X = torch.rand(5, 3, 4, device=device, dtype=dtype)
            Y = torch.rand(5, device=device, dtype=dtype)
            ics = initialize_q_batch_simple(X=X, Y=Y, n=2)
            self.assertEqual(ics.shape, torch.Size([2, 3, 4]))
            self.assertEqual(ics.device, X.device)
            self.assertEqual(ics.dtype, X.dtype)
            # ensure nothing happens if we want all samples
            ics = initialize_q_batch_simple(X=X, Y=Y, n=5)
            self.assertTrue(torch.equal(X, ics))
            # make sure things work with non-noisy inputs
            Y = torch.ones(5, device=device, dtype=dtype)
            ics = initialize_q_batch_simple(X=X, Y=Y, n=2)
            self.assertEqual(ics.shape, torch.Size([2, 3, 4]))
            self.assertEqual(ics.device, X.device)
            self.assertEqual(ics.dtype, X.dtype)
            # ensure raises correct exceptions
            Y = torch.zeros(5, device=device, dtype=dtype)
            with self.assertRaises(BadInitialCandidatesError):
                initialize_q_batch_simple(X=X, Y=Y, n=2)
            with self.assertRaises(RuntimeError):
                initialize_q_batch_simple(X=X, Y=Y, n=10)

    def test_initialize_q_batch_simple_cuda(self):
        if torch.cuda.is_available():
            self.test_initialize_q_batch_simple(cuda=True)


if __name__ == "__main__":
    unittest.main()
