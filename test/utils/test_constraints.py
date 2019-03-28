#! /usr/bin/env python3

import unittest

import torch
from botorch.utils import get_outcome_constraint_transforms


class TestGetOutcomeConstraintTransform(unittest.TestCase):
    def setUp(self):
        self.A = torch.tensor([[-1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
        self.b = torch.tensor([[-0.5], [1.0]])
        self.Ys = torch.tensor([[0.75, 1.0, 0.5], [0.25, 1.5, 1.0]]).unsqueeze(0)
        self.results = torch.tensor([[-0.25, 0.5], [0.25, 1.5]]).view(1, 2, 2)

    def test_None(self):
        self.assertIsNone(get_outcome_constraint_transforms(None))

    def test_BasicEvaluation(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            A = self.A.to(dtype=dtype, device=device)
            b = self.b.to(dtype=dtype, device=device)
            Ys = self.Ys.to(dtype=dtype, device=device)
            results = self.results.to(dtype=dtype, device=device)
            ocs = get_outcome_constraint_transforms((A, b))
            self.assertEqual(len(ocs), 2)
            for i in (0, 1):
                for j in (0, 1):
                    print(f"Actual: {ocs[j](Ys[:,i])}")
                    print(f"Expected: {results[:, i, j]}")
                    self.assertTrue(torch.equal(ocs[j](Ys[:, i]), results[:, i, j]))

    def test_BasicEvaluation_cuda(self):
        if torch.cuda.is_available():
            self.test_BasicEvaluation(cuda=True)

    def test_BroadcastEvaluation(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        k, t = 3, 4
        mc_samples, b, q = 6, 4, 5
        for dtype in (torch.float, torch.double):
            A_ = torch.randn(k, t, dtype=dtype, device=device)
            b_ = torch.randn(k, 1, dtype=dtype, device=device)
            Y = torch.randn(mc_samples, b, q, t, dtype=dtype, device=device)
            ocs = get_outcome_constraint_transforms((A_, b_))
            self.assertEqual(len(ocs), k)
            self.assertEqual(ocs[0](Y).shape, torch.Size([mc_samples, b, q]))

    def test_BroadcastEvaluation_cuda(self):
        if torch.cuda.is_available():
            self.test_BroadcastEvaluation(cuda=True)
