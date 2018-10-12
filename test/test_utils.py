#! /usr/bin/env python3

import unittest

import torch
from botorch.utils import check_convergence, columnwise_clamp, manual_seed


class TestCheckConvergence(unittest.TestCase):
    def test_check_convergence(self, cuda=False):
        losses = torch.rand(5).tolist()
        self.assertTrue(
            check_convergence(
                loss_trajectory=losses, param_trajectory=[], options={}, max_iter=5
            )
        )
        self.assertFalse(
            check_convergence(
                loss_trajectory=losses, param_trajectory=[], options={}, max_iter=6
            )
        )

    def test_check_convergence_cuda(self):
        if torch.cuda.is_available():
            self.test_check_convergence(cuda=True)


class TestColumnWiseClamp(unittest.TestCase):
    def setUp(self):
        self.X = torch.tensor([[-2, 1], [0.5, -0.5]])
        self.X_expected = torch.tensor([[-1, 0.5], [0.5, -0.5]])

    def test_column_wise_clamp_scalars(self, cuda=False):
        X = self.X.cuda() if cuda else self.X
        X_expected = self.X_expected.cuda() if cuda else self.X_expected
        with self.assertRaises(ValueError):
            X_clmp = columnwise_clamp(X, 1, -1)
        X_clmp = columnwise_clamp(X, -1, 0.5)
        self.assertTrue(torch.equal(X_clmp, X_expected))
        X_clmp = columnwise_clamp(X, -3, 3)
        self.assertTrue(torch.equal(X_clmp, X))

    def test_column_wise_clamp_scalars_cuda(self):
        if torch.cuda.is_available():
            self.test_column_wise_clamp_scalars(cuda=True)

    def test_column_wise_clamp_scalar_tensors(self, cuda=False):
        X = self.X.cuda() if cuda else self.X
        X_expected = self.X_expected.cuda() if cuda else self.X_expected
        with self.assertRaises(ValueError):
            X_clmp = columnwise_clamp(X, torch.tensor(1), torch.tensor(-1))
        X_clmp = columnwise_clamp(X, torch.tensor(-1), torch.tensor(0.5))
        self.assertTrue(torch.equal(X_clmp, X_expected))
        X_clmp = columnwise_clamp(X, torch.tensor(-3), torch.tensor(3))
        self.assertTrue(torch.equal(X_clmp, X))

    def test_column_wise_clamp_scalar_tensors_cuda(self):
        if torch.cuda.is_available():
            self.test_column_wise_clamp_scalar_tensors(cuda=True)

    def test_column_wise_clamp_tensors(self, cuda=False):
        X = self.X.cuda() if cuda else self.X
        X_expected = self.X_expected.cuda() if cuda else self.X_expected
        with self.assertRaises(ValueError):
            X_clmp = columnwise_clamp(X, torch.ones(2), torch.zeros(2))
        with self.assertRaises(RuntimeError):
            X_clmp = columnwise_clamp(X, torch.zeros(3), torch.ones(3))
        X_clmp = columnwise_clamp(X, torch.tensor([-1, -1]), torch.tensor([0.5, 0.5]))
        self.assertTrue(torch.equal(X_clmp, X_expected))
        X_clmp = columnwise_clamp(X, torch.tensor([-3, -3]), torch.tensor([3, 3]))
        self.assertTrue(torch.equal(X_clmp, X))

    def test_column_wise_clamp_tensors_cuda(self):
        if torch.cuda.is_available():
            self.test_column_wise_clamp_tensors(cuda=True)


class TestManualSeed(unittest.TestCase):
    def test_manual_seed(self):
        initial_state = torch.random.get_rng_state()
        with manual_seed():
            self.assertTrue(torch.all(torch.random.get_rng_state() == initial_state))
        with manual_seed(1234):
            self.assertFalse(torch.all(torch.random.get_rng_state() == initial_state))
        self.assertTrue(torch.all(torch.random.get_rng_state() == initial_state))


if __name__ == "__main__":
    unittest.main()
