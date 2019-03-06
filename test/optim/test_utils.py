#! /usr/bin/env python3

import unittest

import torch
from botorch.optim.utils import check_convergence, columnwise_clamp, fix_features


class TestCheckConvergence(unittest.TestCase):
    def test_check_convergence(self, cuda=False):
        losses = torch.rand(5).tolist()
        self.assertTrue(
            check_convergence(
                loss_trajectory=losses, param_trajectory=[], options={"maxiter": 5}
            )
        )
        self.assertFalse(
            check_convergence(
                loss_trajectory=losses, param_trajectory=[], options={"maxiter": 6}
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


class TestFixFeatures(unittest.TestCase):
    def _getTensors(self, device):
        X = torch.tensor([[-2, 1, 3], [0.5, -0.5, 1.0]], device=device)
        X_null_two = torch.tensor([[-2, 1, 3], [0.5, -0.5, 1.0]], device=device)
        X_expected = torch.tensor([[-1, 1, -2], [-1, -0.5, -2]], device=device)
        X_expected_null_two = torch.tensor([[-1, 1, 3], [-1, -0.5, 1.0]], device=device)
        return X, X_null_two, X_expected, X_expected_null_two

    def test_fix_features(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        X, X_null_two, X_expected, X_expected_null_two = self._getTensors(device)
        X.requires_grad_(True)
        X_null_two.requires_grad_(True)

        X_fix = fix_features(X, {0: -1, 2: -2})
        X_fix_null_two = fix_features(X_null_two, {0: -1, 2: None})

        self.assertTrue(torch.equal(X_fix, X_expected))
        self.assertTrue(torch.equal(X_fix_null_two, X_expected_null_two))

        def f(X):
            return X.sum()

        f(X).backward()
        self.assertTrue(torch.equal(X.grad, torch.ones_like(X)))
        X.grad.zero_()

        f(X_fix).backward()
        self.assertTrue(
            torch.equal(
                X.grad, torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]], device=device)
            )
        )

        f(X_null_two).backward()
        self.assertTrue(torch.equal(X_null_two.grad, torch.ones_like(X)))
        X_null_two.grad.zero_()
        f(X_fix_null_two).backward()
        self.assertTrue(
            torch.equal(
                X_null_two.grad,
                torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]], device=device),
            )
        )

    def test_fix_features_cuda(self):
        if torch.cuda.is_available():
            self.test_fix_features(cuda=True)
