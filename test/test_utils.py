#! /usr/bin/env python3

import unittest

import torch
from botorch.utils import (
    check_convergence,
    columnwise_clamp,
    fix_features,
    gen_x_uniform,
    get_objective_weights_transform,
    manual_seed,
)


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


class TestFixFeatures(unittest.TestCase):
    def setUp(self):
        self.X = torch.tensor([[-2, 1, 3], [0.5, -0.5, 1.0]], requires_grad=True)
        self.X_null_two = torch.tensor(
            [[-2, 1, 3], [0.5, -0.5, 1.0]], requires_grad=True
        )
        self.X_expected = torch.tensor([[-1, 1, -2], [-1, -0.5, -2]])
        self.X_expected_null_two = torch.tensor([[-1, 1, 3], [-1, -0.5, 1.0]])

    def test_fix_features(self, cuda=False):
        X = self.X.cuda() if cuda else self.X
        X_expected = self.X_expected.cuda() if cuda else self.X_expected
        X_null_two = self.X_null_two.cuda() if cuda else self.X_null_two
        X_expected_null_two = (
            self.X_expected_null_two.cuda() if cuda else self.X_expected_null_two
        )
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
            torch.equal(X.grad, torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]))
        )

        f(X_null_two).backward()
        self.assertTrue(torch.equal(X_null_two.grad, torch.ones_like(X)))
        X_null_two.grad.zero_()
        f(X_fix_null_two).backward()
        self.assertTrue(
            torch.equal(
                X_null_two.grad, torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
            )
        )

    def test_fix_features_cuda(self):
        if torch.cuda.is_available():
            self.test_fix_features(cuda=True)


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


class TestGenXUniform(unittest.TestCase):
    def testGenX(self):
        n = 4
        bounds = torch.tensor([[0.0, 1.0, 2.0], [1.0, 4.0, 5.0]])
        X = gen_x_uniform(n, bounds)
        # Check shape
        self.assertTrue(X.shape == torch.Size((n, bounds.shape[1])))
        # Make sure bounds are satisfied
        self.assertTrue(
            torch.sum(torch.max(X, dim=0)[0] <= bounds[1]) == bounds.shape[1]
        )
        self.assertTrue(
            torch.sum(torch.min(X, dim=0)[0] >= bounds[0]) == bounds.shape[1]
        )


class TestGetObjectiveWeightsTransform(unittest.TestCase):
    def setUp(self):
        self.b = 2
        self.q = 4
        self.mc_samples = 5

    def testNoWeights(self):
        X = torch.ones((self.mc_samples, self.b, self.q), dtype=torch.float32)
        objective_transform = get_objective_weights_transform(None)
        X_transformed = objective_transform(X)
        self.assertTrue(torch.equal(X, X_transformed))
        objective_transform = get_objective_weights_transform(torch.tensor([]))
        X_transformed = objective_transform(X)
        self.assertTrue(torch.equal(X, X_transformed))

    def testOneWeight(self):
        X = torch.ones((self.mc_samples, self.b, self.q))
        objective_transform = get_objective_weights_transform(torch.tensor([-1.0]))
        X_transformed = objective_transform(X)
        self.assertTrue(torch.equal(X, -1 * X_transformed))

    def testMultiTaskWeights(self):
        X = torch.ones((self.mc_samples, self.b, self.q, 2))
        objective_transform = get_objective_weights_transform(torch.tensor([1.0, 1.0]))
        X_transformed = objective_transform(X)
        self.assertTrue(torch.equal(torch.sum(X, dim=-1), X_transformed))

    def testNoMCSamples(self):
        X = torch.ones((self.b, self.q, 2))
        objective_transform = get_objective_weights_transform(torch.tensor([1.0, 1.0]))
        X_transformed = objective_transform(X)
        self.assertTrue(torch.equal(torch.sum(X, dim=-1), X_transformed))


if __name__ == "__main__":
    unittest.main()
