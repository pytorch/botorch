#! /usr/bin/env python3

import unittest
from test.utils.mock import MockModel, MockPosterior

import torch
from botorch.acquisition.functional.batch_acquisition import (
    batch_expected_improvement,
    batch_noisy_expected_improvement,
    get_infeasible_cost,
)


class TestFunctionalBatchAcquisition(unittest.TestCase):
    def test_batch_expected_improvement(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            samples = torch.zeros(1, 1, 1, 1, device=device, dtype=dtype)
            mm = MockModel(MockPosterior(samples=samples))
            X = torch.zeros(1, device=device, dtype=dtype)  # dummy for type checking
            # basic test
            res = batch_expected_improvement(X=X, model=mm, best_f=0, mc_samples=2)
            self.assertEqual(res.item(), 0)
            # test shifting best_f value
            res2 = batch_expected_improvement(X=X, model=mm, best_f=-1, mc_samples=2)
            self.assertEqual(res2.item(), 1)
            # test modifying the objective
            res3 = batch_expected_improvement(
                X=X,
                model=mm,
                best_f=0,
                objective=lambda Y: torch.ones_like(Y).squeeze(-1),
                mc_samples=2,
            )
            self.assertEqual(res3.item(), 1)
            # test constraints
            res4 = batch_expected_improvement(
                X=X,
                model=mm,
                best_f=-1,
                constraints=[lambda Y: torch.zeros_like(Y).squeeze(-1)],
                mc_samples=2,
            )
            # the constraint returns 0 (feasible), obj samples are 0, so the expected
            # improvement is 0 - (-1) = 1. Note: there is no longer a soft constraint.
            self.assertEqual(res4.item(), 1.0)

    def test_batch_expected_improvement_cuda(self):
        if torch.cuda.is_available():
            self.test_batch_expected_improvement(cuda=True)

    def test_batch_noisy_expected_improvement(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")

        for dtype in (torch.float, torch.double):
            samples_noisy = torch.tensor([1.0, 0.0], device=device, dtype=dtype)
            samples_noisy = samples_noisy.view(1, 1, 2, 1)
            X_observed = torch.zeros(1, 1, device=device, dtype=dtype)
            mm_noisy = MockModel(MockPosterior(samples=samples_noisy))

            X = torch.zeros(1, 1, device=device, dtype=dtype)
            # basic test
            res = batch_noisy_expected_improvement(
                X=X, model=mm_noisy, X_observed=X_observed, mc_samples=2
            )
            self.assertEqual(res.item(), 1)
            # test modifying the objective
            res3 = batch_noisy_expected_improvement(
                X=X,
                model=mm_noisy,
                X_observed=X_observed,
                objective=lambda Y: torch.zeros_like(Y).squeeze(-1),
                mc_samples=2,
            )
            self.assertEqual(res3.item(), 0)
            # test constraints
            res4 = batch_noisy_expected_improvement(
                X=X,
                model=mm_noisy,
                X_observed=X_observed,
                constraints=[lambda Y: torch.zeros_like(Y).squeeze(-1)],
                mc_samples=2,
            )
            # the constraint returns 0 (feasible), samples at X_observed are 0,
            # samples at X are 1 so the noisy expected improvement is 1 - 0 = 1.
            # Note: there is no longer a soft constraint.
            self.assertEqual(res4.item(), 1.0)

    def test_batch_noisy_expected_improvement_cuda(self):
        if torch.cuda.is_available():
            self.test_batch_noisy_expected_improvement(cuda=True)

    def test_get_infeasible_cost(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")

        for dtype in (torch.float, torch.double):
            X = torch.zeros(5, 1)
            means = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=device, dtype=dtype)
            variances = torch.tensor(
                [0.09, 0.25, 0.36, 0.25, 0.09], device=device, dtype=dtype
            )
            mm = MockModel(MockPosterior(mean=means, variance=variances))
            # means - 6 * std = [-0.8, -1, -0.6, 1, 3.2],
            # after applying objective, the minimum becomes -6.0
            # so 6.0 should be returned.
            M = get_infeasible_cost(
                X=X, model=mm, objective=lambda Y: Y.squeeze(-1) - 5
            )
            self.assertEqual(M, 6.0)

    def test_get_infeasible_cost_cuda(self):
        if torch.cuda.is_available():
            self.test_get_infeasible_cost(cuda=True)


if __name__ == "__main__":
    unittest.main()
