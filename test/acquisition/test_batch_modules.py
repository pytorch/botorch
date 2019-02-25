#! /usr/bin/env python3

import unittest
from test.utils.mock import MockModel, MockPosterior

import torch
from botorch.acquisition.batch_modules import (
    qExpectedImprovement,
    qNoisyExpectedImprovement,
)


class TestBatchAcquisitionModules(unittest.TestCase):
    def test_q_expected_improvement(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            samples = torch.zeros((1, 1, 1, 1), device=device, dtype=dtype)
            mm = MockModel(MockPosterior(samples=samples))
            X = torch.zeros((1, 1), device=device, dtype=dtype)  # dummy for typing
            # basic test
            acq_module = qExpectedImprovement(
                model=mm, best_f=0, mc_samples=2, qmc=False
            )
            res = acq_module(X)
            self.assertEqual(res.item(), 0)
            # test X_pending
            samples = torch.zeros((1, 1, 2, 1), device=device, dtype=dtype)
            mm = MockModel(MockPosterior(samples=samples))
            X_pending = torch.zeros((1, 1), device=device, dtype=dtype)
            acq_module = qExpectedImprovement(
                model=mm, best_f=0, mc_samples=2, X_pending=X_pending, qmc=False
            )
            res2 = acq_module(X)
            self.assertEqual(res2.item(), 0)
            # test X_pending w/ batch mode
            samples = torch.zeros((1, 2, 2, 1), device=device, dtype=dtype)
            samples[0, 0, 0, 0] = 1
            mm = MockModel(MockPosterior(samples=samples))
            acq_module = qExpectedImprovement(
                model=mm, best_f=0, mc_samples=2, X_pending=X_pending, qmc=False
            )
            res3 = acq_module(X.unsqueeze(0).expand(2, -1, -1))
            self.assertEqual(res3[0].item(), 1)
            self.assertEqual(res3[1].item(), 0)

    def test_q_expected_improvement_cuda(self):
        if torch.cuda.is_available():
            self.test_q_expected_improvement(cuda=True)

    def test_q_noisy_expected_improvement(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            samples_noisy = torch.tensor([1.0, 0.0], device=device, dtype=dtype)
            samples_noisy = samples_noisy.view(1, 1, 2, 1)
            X_observed = torch.zeros((1, 1), device=device, dtype=dtype)
            mm_noisy = MockModel(MockPosterior(samples=samples_noisy))

            X = torch.zeros((1, 1), device=device, dtype=dtype)
            # basic test
            acq_module = qNoisyExpectedImprovement(
                model=mm_noisy, X_observed=X_observed, mc_samples=2, qmc=False
            )
            res = acq_module(X)
            self.assertEqual(res.item(), 1)

            # test X_pending
            samples_noisy = torch.tensor(
                [1.0, 0.0, 0.0], device=device, dtype=dtype
            ).view(1, 1, 3, 1)
            mm_noisy = MockModel(MockPosterior(samples=samples_noisy))
            X_pending = torch.zeros((1, 1), device=device, dtype=dtype)
            acq_module = qExpectedImprovement(
                model=mm_noisy, best_f=0, mc_samples=2, X_pending=X_pending, qmc=False
            )
            res2 = acq_module(X)
            self.assertEqual(res2.item(), 1)

            # test X_pending w/ batch mode
            samples_noisy = torch.zeros((1, 2, 3, 1), device=device, dtype=dtype)
            samples_noisy[0, 0, 0, 0] = 1.0
            mm_noisy = MockModel(MockPosterior(samples=samples_noisy))
            X_pending = torch.zeros((1, 1), device=device, dtype=dtype)
            acq_module = qExpectedImprovement(
                model=mm_noisy, best_f=0, mc_samples=2, X_pending=X_pending, qmc=False
            )
            res3 = acq_module(X.unsqueeze(0).expand(2, -1, -1))
            self.assertEqual(res3[0].item(), 1)
            self.assertEqual(res3[1].item(), 0)

    def test_q_noisy_expected_improvement_cuda(self):
        if torch.cuda.is_available():
            self.test_q_noisy_expected_improvement(cuda=True)
