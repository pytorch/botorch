#! /usr/bin/env python3

import unittest

import torch
from botorch.acquisition.monte_carlo import (
    MCAcquisitionFunction,
    qExpectedImprovement,
    qNoisyExpectedImprovement,
    qProbabilityOfImprovement,
    qSimpleRegret,
    qUpperConfidenceBound,
)
from botorch.acquisition.sampler import IIDNormalSampler, SobolQMCNormalSampler

from ..mock import MockModel, MockPosterior


# TODO: T41739913 Implement tests for all MCAcquisitionFunctions


class TestMCAcquisitionFunction(unittest.TestCase):
    def test_abstract_raises(self):
        with self.assertRaises(TypeError):
            MCAcquisitionFunction()


class TestQExpectedImprovement(unittest.TestCase):
    def test_q_expected_improvement(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            # the event shape is `b x q x t` = 1 x 1 x 1
            samples = torch.zeros(1, 1, 1, device=device, dtype=dtype)
            mm = MockModel(MockPosterior(samples=samples))
            # X is `q x d` = 1 x 1. X is a dummy and unused b/c of mocking
            X = torch.zeros(1, 1, device=device, dtype=dtype)

            # basic test
            sampler = IIDNormalSampler(num_samples=2)
            acqf = qExpectedImprovement(model=mm, best_f=0, sampler=sampler)
            res = acqf(X)
            self.assertEqual(res.item(), 0.0)

            # test shifting best_f value
            acqf = qExpectedImprovement(model=mm, best_f=-1, sampler=sampler)
            res = acqf(X)
            self.assertEqual(res.item(), 1.0)

            # basic test, no resample
            sampler = IIDNormalSampler(num_samples=2, seed=12345)
            acqf = qExpectedImprovement(model=mm, best_f=0, sampler=sampler)
            res = acqf(X)
            self.assertEqual(res.item(), 0.0)
            self.assertEqual(acqf.sampler.base_samples.shape, torch.Size([2, 1, 1, 1]))
            bs = acqf.sampler.base_samples.clone()
            res = acqf(X)
            self.assertTrue(torch.equal(acqf.sampler.base_samples, bs))

            # basic test, qmc, no resample
            sampler = SobolQMCNormalSampler(num_samples=2)
            acqf = qExpectedImprovement(model=mm, best_f=0, sampler=sampler)
            res = acqf(X)
            self.assertEqual(res.item(), 0.0)
            self.assertEqual(acqf.sampler.base_samples.shape, torch.Size([2, 1, 1, 1]))
            bs = acqf.sampler.base_samples.clone()
            acqf(X)
            self.assertTrue(torch.equal(acqf.sampler.base_samples, bs))

            # basic test, qmc, resample
            sampler = SobolQMCNormalSampler(num_samples=2, resample=True)
            acqf = qExpectedImprovement(model=mm, best_f=0, sampler=sampler)
            res = acqf(X)
            self.assertEqual(res.item(), 0.0)
            self.assertEqual(acqf.sampler.base_samples.shape, torch.Size([2, 1, 1, 1]))
            bs = acqf.sampler.base_samples.clone()
            acqf(X)
            self.assertFalse(torch.equal(acqf.sampler.base_samples, bs))

    def test_q_expected_improvement_cuda(self):
        if torch.cuda.is_available():
            self.test_q_expected_improvement(cuda=True)

    def test_q_expected_improvement_batch(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            # the event shape is `b x q x t` = 2 x 2 x 1
            samples = torch.zeros(2, 2, 1, device=device, dtype=dtype)
            samples[0, 0, 0] = 1.0
            mm = MockModel(MockPosterior(samples=samples))

            # X is a dummy and unused b/c of mocking
            X = torch.zeros(1, 1, 1, device=device, dtype=dtype)

            # test batch mode
            sampler = IIDNormalSampler(num_samples=2)
            acqf = qExpectedImprovement(model=mm, best_f=0, sampler=sampler)
            res = acqf(X)
            self.assertEqual(res[0].item(), 1.0)
            self.assertEqual(res[1].item(), 0.0)

            # test shifting best_f value
            acqf = qExpectedImprovement(model=mm, best_f=-1, sampler=sampler)
            res = acqf(X)
            self.assertEqual(res[0].item(), 2.0)
            self.assertEqual(res[1].item(), 1.0)

            # test batch mode, no resample
            sampler = IIDNormalSampler(num_samples=2, seed=12345)
            acqf = qExpectedImprovement(model=mm, best_f=0, sampler=sampler)
            res = acqf(X)  # 1-dim batch
            self.assertEqual(res[0].item(), 1.0)
            self.assertEqual(res[1].item(), 0.0)
            self.assertEqual(acqf.sampler.base_samples.shape, torch.Size([2, 1, 2, 1]))
            bs = acqf.sampler.base_samples.clone()
            acqf(X)
            self.assertTrue(torch.equal(acqf.sampler.base_samples, bs))
            res = acqf(X.expand(2, 1, 1))  # 2-dim batch
            self.assertEqual(res[0].item(), 1.0)
            self.assertEqual(res[1].item(), 0.0)
            # the base samples should have the batch dim collapsed
            self.assertEqual(acqf.sampler.base_samples.shape, torch.Size([2, 1, 2, 1]))
            bs = acqf.sampler.base_samples.clone()
            acqf(X.expand(2, 1, 1))
            self.assertTrue(torch.equal(acqf.sampler.base_samples, bs))

            # test batch mode, qmc, no resample
            sampler = SobolQMCNormalSampler(num_samples=2)
            acqf = qExpectedImprovement(model=mm, best_f=0, sampler=sampler)
            res = acqf(X)
            self.assertEqual(res[0].item(), 1.0)
            self.assertEqual(res[1].item(), 0.0)
            self.assertEqual(acqf.sampler.base_samples.shape, torch.Size([2, 1, 2, 1]))
            bs = acqf.sampler.base_samples.clone()
            acqf(X)
            self.assertTrue(torch.equal(acqf.sampler.base_samples, bs))

            # test batch mode, qmc, resample
            sampler = SobolQMCNormalSampler(num_samples=2, resample=True)
            acqf = qExpectedImprovement(model=mm, best_f=0, sampler=sampler)
            res = acqf(X)  # 1-dim batch
            self.assertEqual(res[0].item(), 1.0)
            self.assertEqual(res[1].item(), 0.0)
            self.assertEqual(acqf.sampler.base_samples.shape, torch.Size([2, 1, 2, 1]))
            bs = acqf.sampler.base_samples.clone()
            acqf(X)
            self.assertFalse(torch.equal(acqf.sampler.base_samples, bs))
            res = acqf(X.expand(2, 1, 1))  # 2-dim batch
            self.assertEqual(res[0].item(), 1.0)
            self.assertEqual(res[1].item(), 0.0)
            # the base samples should have the batch dim collapsed
            self.assertEqual(acqf.sampler.base_samples.shape, torch.Size([2, 1, 2, 1]))
            bs = acqf.sampler.base_samples.clone()
            acqf(X.expand(2, 1, 1))
            self.assertFalse(torch.equal(acqf.sampler.base_samples, bs))

    def test_q_expected_improvement_batch_cuda(self):
        if torch.cuda.is_available():
            self.test_q_expected_improvement_batch(cuda=True)

    # TODO: Test different objectives (incl. constraints)


class TestQNoisyExpectedImprovement(unittest.TestCase):
    def test_q_noisy_expected_improvement(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            # the event shape is `b x q x t` = 1 x 2 x 1
            samples_noisy = torch.tensor([1.0, 0.0], device=device, dtype=dtype)
            samples_noisy = samples_noisy.view(1, 2, 1)
            # X_baseline is `q' x d` = 1 x 1
            X_baseline = torch.zeros(1, 1, device=device, dtype=dtype)
            mm_noisy = MockModel(MockPosterior(samples=samples_noisy))
            # X is `q x d` = 1 x 1
            X = torch.zeros(1, 1, device=device, dtype=dtype)

            # basic test
            sampler = IIDNormalSampler(num_samples=2)
            acqf = qNoisyExpectedImprovement(
                model=mm_noisy, X_baseline=X_baseline, sampler=sampler
            )
            res = acqf(X)
            self.assertEqual(res.item(), 1.0)

            # basic test, no resample
            sampler = IIDNormalSampler(num_samples=2, seed=12345)
            acqf = qNoisyExpectedImprovement(
                model=mm_noisy, X_baseline=X_baseline, sampler=sampler
            )
            res = acqf(X)
            self.assertEqual(res.item(), 1.0)
            self.assertEqual(acqf.sampler.base_samples.shape, torch.Size([2, 1, 2, 1]))
            bs = acqf.sampler.base_samples.clone()
            acqf(X)
            self.assertTrue(torch.equal(acqf.sampler.base_samples, bs))

            # basic test, qmc, no resample
            sampler = SobolQMCNormalSampler(num_samples=2)
            acqf = qNoisyExpectedImprovement(
                model=mm_noisy, X_baseline=X_baseline, sampler=sampler
            )
            res = acqf(X)
            self.assertEqual(res.item(), 1.0)
            self.assertEqual(acqf.sampler.base_samples.shape, torch.Size([2, 1, 2, 1]))
            bs = acqf.sampler.base_samples.clone()
            acqf(X)
            self.assertTrue(torch.equal(acqf.sampler.base_samples, bs))

            # basic test, qmc, resample
            sampler = SobolQMCNormalSampler(num_samples=2, resample=True, seed=12345)
            acqf = qNoisyExpectedImprovement(
                model=mm_noisy, X_baseline=X_baseline, sampler=sampler
            )
            res = acqf(X)
            self.assertEqual(res.item(), 1.0)
            self.assertEqual(acqf.sampler.base_samples.shape, torch.Size([2, 1, 2, 1]))
            bs = acqf.sampler.base_samples.clone()
            acqf(X)
            self.assertFalse(torch.equal(acqf.sampler.base_samples, bs))

    def test_q_noisy_expected_improvement_cuda(self):
        if torch.cuda.is_available():
            self.test_q_noisy_expected_improvement(cuda=True)

    def test_q_noisy_expected_improvement_batch(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            # the event shape is `b x q x t` = 2 x 3 x 1
            samples_noisy = torch.zeros(2, 3, 1, device=device, dtype=dtype)
            samples_noisy[0, 0, 0] = 1.0
            mm_noisy = MockModel(MockPosterior(samples=samples_noisy))
            # X is `q x d` = 1 x 1
            X = torch.zeros(1, 1, 1, device=device, dtype=dtype)
            X_baseline = torch.zeros(1, 1, device=device, dtype=dtype)

            # test batch mode
            sampler = IIDNormalSampler(num_samples=2)
            acqf = qNoisyExpectedImprovement(
                model=mm_noisy, X_baseline=X_baseline, sampler=sampler
            )
            res = acqf(X)
            self.assertEqual(res[0].item(), 1.0)
            self.assertEqual(res[1].item(), 0.0)

            # test batch mode, no resample
            sampler = IIDNormalSampler(num_samples=2, seed=12345)
            acqf = qNoisyExpectedImprovement(
                model=mm_noisy, X_baseline=X_baseline, sampler=sampler
            )
            res = acqf(X)  # 1-dim batch
            self.assertEqual(res[0].item(), 1.0)
            self.assertEqual(res[1].item(), 0.0)
            self.assertEqual(acqf.sampler.base_samples.shape, torch.Size([2, 1, 3, 1]))
            bs = acqf.sampler.base_samples.clone()
            acqf(X)
            self.assertTrue(torch.equal(acqf.sampler.base_samples, bs))
            res = acqf(X.expand(2, 1, 1))  # 2-dim batch
            self.assertEqual(res[0].item(), 1.0)
            self.assertEqual(res[1].item(), 0.0)
            # the base samples should have the batch dim collapsed
            self.assertEqual(acqf.sampler.base_samples.shape, torch.Size([2, 1, 3, 1]))
            bs = acqf.sampler.base_samples.clone()
            acqf(X.expand(2, 1, 1))
            self.assertTrue(torch.equal(acqf.sampler.base_samples, bs))

            # test batch mode, qmc, no resample
            sampler = SobolQMCNormalSampler(num_samples=2)
            acqf = qNoisyExpectedImprovement(
                model=mm_noisy, X_baseline=X_baseline, sampler=sampler
            )
            res = acqf(X)
            self.assertEqual(res[0].item(), 1.0)
            self.assertEqual(res[1].item(), 0.0)
            self.assertEqual(acqf.sampler.base_samples.shape, torch.Size([2, 1, 3, 1]))
            bs = acqf.sampler.base_samples.clone()
            acqf(X)
            self.assertTrue(torch.equal(acqf.sampler.base_samples, bs))

            # test X_pending w/ batch mode, qmc, resample
            sampler = SobolQMCNormalSampler(num_samples=2, resample=True, seed=12345)
            acqf = qNoisyExpectedImprovement(
                model=mm_noisy, X_baseline=X_baseline, sampler=sampler
            )
            res = acqf(X)  # 1-dim batch
            self.assertEqual(res[0].item(), 1.0)
            self.assertEqual(res[1].item(), 0.0)
            self.assertEqual(acqf.sampler.base_samples.shape, torch.Size([2, 1, 3, 1]))
            bs = acqf.sampler.base_samples.clone()
            acqf(X)
            self.assertFalse(torch.equal(acqf.sampler.base_samples, bs))
            res = acqf(X.expand(2, 1, 1))  # 2-dim batch
            self.assertEqual(res[0].item(), 1.0)
            self.assertEqual(res[1].item(), 0.0)
            # the base samples should have the batch dim collapsed
            self.assertEqual(acqf.sampler.base_samples.shape, torch.Size([2, 1, 3, 1]))
            bs = acqf.sampler.base_samples.clone()
            acqf(X.expand(2, 1, 1))
            self.assertFalse(torch.equal(acqf.sampler.base_samples, bs))

    def test_q_noisy_expected_improvement_batch_cuda(self):
        if torch.cuda.is_available():
            self.test_q_noisy_expected_improvement_batch(cuda=True)

    # TODO: Test different objectives (incl. constraints)


class TestQProbabilityOfImprovement(unittest.TestCase):
    def test_q_probability_of_improvement(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            # the event shape is `b x q x t` = 1 x 1 x 1
            samples = torch.zeros(1, 1, 1, device=device, dtype=dtype)
            mm = MockModel(MockPosterior(samples=samples))
            # X is `q x d` = 1 x 1. X is a dummy and unused b/c of mocking
            X = torch.zeros(1, 1, device=device, dtype=dtype)

            # basic test
            sampler = IIDNormalSampler(num_samples=2)
            acqf = qProbabilityOfImprovement(model=mm, best_f=0, sampler=sampler)
            res = acqf(X)
            self.assertEqual(res.item(), 0.5)

            # basic test, no resample
            sampler = IIDNormalSampler(num_samples=2, seed=12345)
            acqf = qProbabilityOfImprovement(model=mm, best_f=0, sampler=sampler)
            res = acqf(X)
            self.assertEqual(res.item(), 0.5)
            self.assertEqual(acqf.sampler.base_samples.shape, torch.Size([2, 1, 1, 1]))
            bs = acqf.sampler.base_samples.clone()
            res = acqf(X)
            self.assertTrue(torch.equal(acqf.sampler.base_samples, bs))

            # basic test, qmc, no resample
            sampler = SobolQMCNormalSampler(num_samples=2)
            acqf = qProbabilityOfImprovement(model=mm, best_f=0, sampler=sampler)
            res = acqf(X)
            self.assertEqual(res.item(), 0.5)
            self.assertEqual(acqf.sampler.base_samples.shape, torch.Size([2, 1, 1, 1]))
            bs = acqf.sampler.base_samples.clone()
            acqf(X)
            self.assertTrue(torch.equal(acqf.sampler.base_samples, bs))

            # basic test, qmc, resample
            sampler = SobolQMCNormalSampler(num_samples=2, resample=True)
            acqf = qProbabilityOfImprovement(model=mm, best_f=0, sampler=sampler)
            res = acqf(X)
            self.assertEqual(res.item(), 0.5)
            self.assertEqual(acqf.sampler.base_samples.shape, torch.Size([2, 1, 1, 1]))
            bs = acqf.sampler.base_samples.clone()
            acqf(X)
            self.assertFalse(torch.equal(acqf.sampler.base_samples, bs))

    def test_q_probability_of_improvement_cuda(self):
        if torch.cuda.is_available():
            self.test_q_probability_of_improvement(cuda=True)

    def test_q_probability_of_improvement_batch(self, cuda=False):
        # the event shape is `b x q x t` = 2 x 2 x 1
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            samples = torch.zeros(2, 2, 1, device=device, dtype=dtype)
            samples[0, 0, 0] = 1.0
            mm = MockModel(MockPosterior(samples=samples))

            # X is a dummy and unused b/c of mocking
            X = torch.zeros(1, 1, 1, device=device, dtype=dtype)

            # test batch mode
            sampler = IIDNormalSampler(num_samples=2)
            acqf = qProbabilityOfImprovement(model=mm, best_f=0, sampler=sampler)
            res = acqf(X)
            self.assertEqual(res[0].item(), 1.0)
            self.assertEqual(res[1].item(), 0.5)

            # test batch mode, no resample
            sampler = IIDNormalSampler(num_samples=2, seed=12345)
            acqf = qProbabilityOfImprovement(model=mm, best_f=0, sampler=sampler)
            res = acqf(X)  # 1-dim batch
            self.assertEqual(res[0].item(), 1.0)
            self.assertEqual(res[1].item(), 0.5)
            self.assertEqual(acqf.sampler.base_samples.shape, torch.Size([2, 1, 2, 1]))
            bs = acqf.sampler.base_samples.clone()
            acqf(X)
            self.assertTrue(torch.equal(acqf.sampler.base_samples, bs))
            res = acqf(X.expand(2, 1, 1))  # 2-dim batch
            self.assertEqual(res[0].item(), 1.0)
            self.assertEqual(res[1].item(), 0.5)
            # the base samples should have the batch dim collapsed
            self.assertEqual(acqf.sampler.base_samples.shape, torch.Size([2, 1, 2, 1]))
            bs = acqf.sampler.base_samples.clone()
            acqf(X.expand(2, 1, 1))
            self.assertTrue(torch.equal(acqf.sampler.base_samples, bs))

            # test batch mode, qmc, no resample
            sampler = SobolQMCNormalSampler(num_samples=2)
            acqf = qProbabilityOfImprovement(model=mm, best_f=0, sampler=sampler)
            res = acqf(X)
            self.assertEqual(res[0].item(), 1.0)
            self.assertEqual(res[1].item(), 0.5)
            self.assertEqual(acqf.sampler.base_samples.shape, torch.Size([2, 1, 2, 1]))
            bs = acqf.sampler.base_samples.clone()
            acqf(X)
            self.assertTrue(torch.equal(acqf.sampler.base_samples, bs))

            # test batch mode, qmc, resample
            sampler = SobolQMCNormalSampler(num_samples=2, resample=True)
            acqf = qProbabilityOfImprovement(model=mm, best_f=0, sampler=sampler)
            res = acqf(X)  # 1-dim batch
            self.assertEqual(res[0].item(), 1.0)
            self.assertEqual(res[1].item(), 0.5)
            self.assertEqual(acqf.sampler.base_samples.shape, torch.Size([2, 1, 2, 1]))
            bs = acqf.sampler.base_samples.clone()
            acqf(X)
            self.assertFalse(torch.equal(acqf.sampler.base_samples, bs))
            res = acqf(X.expand(2, 1, 1))  # 2-dim batch
            self.assertEqual(res[0].item(), 1.0)
            self.assertEqual(res[1].item(), 0.5)
            # the base samples should have the batch dim collapsed
            self.assertEqual(acqf.sampler.base_samples.shape, torch.Size([2, 1, 2, 1]))
            bs = acqf.sampler.base_samples.clone()
            acqf(X.expand(2, 1, 1))
            self.assertFalse(torch.equal(acqf.sampler.base_samples, bs))

    def test_q_probability_of_improvement_batch_cuda(self):
        if torch.cuda.is_available():
            self.test_q_probability_of_improvement_batch(cuda=True)

    # TODO: Test different objectives (incl. constraints)


class TestQSimpleRegret(unittest.TestCase):
    def test_q_simple_regret(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            # the event shape is `b x q x t` = 1 x 1 x 1
            samples = torch.zeros(1, 1, 1, device=device, dtype=dtype)
            mm = MockModel(MockPosterior(samples=samples))
            # X is `q x d` = 1 x 1. X is a dummy and unused b/c of mocking
            X = torch.zeros(1, 1, device=device, dtype=dtype)

            # basic test
            sampler = IIDNormalSampler(num_samples=2)
            acqf = qSimpleRegret(model=mm, sampler=sampler)
            res = acqf(X)
            self.assertEqual(res.item(), 0.0)

            # basic test, no resample
            sampler = IIDNormalSampler(num_samples=2, seed=12345)
            acqf = qSimpleRegret(model=mm, sampler=sampler)
            res = acqf(X)
            self.assertEqual(res.item(), 0.0)
            self.assertEqual(acqf.sampler.base_samples.shape, torch.Size([2, 1, 1, 1]))
            bs = acqf.sampler.base_samples.clone()
            res = acqf(X)
            self.assertTrue(torch.equal(acqf.sampler.base_samples, bs))

            # basic test, qmc, no resample
            sampler = SobolQMCNormalSampler(num_samples=2)
            acqf = qSimpleRegret(model=mm, sampler=sampler)
            res = acqf(X)
            self.assertEqual(res.item(), 0.0)
            self.assertEqual(acqf.sampler.base_samples.shape, torch.Size([2, 1, 1, 1]))
            bs = acqf.sampler.base_samples.clone()
            acqf(X)
            self.assertTrue(torch.equal(acqf.sampler.base_samples, bs))

            # basic test, qmc, resample
            sampler = SobolQMCNormalSampler(num_samples=2, resample=True)
            acqf = qSimpleRegret(model=mm, sampler=sampler)
            res = acqf(X)
            self.assertEqual(res.item(), 0.0)
            self.assertEqual(acqf.sampler.base_samples.shape, torch.Size([2, 1, 1, 1]))
            bs = acqf.sampler.base_samples.clone()
            acqf(X)
            self.assertFalse(torch.equal(acqf.sampler.base_samples, bs))

    def test_q_simple_regret_cuda(self):
        if torch.cuda.is_available():
            self.test_q_simple_regret(cuda=True)

    def test_q_simple_regret_batch(self, cuda=False):
        # the event shape is `b x q x t` = 2 x 2 x 1
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            samples = torch.zeros(2, 2, 1, device=device, dtype=dtype)
            samples[0, 0, 0] = 1.0
            mm = MockModel(MockPosterior(samples=samples))
            # X is a dummy and unused b/c of mocking
            X = torch.zeros(1, 1, 1, device=device, dtype=dtype)

            # test batch mode
            sampler = IIDNormalSampler(num_samples=2)
            acqf = qSimpleRegret(model=mm, sampler=sampler)
            res = acqf(X)
            self.assertEqual(res[0].item(), 1.0)
            self.assertEqual(res[1].item(), 0.0)

            # test batch mode, no resample
            sampler = IIDNormalSampler(num_samples=2, seed=12345)
            acqf = qSimpleRegret(model=mm, sampler=sampler)
            res = acqf(X)  # 1-dim batch
            self.assertEqual(res[0].item(), 1.0)
            self.assertEqual(res[1].item(), 0.0)
            self.assertEqual(acqf.sampler.base_samples.shape, torch.Size([2, 1, 2, 1]))
            bs = acqf.sampler.base_samples.clone()
            acqf(X)
            self.assertTrue(torch.equal(acqf.sampler.base_samples, bs))
            res = acqf(X.expand(2, 1, 1))  # 2-dim batch
            self.assertEqual(res[0].item(), 1.0)
            self.assertEqual(res[1].item(), 0.0)
            # the base samples should have the batch dim collapsed
            self.assertEqual(acqf.sampler.base_samples.shape, torch.Size([2, 1, 2, 1]))
            bs = acqf.sampler.base_samples.clone()
            acqf(X.expand(2, 1, 1))
            self.assertTrue(torch.equal(acqf.sampler.base_samples, bs))

            # test batch mode, qmc, no resample
            sampler = SobolQMCNormalSampler(num_samples=2)
            acqf = qSimpleRegret(model=mm, sampler=sampler)
            res = acqf(X)
            self.assertEqual(res[0].item(), 1.0)
            self.assertEqual(res[1].item(), 0.0)
            self.assertEqual(acqf.sampler.base_samples.shape, torch.Size([2, 1, 2, 1]))
            bs = acqf.sampler.base_samples.clone()
            acqf(X)
            self.assertTrue(torch.equal(acqf.sampler.base_samples, bs))

            # test batch mode, qmc, resample
            sampler = SobolQMCNormalSampler(num_samples=2, resample=True)
            acqf = qSimpleRegret(model=mm, sampler=sampler)
            res = acqf(X)  # 1-dim batch
            self.assertEqual(res[0].item(), 1.0)
            self.assertEqual(res[1].item(), 0.0)
            self.assertEqual(acqf.sampler.base_samples.shape, torch.Size([2, 1, 2, 1]))
            bs = acqf.sampler.base_samples.clone()
            acqf(X)
            self.assertFalse(torch.equal(acqf.sampler.base_samples, bs))
            res = acqf(X.expand(2, 1, 1))  # 2-dim batch
            self.assertEqual(res[0].item(), 1.0)
            self.assertEqual(res[1].item(), 0.0)
            # the base samples should have the batch dim collapsed
            self.assertEqual(acqf.sampler.base_samples.shape, torch.Size([2, 1, 2, 1]))
            bs = acqf.sampler.base_samples.clone()
            acqf(X.expand(2, 1, 1))
            self.assertFalse(torch.equal(acqf.sampler.base_samples, bs))

    def test_q_simple_regret_batch_cuda(self):
        if torch.cuda.is_available():
            self.test_q_simple_regret_batch(cuda=True)

    # TODO: Test different objectives (incl. constraints)


class TestQUpperConfidenceBound(unittest.TestCase):
    def test_q_upper_confidence_bound(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            # the event shape is `b x q x t` = 1 x 1 x 1
            samples = torch.zeros(1, 1, 1, device=device, dtype=dtype)
            mm = MockModel(MockPosterior(samples=samples))
            # X is `q x d` = 1 x 1. X is a dummy and unused b/c of mocking
            X = torch.zeros(1, 1, device=device, dtype=dtype)

            # basic test
            sampler = IIDNormalSampler(num_samples=2)
            acqf = qUpperConfidenceBound(model=mm, beta=0.5, sampler=sampler)
            res = acqf(X)
            self.assertEqual(res.item(), 0.0)

            # basic test, no resample
            sampler = IIDNormalSampler(num_samples=2, seed=12345)
            acqf = qUpperConfidenceBound(model=mm, beta=0.5, sampler=sampler)
            res = acqf(X)
            self.assertEqual(res.item(), 0.0)
            self.assertEqual(acqf.sampler.base_samples.shape, torch.Size([2, 1, 1, 1]))
            bs = acqf.sampler.base_samples.clone()
            res = acqf(X)
            self.assertTrue(torch.equal(acqf.sampler.base_samples, bs))

            # basic test, qmc, no resample
            sampler = SobolQMCNormalSampler(num_samples=2)
            acqf = qUpperConfidenceBound(model=mm, beta=0.5, sampler=sampler)
            res = acqf(X)
            self.assertEqual(res.item(), 0.0)
            self.assertEqual(acqf.sampler.base_samples.shape, torch.Size([2, 1, 1, 1]))
            bs = acqf.sampler.base_samples.clone()
            acqf(X)
            self.assertTrue(torch.equal(acqf.sampler.base_samples, bs))

            # basic test, qmc, resample
            sampler = SobolQMCNormalSampler(num_samples=2, resample=True)
            acqf = qUpperConfidenceBound(model=mm, beta=0.5, sampler=sampler)
            res = acqf(X)
            self.assertEqual(res.item(), 0.0)
            self.assertEqual(acqf.sampler.base_samples.shape, torch.Size([2, 1, 1, 1]))
            bs = acqf.sampler.base_samples.clone()
            acqf(X)
            self.assertFalse(torch.equal(acqf.sampler.base_samples, bs))

    def test_q_upper_confidence_bound_cuda(self):
        if torch.cuda.is_available():
            self.test_q_upper_confidence_bound(cuda=True)

    def test_q_upper_confidence_bound_batch(self, cuda=False):
        # TODO: T41739913 Implement tests for all MCAcquisitionFunctions
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            samples = torch.zeros(2, 2, 1, device=device, dtype=dtype)
            samples[0, 0, 0] = 1.0
            mm = MockModel(MockPosterior(samples=samples))
            # X is a dummy and unused b/c of mocking
            X = torch.zeros(1, 1, 1, device=device, dtype=dtype)

            # test batch mode
            sampler = IIDNormalSampler(num_samples=2)
            acqf = qUpperConfidenceBound(model=mm, beta=0.5, sampler=sampler)
            res = acqf(X)
            self.assertEqual(res[0].item(), 1.0)
            self.assertEqual(res[1].item(), 0.0)

            # test batch mode, no resample
            sampler = IIDNormalSampler(num_samples=2, seed=12345)
            acqf = qUpperConfidenceBound(model=mm, beta=0.5, sampler=sampler)
            res = acqf(X)  # 1-dim batch
            self.assertEqual(res[0].item(), 1.0)
            self.assertEqual(res[1].item(), 0.0)
            self.assertEqual(acqf.sampler.base_samples.shape, torch.Size([2, 1, 2, 1]))
            bs = acqf.sampler.base_samples.clone()
            acqf(X)
            self.assertTrue(torch.equal(acqf.sampler.base_samples, bs))
            res = acqf(X.expand(2, 1, 1))  # 2-dim batch
            self.assertEqual(res[0].item(), 1.0)
            self.assertEqual(res[1].item(), 0.0)
            # the base samples should have the batch dim collapsed
            self.assertEqual(acqf.sampler.base_samples.shape, torch.Size([2, 1, 2, 1]))
            bs = acqf.sampler.base_samples.clone()
            acqf(X.expand(2, 1, 1))
            self.assertTrue(torch.equal(acqf.sampler.base_samples, bs))

            # test batch mode, qmc, no resample
            sampler = SobolQMCNormalSampler(num_samples=2)
            acqf = qUpperConfidenceBound(model=mm, beta=0.5, sampler=sampler)
            res = acqf(X)
            self.assertEqual(res[0].item(), 1.0)
            self.assertEqual(res[1].item(), 0.0)
            self.assertEqual(acqf.sampler.base_samples.shape, torch.Size([2, 1, 2, 1]))
            bs = acqf.sampler.base_samples.clone()
            acqf(X)
            self.assertTrue(torch.equal(acqf.sampler.base_samples, bs))

            # test batch mode, qmc, resample
            sampler = SobolQMCNormalSampler(num_samples=2, resample=True)
            acqf = qUpperConfidenceBound(model=mm, beta=0.5, sampler=sampler)
            res = acqf(X)  # 1-dim batch
            self.assertEqual(res[0].item(), 1.0)
            self.assertEqual(res[1].item(), 0.0)
            self.assertEqual(acqf.sampler.base_samples.shape, torch.Size([2, 1, 2, 1]))
            bs = acqf.sampler.base_samples.clone()
            acqf(X)
            self.assertFalse(torch.equal(acqf.sampler.base_samples, bs))
            res = acqf(X.expand(2, 1, 1))  # 2-dim batch
            self.assertEqual(res[0].item(), 1.0)
            self.assertEqual(res[1].item(), 0.0)
            # the base samples should have the batch dim collapsed
            self.assertEqual(acqf.sampler.base_samples.shape, torch.Size([2, 1, 2, 1]))
            bs = acqf.sampler.base_samples.clone()
            acqf(X.expand(2, 1, 1))
            self.assertFalse(torch.equal(acqf.sampler.base_samples, bs))

    def test_q_upper_confidence_bound_batch_cuda(self):
        if torch.cuda.is_available():
            self.test_q_upper_confidence_bound_batch(cuda=True)

    # TODO: Test different objectives (incl. constraints)
