#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from unittest import mock

import torch
from botorch import settings
from botorch.acquisition.monte_carlo import (
    MCAcquisitionFunction,
    qExpectedImprovement,
    qNoisyExpectedImprovement,
    qProbabilityOfImprovement,
    qSimpleRegret,
    qUpperConfidenceBound,
)
from botorch.acquisition.objective import ScalarizedObjective
from botorch.exceptions import BotorchWarning, UnsupportedError
from botorch.sampling.samplers import IIDNormalSampler, SobolQMCNormalSampler
from botorch.utils.testing import BotorchTestCase, MockModel, MockPosterior


class DummyMCAcquisitionFunction(MCAcquisitionFunction):
    def forward(self, X):
        pass


class TestMCAcquisitionFunction(BotorchTestCase):
    def test_abstract_raises(self):
        with self.assertRaises(TypeError):
            MCAcquisitionFunction()
        # raise if model is multi-output, but no objective is given
        no = "botorch.utils.testing.MockModel.num_outputs"
        with mock.patch(no, new_callable=mock.PropertyMock) as mock_num_outputs:
            mock_num_outputs.return_value = 2
            mm = MockModel(MockPosterior())
            with self.assertRaises(UnsupportedError):
                DummyMCAcquisitionFunction(model=mm)


class TestQExpectedImprovement(BotorchTestCase):
    def test_q_expected_improvement(self):
        for dtype in (torch.float, torch.double):
            # the event shape is `b x q x t` = 1 x 1 x 1
            samples = torch.zeros(1, 1, 1, device=self.device, dtype=dtype)
            mm = MockModel(MockPosterior(samples=samples))
            # X is `q x d` = 1 x 1. X is a dummy and unused b/c of mocking
            X = torch.zeros(1, 1, device=self.device, dtype=dtype)

            # basic test
            sampler = IIDNormalSampler(num_samples=2)
            acqf = qExpectedImprovement(model=mm, best_f=0, sampler=sampler)
            res = acqf(X)
            self.assertEqual(res.item(), 0.0)

            # test shifting best_f value
            acqf = qExpectedImprovement(model=mm, best_f=-1, sampler=sampler)
            res = acqf(X)
            self.assertEqual(res.item(), 1.0)

            # TODO: Test batched best_f, batched model, batched evaluation

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

            # basic test for X_pending and warning
            acqf.set_X_pending()
            self.assertIsNone(acqf.X_pending)
            acqf.set_X_pending(None)
            self.assertIsNone(acqf.X_pending)
            acqf.set_X_pending(X)
            self.assertEqual(acqf.X_pending, X)
            res = acqf(X)
            X2 = torch.zeros(
                1, 1, 1, device=self.device, dtype=dtype, requires_grad=True
            )
            with warnings.catch_warnings(record=True) as ws, settings.debug(True):
                acqf.set_X_pending(X2)
                self.assertEqual(acqf.X_pending, X2)
                self.assertEqual(len(ws), 1)
                self.assertTrue(issubclass(ws[-1].category, BotorchWarning))

        # test bad objective type
        obj = ScalarizedObjective(
            weights=torch.rand(2, device=self.device, dtype=dtype)
        )
        with self.assertRaises(UnsupportedError):
            qExpectedImprovement(model=mm, best_f=0, sampler=sampler, objective=obj)

    def test_q_expected_improvement_batch(self):
        for dtype in (torch.float, torch.double):
            # the event shape is `b x q x t` = 2 x 2 x 1
            samples = torch.zeros(2, 2, 1, device=self.device, dtype=dtype)
            samples[0, 0, 0] = 1.0
            mm = MockModel(MockPosterior(samples=samples))

            # X is a dummy and unused b/c of mocking
            X = torch.zeros(1, 1, 1, device=self.device, dtype=dtype)

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

    # TODO: Test different objectives (incl. constraints)


class TestQNoisyExpectedImprovement(BotorchTestCase):
    def test_q_noisy_expected_improvement(self):
        for dtype in (torch.float, torch.double):
            # the event shape is `b x q x t` = 1 x 2 x 1
            samples_noisy = torch.tensor([1.0, 0.0], device=self.device, dtype=dtype)
            samples_noisy = samples_noisy.view(1, 2, 1)
            # X_baseline is `q' x d` = 1 x 1
            X_baseline = torch.zeros(1, 1, device=self.device, dtype=dtype)
            mm_noisy = MockModel(MockPosterior(samples=samples_noisy))
            # X is `q x d` = 1 x 1
            X = torch.zeros(1, 1, device=self.device, dtype=dtype)

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

            # basic test for X_pending and warning
            sampler = SobolQMCNormalSampler(num_samples=2)
            samples_noisy_pending = torch.tensor(
                [1.0, 0.0, 0.0], device=self.device, dtype=dtype
            )
            samples_noisy_pending = samples_noisy_pending.view(1, 3, 1)
            mm_noisy_pending = MockModel(MockPosterior(samples=samples_noisy_pending))
            acqf = qNoisyExpectedImprovement(
                model=mm_noisy_pending, X_baseline=X_baseline, sampler=sampler
            )
            acqf.set_X_pending()
            self.assertIsNone(acqf.X_pending)
            acqf.set_X_pending(None)
            self.assertIsNone(acqf.X_pending)
            acqf.set_X_pending(X)
            self.assertEqual(acqf.X_pending, X)
            res = acqf(X)
            X2 = torch.zeros(
                1, 1, 1, device=self.device, dtype=dtype, requires_grad=True
            )
            with warnings.catch_warnings(record=True) as ws, settings.debug(True):
                acqf.set_X_pending(X2)
                self.assertEqual(acqf.X_pending, X2)
                self.assertEqual(len(ws), 1)
                self.assertTrue(issubclass(ws[-1].category, BotorchWarning))

    def test_q_noisy_expected_improvement_batch(self):
        for dtype in (torch.float, torch.double):
            # the event shape is `b x q x t` = 2 x 3 x 1
            samples_noisy = torch.zeros(2, 3, 1, device=self.device, dtype=dtype)
            samples_noisy[0, 0, 0] = 1.0
            mm_noisy = MockModel(MockPosterior(samples=samples_noisy))
            # X is `q x d` = 1 x 1
            X = torch.zeros(1, 1, 1, device=self.device, dtype=dtype)
            X_baseline = torch.zeros(1, 1, device=self.device, dtype=dtype)

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

    def test_prune_baseline(self):
        no = "botorch.utils.testing.MockModel.num_outputs"
        prune = "botorch.acquisition.monte_carlo.prune_inferior_points"
        for dtype in (torch.float, torch.double):
            X_baseline = torch.zeros(1, 1, device=self.device, dtype=dtype)
            X_pruned = torch.rand(1, 1, device=self.device, dtype=dtype)
            with mock.patch(no, new_callable=mock.PropertyMock) as mock_num_outputs:
                mock_num_outputs.return_value = 1
                mm = MockModel(mock.Mock())
                with mock.patch(prune, return_value=X_pruned) as mock_prune:
                    acqf = qNoisyExpectedImprovement(
                        model=mm, X_baseline=X_baseline, prune_baseline=True
                    )
                mock_prune.assert_called_once()
                self.assertTrue(torch.equal(acqf.X_baseline, X_pruned))

    # TODO: Test different objectives (incl. constraints)


class TestQProbabilityOfImprovement(BotorchTestCase):
    def test_q_probability_of_improvement(self):
        for dtype in (torch.float, torch.double):
            # the event shape is `b x q x t` = 1 x 1 x 1
            samples = torch.zeros(1, 1, 1, device=self.device, dtype=dtype)
            mm = MockModel(MockPosterior(samples=samples))
            # X is `q x d` = 1 x 1. X is a dummy and unused b/c of mocking
            X = torch.zeros(1, 1, device=self.device, dtype=dtype)

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

            # basic test for X_pending and warning
            acqf.set_X_pending()
            self.assertIsNone(acqf.X_pending)
            acqf.set_X_pending(None)
            self.assertIsNone(acqf.X_pending)
            acqf.set_X_pending(X)
            self.assertEqual(acqf.X_pending, X)
            res = acqf(X)
            X2 = torch.zeros(
                1, 1, 1, device=self.device, dtype=dtype, requires_grad=True
            )
            with warnings.catch_warnings(record=True) as ws, settings.debug(True):
                acqf.set_X_pending(X2)
                self.assertEqual(acqf.X_pending, X2)
                self.assertEqual(len(ws), 1)
                self.assertTrue(issubclass(ws[-1].category, BotorchWarning))

    def test_q_probability_of_improvement_batch(self):
        # the event shape is `b x q x t` = 2 x 2 x 1
        for dtype in (torch.float, torch.double):
            samples = torch.zeros(2, 2, 1, device=self.device, dtype=dtype)
            samples[0, 0, 0] = 1.0
            mm = MockModel(MockPosterior(samples=samples))

            # X is a dummy and unused b/c of mocking
            X = torch.zeros(1, 1, 1, device=self.device, dtype=dtype)

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

    # TODO: Test different objectives (incl. constraints)


class TestQSimpleRegret(BotorchTestCase):
    def test_q_simple_regret(self):
        for dtype in (torch.float, torch.double):
            # the event shape is `b x q x t` = 1 x 1 x 1
            samples = torch.zeros(1, 1, 1, device=self.device, dtype=dtype)
            mm = MockModel(MockPosterior(samples=samples))
            # X is `q x d` = 1 x 1. X is a dummy and unused b/c of mocking
            X = torch.zeros(1, 1, device=self.device, dtype=dtype)

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

            # basic test for X_pending and warning
            acqf.set_X_pending()
            self.assertIsNone(acqf.X_pending)
            acqf.set_X_pending(None)
            self.assertIsNone(acqf.X_pending)
            acqf.set_X_pending(X)
            self.assertEqual(acqf.X_pending, X)
            res = acqf(X)
            X2 = torch.zeros(
                1, 1, 1, device=self.device, dtype=dtype, requires_grad=True
            )
            with warnings.catch_warnings(record=True) as ws, settings.debug(True):
                acqf.set_X_pending(X2)
                self.assertEqual(acqf.X_pending, X2)
                self.assertEqual(len(ws), 1)
                self.assertTrue(issubclass(ws[-1].category, BotorchWarning))

    def test_q_simple_regret_batch(self):
        # the event shape is `b x q x t` = 2 x 2 x 1
        for dtype in (torch.float, torch.double):
            samples = torch.zeros(2, 2, 1, device=self.device, dtype=dtype)
            samples[0, 0, 0] = 1.0
            mm = MockModel(MockPosterior(samples=samples))
            # X is a dummy and unused b/c of mocking
            X = torch.zeros(1, 1, 1, device=self.device, dtype=dtype)

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

    # TODO: Test different objectives (incl. constraints)


class TestQUpperConfidenceBound(BotorchTestCase):
    def test_q_upper_confidence_bound(self):
        for dtype in (torch.float, torch.double):
            # the event shape is `b x q x t` = 1 x 1 x 1
            samples = torch.zeros(1, 1, 1, device=self.device, dtype=dtype)
            mm = MockModel(MockPosterior(samples=samples))
            # X is `q x d` = 1 x 1. X is a dummy and unused b/c of mocking
            X = torch.zeros(1, 1, device=self.device, dtype=dtype)

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

            # basic test for X_pending and warning
            acqf.set_X_pending()
            self.assertIsNone(acqf.X_pending)
            acqf.set_X_pending(None)
            self.assertIsNone(acqf.X_pending)
            acqf.set_X_pending(X)
            self.assertEqual(acqf.X_pending, X)
            res = acqf(X)
            X2 = torch.zeros(
                1, 1, 1, device=self.device, dtype=dtype, requires_grad=True
            )
            with warnings.catch_warnings(record=True) as ws, settings.debug(True):
                acqf.set_X_pending(X2)
                self.assertEqual(acqf.X_pending, X2)
                self.assertEqual(len(ws), 1)
                self.assertTrue(issubclass(ws[-1].category, BotorchWarning))

    def test_q_upper_confidence_bound_batch(self):
        # TODO: T41739913 Implement tests for all MCAcquisitionFunctions
        for dtype in (torch.float, torch.double):
            samples = torch.zeros(2, 2, 1, device=self.device, dtype=dtype)
            samples[0, 0, 0] = 1.0
            mm = MockModel(MockPosterior(samples=samples))
            # X is a dummy and unused b/c of mocking
            X = torch.zeros(1, 1, 1, device=self.device, dtype=dtype)

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

            # basic test for X_pending and warning
            acqf.set_X_pending()
            self.assertIsNone(acqf.X_pending)
            acqf.set_X_pending(None)
            self.assertIsNone(acqf.X_pending)
            acqf.set_X_pending(X)
            self.assertEqual(acqf.X_pending, X)
            res = acqf(X)
            X2 = torch.zeros(
                1, 1, 1, device=self.device, dtype=dtype, requires_grad=True
            )
            with warnings.catch_warnings(record=True) as ws, settings.debug(True):
                acqf.set_X_pending(X2)
                self.assertEqual(acqf.X_pending, X2)
                self.assertEqual(len(ws), 1)
                self.assertTrue(issubclass(ws[-1].category, BotorchWarning))

    # TODO: Test different objectives (incl. constraints)
