#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

import warnings
from warnings import catch_warnings, simplefilter

import torch
from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from botorch_community.acquisition.rei import (
    LogRegionalExpectedImprovement,
    qRegionalExpectedImprovement,
)
from botorch.acquisition.objective import (
    ScalarizedPosteriorTransform,
)
from botorch.exceptions import BotorchWarning, UnsupportedError
from botorch.exceptions.warnings import NumericsWarning
from botorch.posteriors import GPyTorchPosterior
from botorch.sampling.normal import IIDNormalSampler, SobolQMCNormalSampler
from botorch.utils.testing import BotorchTestCase, MockModel, MockPosterior
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal


class TestLogRegionalExpectedImprovement(BotorchTestCase):
    def test_log_regional_expected_improvement(self):
        for dtype in (torch.float, torch.double):
            with catch_warnings():
                simplefilter("ignore", category=NumericsWarning)
                self._test_log_regional_expected_improvement(dtype=dtype)

    def _test_log_regional_expected_improvement(self, dtype: torch.dtype) -> None:
        mean = torch.tensor([[-0.5]], device=self.device, dtype=dtype)
        variance = torch.ones(1, 1, device=self.device, dtype=dtype)
        mm = MockModel(MockPosterior(mean=mean, variance=variance))

        # basic test
        X_dev = torch.empty(1, 1, device=self.device, dtype=dtype)
        module = LogRegionalExpectedImprovement(model=mm, best_f=0.0, X_dev=X_dev)
        X = torch.empty(1, 1, device=self.device, dtype=dtype)
        log_rei = module(X)
        rei_expected = torch.tensor([0.19780], device=self.device, dtype=dtype)
        self.assertAllClose(log_rei, rei_expected.log(), atol=1e-4)

        # test maximize
        X_dev = torch.empty(1, 1, device=self.device, dtype=dtype)
        module = LogRegionalExpectedImprovement(
            model=mm, best_f=0.0, X_dev=X_dev, maximize=False
        )
        X = torch.empty(1, 1, device=self.device, dtype=dtype)
        log_rei = module(X)
        rei_expected = torch.tensor([0.6978], device=self.device, dtype=dtype)
        self.assertAllClose(log_rei, rei_expected.log(), atol=1e-4)

        with self.assertRaises(UnsupportedError):
            module.set_X_pending(None)

        # test posterior transform (single-output)
        mean = torch.tensor([0.5], device=self.device, dtype=dtype)
        covar = torch.tensor([[0.16]], device=self.device, dtype=dtype)
        mvn = MultivariateNormal(mean, covar)
        p = GPyTorchPosterior(mvn)
        mm = MockModel(p)
        weights = torch.tensor([0.5], device=self.device, dtype=dtype)
        transform = ScalarizedPosteriorTransform(weights)
        X_dev = torch.rand(1, 2, device=self.device, dtype=dtype)
        log_rei = LogRegionalExpectedImprovement(
            model=mm, best_f=0.0, X_dev=X_dev, posterior_transform=transform
        )
        X = torch.rand(1, 2, device=self.device, dtype=dtype)
        rei_expected = torch.tensor([0.2601], device=self.device, dtype=dtype)
        self.assertAllClose(log_rei(X), rei_expected.log(), atol=1e-4)

        # test posterior transform (multi-output)
        mean = torch.tensor([[-0.25, 0.5]], device=self.device, dtype=dtype)
        covar = torch.tensor(
            [[[0.5, 0.125], [0.125, 0.5]]], device=self.device, dtype=dtype
        )
        mvn = MultitaskMultivariateNormal(mean, covar)
        p = GPyTorchPosterior(mvn)
        mm = MockModel(p)
        weights = torch.tensor([2.0, 1.0], device=self.device, dtype=dtype)
        transform = ScalarizedPosteriorTransform(weights)
        X_dev = torch.rand(1, 2, device=self.device, dtype=dtype)
        log_rei = LogRegionalExpectedImprovement(
            model=mm, best_f=0.0, X_dev=X_dev, posterior_transform=transform
        )
        X = torch.rand(1, 2, device=self.device, dtype=dtype)
        rei_expected = torch.tensor([0.6910], device=self.device, dtype=dtype)
        self.assertAllClose(log_rei(X), rei_expected.log(), atol=1e-4)


class TestQRegionalExpectedImprovement(BotorchTestCase):
    def test_q_regional_expected_improvement(self):
        for dtype in (torch.float, torch.double):
            with self.subTest(dtype=dtype):
                with catch_warnings():
                    simplefilter("ignore", category=NumericsWarning)
                    self._test_q_regional_expected_improvement(dtype)

    def _test_q_regional_expected_improvement(self, dtype: torch.dtype) -> None:
        tkwargs: dict[str, Any] = {"device": self.device, "dtype": dtype}
        # the event shape is `b x q x t` = 1 x 1 x 1
        samples = torch.zeros(1, 1, 1, **tkwargs)
        mm = MockModel(MockPosterior(samples=samples))
        # X is `q x d` = 1 x 1. X is a dummy and unused b/c of mocking
        X = torch.zeros(1, 1, **tkwargs)
        X_dev = torch.zeros(1, 1, **tkwargs)

        # basic test
        sampler = IIDNormalSampler(sample_shape=torch.Size([2]))
        acqf = qRegionalExpectedImprovement(
            model=mm, best_f=0, X_dev=X_dev, sampler=sampler
        )
        # test initialization
        for k in ["objective", "sampler"]:
            self.assertIn(k, acqf._modules)

        res = acqf(X)
        self.assertEqual(res.item(), 0.0)

        # test shifting best_f value
        acqf = qRegionalExpectedImprovement(
            model=mm, best_f=-1, X_dev=X_dev, sampler=sampler
        )
        res = acqf(X)
        self.assertEqual(res.item(), 1.0)

        # basic test, no resample
        sampler = IIDNormalSampler(sample_shape=torch.Size([2]), seed=12345)
        acqf = qRegionalExpectedImprovement(
            model=mm, best_f=0, X_dev=X_dev, sampler=sampler
        )
        res = acqf(X)
        self.assertEqual(res.item(), 0.0)
        self.assertEqual(acqf.sampler.base_samples.shape, torch.Size([2, 1, 1, 1]))
        bs = acqf.sampler.base_samples.clone()
        res = acqf(X)
        self.assertTrue(torch.equal(acqf.sampler.base_samples, bs))

        # basic test, qmc
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([2]))
        acqf = qRegionalExpectedImprovement(
            model=mm, best_f=0, X_dev=X_dev, sampler=sampler
        )
        res = acqf(X)
        self.assertEqual(res.item(), 0.0)
        self.assertEqual(acqf.sampler.base_samples.shape, torch.Size([2, 1, 1, 1]))
        bs = acqf.sampler.base_samples.clone()
        acqf(X)
        self.assertTrue(torch.equal(acqf.sampler.base_samples, bs))

        # basic test for X_pending and warning
        acqf.set_X_pending()
        self.assertIsNone(acqf.X_pending)
        acqf.set_X_pending(None)
        self.assertIsNone(acqf.X_pending)
        acqf.set_X_pending(X)
        self.assertEqual(acqf.X_pending, X)
        mm._posterior._samples = torch.zeros(1, 2, 1, **tkwargs)
        res = acqf(X)
        X2 = torch.zeros(1, 1, 1, **tkwargs, requires_grad=True)
        with warnings.catch_warnings(record=True) as ws:
            acqf.set_X_pending(X2)
        self.assertEqual(acqf.X_pending, X2)
        self.assertEqual(sum(issubclass(w.category, BotorchWarning) for w in ws), 1)

    def test_q_regional_expected_improvement_batch(self):
        for dtype in (torch.float, torch.double):
            with self.subTest(dtype=dtype):
                with catch_warnings():
                    simplefilter("ignore", category=NumericsWarning)
                    self._test_q_regional_expected_improvement_batch(dtype)

    def _test_q_regional_expected_improvement_batch(self, dtype: torch.dtype) -> None:
        # the event shape is `b x q x t` = 2 x 2 x 1
        samples = torch.zeros(2, 2, 1, device=self.device, dtype=dtype)
        samples[0, 0, 0] = 1.0
        mm = MockModel(MockPosterior(samples=samples))

        # X is a dummy and unused b/c of mocking
        X = torch.zeros(2, 2, 1, device=self.device, dtype=dtype)
        X_dev = torch.zeros(1, 1, device=self.device, dtype=dtype)

        # test batch mode
        sampler = IIDNormalSampler(sample_shape=torch.Size([2]))
        acqf = qRegionalExpectedImprovement(
            model=mm, best_f=0, X_dev=X_dev, sampler=sampler
        )
        res = acqf(X)
        self.assertEqual(res[0].item(), 1.0)
        self.assertEqual(res[1].item(), 0.0)

        # test batch model, batched best_f values
        sampler = IIDNormalSampler(sample_shape=torch.Size([3]))
        acqf = qRegionalExpectedImprovement(
            model=mm, best_f=torch.Tensor([0, 0]), X_dev=X_dev, sampler=sampler
        )
        res = acqf(X)
        self.assertEqual(res[0].item(), 1.0)
        self.assertEqual(res[1].item(), 0.0)

        # test shifting best_f value
        acqf = qRegionalExpectedImprovement(
            model=mm, best_f=-1, X_dev=X_dev, sampler=sampler
        )
        res = acqf(X)
        self.assertEqual(res[0].item(), 2.0)
        self.assertEqual(res[1].item(), 1.0)

        # test batch mode
        sampler = IIDNormalSampler(sample_shape=torch.Size([2]), seed=12345)
        acqf = qRegionalExpectedImprovement(
            model=mm, best_f=0, X_dev=X_dev, sampler=sampler
        )
        res = acqf(X)  # 1-dim batch
        self.assertEqual(res[0].item(), 1.0)
        self.assertEqual(res[1].item(), 0.0)
        self.assertEqual(acqf.sampler.base_samples.shape, torch.Size([2, 1, 2, 1]))
        bs = acqf.sampler.base_samples.clone()
        acqf(X)
        self.assertTrue(torch.equal(acqf.sampler.base_samples, bs))
        res = acqf(X.expand(2, 2, 1))  # 2-dim batch
        self.assertEqual(res[0].item(), 1.0)
        self.assertEqual(res[1].item(), 0.0)
        # the base samples should have the batch dim collapsed
        self.assertEqual(acqf.sampler.base_samples.shape, torch.Size([2, 1, 2, 1]))
        bs = acqf.sampler.base_samples.clone()
        acqf(X.expand(2, 2, 1))
        self.assertTrue(torch.equal(acqf.sampler.base_samples, bs))

        # test batch mode, qmc
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([2]))
        acqf = qRegionalExpectedImprovement(
            model=mm, best_f=0, X_dev=X_dev, sampler=sampler
        )
        res = acqf(X)
        self.assertEqual(res[0].item(), 1.0)
        self.assertEqual(res[1].item(), 0.0)
        self.assertEqual(acqf.sampler.base_samples.shape, torch.Size([2, 1, 2, 1]))
        bs = acqf.sampler.base_samples.clone()
        acqf(X)
        self.assertTrue(torch.equal(acqf.sampler.base_samples, bs))
