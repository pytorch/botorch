#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from unittest import mock

import torch
from botorch import settings
from botorch.acquisition import (
    LogImprovementMCAcquisitionFunction,
    qLogExpectedImprovement,
)
from botorch.acquisition.input_constructors import ACQF_INPUT_CONSTRUCTOR_REGISTRY
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.acquisition.objective import (
    ConstrainedMCObjective,
    IdentityMCObjective,
    PosteriorTransform,
)
from botorch.exceptions import BotorchWarning, UnsupportedError
from botorch.exceptions.errors import BotorchError
from botorch.sampling.normal import IIDNormalSampler, SobolQMCNormalSampler
from botorch.utils.testing import BotorchTestCase, MockModel, MockPosterior
from torch import Tensor


def infeasible_con(samples: Tensor) -> Tensor:
    return torch.ones_like(samples[..., 0])


def feasible_con(samples: Tensor) -> Tensor:
    return -torch.ones_like(samples[..., 0])


class DummyLogImprovementAcquisitionFunction(LogImprovementMCAcquisitionFunction):
    def _sample_forward(self, X):
        pass


class DummyNonScalarizingPosteriorTransform(PosteriorTransform):
    scalarize = False

    def evaluate(self, Y):
        pass  # pragma: no cover

    def forward(self, posterior):
        pass  # pragma: no cover


class TestLogImprovementAcquisitionFunction(BotorchTestCase):
    def test_abstract_raises(self):
        with self.assertRaises(TypeError):
            LogImprovementMCAcquisitionFunction()
        # raise if model is multi-output, but no outcome transform or objective
        # are given
        no = "botorch.utils.testing.MockModel.num_outputs"
        with mock.patch(no, new_callable=mock.PropertyMock) as mock_num_outputs:
            mock_num_outputs.return_value = 2
            mm = MockModel(MockPosterior())
            with self.assertRaises(UnsupportedError):
                DummyLogImprovementAcquisitionFunction(model=mm)
        # raise if model is multi-output, but outcome transform does not
        # scalarize and no objetive is given
        with mock.patch(no, new_callable=mock.PropertyMock) as mock_num_outputs:
            mock_num_outputs.return_value = 2
            mm = MockModel(MockPosterior())
            ptf = DummyNonScalarizingPosteriorTransform()
            with self.assertRaises(UnsupportedError):
                DummyLogImprovementAcquisitionFunction(
                    model=mm, posterior_transform=ptf
                )

        mm = MockModel(MockPosterior())
        objective = ConstrainedMCObjective(
            IdentityMCObjective(),
            constraints=[lambda samples: torch.zeros_like(samples[..., 0])],
        )
        with self.assertRaisesRegex(
            BotorchError,
            "Log-Improvement should not be used with `ConstrainedMCObjective`.",
        ):
            DummyLogImprovementAcquisitionFunction(model=mm, objective=objective)


class TestQLogExpectedImprovement(BotorchTestCase):
    def test_q_log_expected_improvement(self):
        self.assertIn(qLogExpectedImprovement, ACQF_INPUT_CONSTRUCTOR_REGISTRY.keys())
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            # the event shape is `b x q x t` = 1 x 1 x 1
            samples = torch.zeros(1, 1, 1, **tkwargs)
            mm = MockModel(MockPosterior(samples=samples))
            # X is `q x d` = 1 x 1. X is a dummy and unused b/c of mocking
            X = torch.zeros(1, 1, **tkwargs)

            # basic test
            sampler = IIDNormalSampler(sample_shape=torch.Size([2]))
            acqf = qExpectedImprovement(model=mm, best_f=0, sampler=sampler)
            log_acqf = qLogExpectedImprovement(model=mm, best_f=0, sampler=sampler)
            self.assertFalse(acqf._fat)  # different default behavior
            self.assertTrue(log_acqf._fat)
            # test initialization
            for k in ["objective", "sampler"]:
                self.assertIn(k, acqf._modules)
                self.assertIn(k, log_acqf._modules)

            res = acqf(X).item()
            self.assertEqual(res, 0.0)
            exp_log_res = log_acqf(X).exp().item()
            # Due to the smooth approximation, the value at zero should be close to, but
            # not exactly zero, and upper-bounded by the tau hyperparameter.
            self.assertTrue(0 < exp_log_res)
            self.assertTrue(exp_log_res <= log_acqf.tau_relu)

            # test shifting best_f value downward to see non-zero improvement
            best_f = -1
            acqf = qExpectedImprovement(model=mm, best_f=best_f, sampler=sampler)
            log_acqf = qLogExpectedImprovement(model=mm, best_f=best_f, sampler=sampler)
            res, exp_log_res = acqf(X), log_acqf(X).exp()
            expected_val = -best_f

            self.assertEqual(res.dtype, dtype)
            self.assertEqual(res.device.type, self.device.type)
            self.assertEqual(res.item(), expected_val)
            # Further away from zero, the value is numerically indistinguishable with
            # single precision arithmetic.
            self.assertTrue(expected_val <= exp_log_res.item())
            self.assertTrue(exp_log_res.item() <= expected_val + log_acqf.tau_relu)

            # test shifting best_f value upward to see advantage of LogEI
            best_f = 1
            acqf = qExpectedImprovement(model=mm, best_f=best_f, sampler=sampler)
            log_acqf = qLogExpectedImprovement(model=mm, best_f=best_f, sampler=sampler)
            res, log_res = acqf(X), log_acqf(X)
            exp_log_res = log_res.exp()
            expected_val = 0
            self.assertEqual(res.item(), expected_val)
            self.assertTrue(expected_val <= exp_log_res.item())
            self.assertTrue(exp_log_res.item() <= expected_val + log_acqf.tau_relu)
            # However, the log value is large and negative with non-vanishing gradients
            self.assertGreater(-1, log_res.item())
            self.assertGreater(log_res.item(), -100)

            # NOTE: The following tests are adapted from the qEI tests.
            # basic test, no resample
            sampler = IIDNormalSampler(sample_shape=torch.Size([2]), seed=12345)
            acqf = qLogExpectedImprovement(model=mm, best_f=0, sampler=sampler)
            res = acqf(X)
            self.assertTrue(0 < res.exp().item())
            self.assertTrue(res.exp().item() < acqf.tau_relu)
            self.assertEqual(acqf.sampler.base_samples.shape, torch.Size([2, 1, 1, 1]))
            bs = acqf.sampler.base_samples.clone()
            res = acqf(X)
            self.assertTrue(torch.equal(acqf.sampler.base_samples, bs))

            # basic test, qmc
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([2]))
            acqf = qLogExpectedImprovement(model=mm, best_f=0, sampler=sampler)
            res = acqf(X)
            self.assertTrue(0 < res.exp().item())
            self.assertTrue(res.exp().item() < acqf.tau_relu)
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
            with warnings.catch_warnings(record=True) as ws, settings.debug(True):
                acqf.set_X_pending(X2)
                self.assertEqual(acqf.X_pending, X2)
                self.assertEqual(
                    sum(issubclass(w.category, BotorchWarning) for w in ws), 1
                )

            # testing with illegal taus
            with self.assertRaisesRegex(ValueError, "tau_max is not a scalar:"):
                qLogExpectedImprovement(
                    model=mm, best_f=0, tau_max=torch.tensor([1, 2])
                )
            with self.assertRaisesRegex(ValueError, "tau_relu is non-positive:"):
                qLogExpectedImprovement(model=mm, best_f=0, tau_relu=-2)

    def test_q_log_expected_improvement_batch(self):
        for dtype in (torch.float, torch.double):
            # the event shape is `b x q x t` = 2 x 2 x 1
            samples = torch.zeros(2, 2, 1, device=self.device, dtype=dtype)
            samples[0, 0, 0] = 1.0
            mm = MockModel(MockPosterior(samples=samples))

            # X is a dummy and unused b/c of mocking
            X = torch.zeros(2, 2, 1, device=self.device, dtype=dtype)

            # test batch mode
            sampler = IIDNormalSampler(sample_shape=torch.Size([2]))
            acqf = qLogExpectedImprovement(model=mm, best_f=0, sampler=sampler)
            exp_log_res = acqf(X).exp()
            # with no approximations (qEI): self.assertEqual(res[0].item(), 1.0)
            # in the batch case, the values get adjusted toward
            self.assertEqual(exp_log_res.dtype, dtype)
            self.assertEqual(exp_log_res.device.type, self.device.type)
            self.assertTrue(1.0 <= exp_log_res[0].item())
            self.assertTrue(exp_log_res[0].item() <= 1.0 + acqf.tau_relu)
            # self.assertAllClose(exp_log_res[0], torch.ones_like(exp_log_res[0]), )

            # with no approximations (qEI): self.assertEqual(res[1].item(), 0.0)
            self.assertTrue(0 < exp_log_res[1].item())
            self.assertTrue(exp_log_res[1].item() <= acqf.tau_relu)

            # test batch model, batched best_f values
            sampler = IIDNormalSampler(sample_shape=torch.Size([3]))
            acqf = qLogExpectedImprovement(
                model=mm, best_f=torch.Tensor([0, 0]), sampler=sampler
            )
            exp_log_res = acqf(X).exp()
            # with no approximations (qEI): self.assertEqual(res[0].item(), 1.0)
            self.assertTrue(1.0 <= exp_log_res[0].item())
            self.assertTrue(exp_log_res[0].item() <= 1.0 + acqf.tau_relu)
            # with no approximations (qEI): self.assertEqual(res[1].item(), 0.0)
            self.assertTrue(0 < exp_log_res[1].item())
            self.assertTrue(exp_log_res[1].item() <= acqf.tau_relu)

            # test shifting best_f value
            acqf = qLogExpectedImprovement(model=mm, best_f=-1, sampler=sampler)
            exp_log_res = acqf(X).exp()
            # with no approximations (qEI): self.assertEqual(res[0].item(), 2.0)
            # TODO: figure out numerically stable tests and principled tolerances
            # With q > 1, maximum value can get moved down due to L_q-norm approximation
            # of the maximum over the q-batch.
            safe_upper_lower_bound = 1.999
            self.assertTrue(safe_upper_lower_bound <= exp_log_res[0].item())
            self.assertTrue(exp_log_res[0].item() <= 2.0 + acqf.tau_relu + acqf.tau_max)
            # with no approximations (qEI): self.assertEqual(res[1].item(), 1.0)
            self.assertTrue(1.0 <= exp_log_res[1].item())
            # ocurring ~tau_max error when all candidates in a q-batch have the
            # acquisition value
            self.assertTrue(exp_log_res[1].item() <= 1.0 + acqf.tau_relu + acqf.tau_max)

            # test batch mode
            sampler = IIDNormalSampler(sample_shape=torch.Size([2]), seed=12345)
            acqf = qLogExpectedImprovement(model=mm, best_f=0, sampler=sampler)
            # res = acqf(X)  # 1-dim batch
            exp_log_res = acqf(X).exp()  # 1-dim batch
            # with no approximations (qEI): self.assertEqual(res[0].item(), 1.0)
            safe_upper_lower_bound = 0.999
            self.assertTrue(safe_upper_lower_bound <= exp_log_res[0].item())
            self.assertTrue(exp_log_res[0].item() <= 1.0 + acqf.tau_relu)
            # with no approximations (qEI): self.assertEqual(res[1].item(), 0.0)
            self.assertTrue(0.0 <= exp_log_res[1].item())
            self.assertTrue(exp_log_res[1].item() <= 0.0 + acqf.tau_relu)
            self.assertEqual(acqf.sampler.base_samples.shape, torch.Size([2, 1, 2, 1]))
            bs = acqf.sampler.base_samples.clone()
            acqf(X)
            self.assertTrue(torch.equal(acqf.sampler.base_samples, bs))
            exp_log_res = acqf(X.expand(2, 2, 1)).exp()  # 2-dim batch
            # self.assertEqual(res[0].item(), 1.0)
            safe_upper_lower_bound = 0.999
            self.assertTrue(safe_upper_lower_bound <= exp_log_res[0].item())
            self.assertTrue(exp_log_res[0].item() <= 1.0 + acqf.tau_relu)
            # self.assertEqual(res[1].item(), 0.0)
            self.assertTrue(0.0 <= exp_log_res[1].item())
            self.assertTrue(exp_log_res[1].item() <= 0.0 + acqf.tau_relu)
            # the base samples should have the batch dim collapsed
            self.assertEqual(acqf.sampler.base_samples.shape, torch.Size([2, 1, 2, 1]))
            bs = acqf.sampler.base_samples.clone()
            acqf(X.expand(2, 2, 1))
            self.assertTrue(torch.equal(acqf.sampler.base_samples, bs))

            # test batch mode, qmc
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([2]))
            acqf = qLogExpectedImprovement(model=mm, best_f=0, sampler=sampler)
            exp_log_res = acqf(X).exp()
            # self.assertEqual(res[0].item(), 1.0)
            safe_upper_lower_bound = 0.999
            self.assertTrue(safe_upper_lower_bound <= exp_log_res[0].item())
            self.assertTrue(exp_log_res[0].item() <= 1.0 + acqf.tau_relu)
            # self.assertEqual(res[1].item(), 0.0)
            self.assertTrue(0.0 <= exp_log_res[1].item())
            self.assertTrue(exp_log_res[1].item() <= 0.0 + acqf.tau_relu)
            self.assertEqual(acqf.sampler.base_samples.shape, torch.Size([2, 1, 2, 1]))
            bs = acqf.sampler.base_samples.clone()
            acqf(X)
            self.assertTrue(torch.equal(acqf.sampler.base_samples, bs))

    # # TODO: Test different objectives (incl. constraints)
