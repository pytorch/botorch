#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.acquisition.penalized import (
    GaussianPenalty,
    group_lasso_regularizer,
    GroupLassoPenalty,
    L1Penalty,
    L1PenaltyObjective,
    L2Penalty,
    PenalizedAcquisitionFunction,
    PenalizedMCObjective,
)
from botorch.exceptions import UnsupportedError
from botorch.sampling.normal import IIDNormalSampler
from botorch.utils.testing import BotorchTestCase, MockModel, MockPosterior
from torch import Tensor


def generic_obj(samples: Tensor, X=None) -> Tensor:
    return torch.log(torch.sum(samples**2, dim=-1))


class TestL2Penalty(BotorchTestCase):
    def test_gaussian_penalty(self):
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            init_point = torch.tensor([1.0, 1.0, 1.0], **tkwargs)
            l2_module = L2Penalty(init_point=init_point)

            # testing a batch of two points
            sample_point = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], **tkwargs)

            diff_norm_squared = (
                torch.norm((sample_point - init_point), p=2, dim=-1) ** 2
            )
            real_value = diff_norm_squared.max(dim=-1).values
            computed_value = l2_module(sample_point)
            self.assertEqual(computed_value.item(), real_value.item())


class TestL1Penalty(BotorchTestCase):
    def test_l1_penalty(self):
        for dtype in (torch.float, torch.double):
            init_point = torch.tensor([1.0, 1.0, 1.0], device=self.device, dtype=dtype)
            l1_module = L1Penalty(init_point=init_point)

            # testing a batch of two points
            sample_point = torch.tensor(
                [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], device=self.device, dtype=dtype
            )

            diff_l1_norm = torch.norm((sample_point - init_point), p=1, dim=-1)
            real_value = diff_l1_norm.max(dim=-1).values
            computed_value = l1_module(sample_point)
            self.assertEqual(computed_value.item(), real_value.item())


class TestGaussianPenalty(BotorchTestCase):
    def test_gaussian_penalty(self):
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            init_point = torch.tensor([1.0, 1.0, 1.0], **tkwargs)
            sigma = 0.1
            gaussian_module = GaussianPenalty(init_point=init_point, sigma=sigma)

            # testing a batch of two points
            sample_point = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], **tkwargs)

            diff_norm_squared = (
                torch.norm((sample_point - init_point), p=2, dim=-1) ** 2
            )
            max_l2_distance = diff_norm_squared.max(dim=-1).values
            real_value = torch.exp(max_l2_distance / 2 / sigma**2)
            computed_value = gaussian_module(sample_point)
            self.assertEqual(computed_value.item(), real_value.item())


class TestGroupLassoPenalty(BotorchTestCase):
    def test_group_lasso_penalty(self):
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            init_point = torch.tensor([0.5, 0.5, 0.5], **tkwargs)
            groups = [[0, 2], [1]]
            group_lasso_module = GroupLassoPenalty(init_point=init_point, groups=groups)

            # testing a single point
            sample_point = torch.tensor([[1.0, 2.0, 3.0]], **tkwargs)
            real_value = group_lasso_regularizer(
                sample_point - init_point, groups
            )  # torch.tensor([5.105551242828369], **tkwargs)
            computed_value = group_lasso_module(sample_point)
            self.assertEqual(computed_value.item(), real_value.item())

            # testing unsupported input dim: X.shape[-2] > 1
            sample_point_2 = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], **tkwargs)
            with self.assertRaises(NotImplementedError):
                group_lasso_module(sample_point_2)


class TestPenalizedAcquisitionFunction(BotorchTestCase):
    def test_penalized_acquisition_function(self):
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            mock_model = MockModel(
                MockPosterior(
                    mean=torch.tensor([[1.0]], **tkwargs),
                    variance=torch.tensor([[1.0]], **tkwargs),
                )
            )
            init_point = torch.tensor([0.5, 0.5, 0.5], **tkwargs)
            groups = [[0, 2], [1]]
            raw_acqf = ExpectedImprovement(model=mock_model, best_f=1.0)
            penalty = GroupLassoPenalty(init_point=init_point, groups=groups)
            lmbda = 0.1
            acqf = PenalizedAcquisitionFunction(
                raw_acqf=raw_acqf, penalty_func=penalty, regularization_parameter=lmbda
            )

            sample_point = torch.tensor([[1.0, 2.0, 3.0]], **tkwargs)
            raw_value = raw_acqf(sample_point)
            penalty_value = penalty(sample_point)
            real_value = raw_value - lmbda * penalty_value
            computed_value = acqf(sample_point)
            self.assertTrue(torch.equal(real_value, computed_value))

            # testing X_pending for analytic raw_acqfn (EI)
            X_pending = torch.tensor([0.1, 0.2, 0.3], **tkwargs)
            with self.assertRaises(UnsupportedError):
                acqf.set_X_pending(X_pending)

            # testing X_pending for non-analytic raw_acqfn (EI)
            sampler = IIDNormalSampler(sample_shape=torch.Size([2]))
            raw_acqf_2 = qExpectedImprovement(
                model=mock_model, best_f=0, sampler=sampler
            )
            init_point = torch.tensor([1.0, 1.0, 1.0], **tkwargs)
            l2_module = L2Penalty(init_point=init_point)
            acqf_2 = PenalizedAcquisitionFunction(
                raw_acqf=raw_acqf_2,
                penalty_func=l2_module,
                regularization_parameter=lmbda,
            )

            X_pending = torch.tensor([0.1, 0.2, 0.3], **tkwargs)
            acqf_2.set_X_pending(X_pending)
            self.assertTrue(torch.equal(acqf_2.X_pending, X_pending))


class TestL1PenaltyObjective(BotorchTestCase):
    def test_l1_penalty(self):
        for dtype in (torch.float, torch.double):
            init_point = torch.tensor([1.0, 1.0, 1.0], device=self.device, dtype=dtype)
            l1_module = L1PenaltyObjective(init_point=init_point)

            # testing a batch of two points
            sample_point = torch.tensor(
                [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], device=self.device, dtype=dtype
            )

            real_values = torch.norm(
                (sample_point - init_point), p=1, dim=-1
            ).unsqueeze(dim=0)
            computed_values = l1_module(sample_point)
            self.assertTrue(torch.equal(real_values, computed_values))


class TestPenalizedMCObjective(BotorchTestCase):
    def test_penalized_mc_objective(self):
        for dtype in (torch.float, torch.double):
            init_point = torch.tensor(
                [0.0, 0.0, 0.0, 0.0, 0.0], device=self.device, dtype=dtype
            )
            l1_penalty_obj = L1PenaltyObjective(init_point=init_point)
            obj = PenalizedMCObjective(
                objective=generic_obj,
                penalty_objective=l1_penalty_obj,
                regularization_parameter=0.1,
            )
            # test 'd' Tensor X
            samples = torch.randn(4, 3, device=self.device, dtype=dtype)
            X = torch.randn(4, 5, device=self.device, dtype=dtype)
            penalized_obj = generic_obj(samples) - 0.1 * l1_penalty_obj(X)
            self.assertTrue(torch.equal(obj(samples, X), penalized_obj))
            # test 'q x d' Tensor X
            samples = torch.randn(4, 2, 3, device=self.device, dtype=dtype)
            X = torch.randn(2, 5, device=self.device, dtype=dtype)
            penalized_obj = generic_obj(samples) - 0.1 * l1_penalty_obj(X)
            self.assertTrue(torch.equal(obj(samples, X), penalized_obj))
            # test 'batch-shape x q x d' Tensor X
            samples = torch.randn(4, 3, 2, 3, device=self.device, dtype=dtype)
            X = torch.randn(3, 2, 5, device=self.device, dtype=dtype)
            penalized_obj = generic_obj(samples) - 0.1 * l1_penalty_obj(X)
            self.assertTrue(torch.equal(obj(samples, X), penalized_obj))
