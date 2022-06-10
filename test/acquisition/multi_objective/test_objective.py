#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools

import torch
from botorch.acquisition.multi_objective.multi_output_risk_measures import (
    MultiOutputExpectation,
)
from botorch.acquisition.multi_objective.objective import (
    FeasibilityWeightedMCMultiOutputObjective,
    IdentityMCMultiOutputObjective,
    MCMultiOutputObjective,
    UnstandardizeAnalyticMultiOutputObjective,
    UnstandardizeMCMultiOutputObjective,
    WeightedMCMultiOutputObjective,
)
from botorch.acquisition.objective import IdentityMCObjective
from botorch.exceptions.errors import BotorchError, BotorchTensorDimensionError
from botorch.models.transforms.outcome import Standardize
from botorch.utils.testing import BotorchTestCase, MockModel, MockPosterior


class TestMCMultiOutputObjective(BotorchTestCase):
    def test_abstract_raises(self):
        with self.assertRaises(TypeError):
            MCMultiOutputObjective()


class TestIdentityMCMultiOutputObjective(BotorchTestCase):
    def test_identity_mc_multi_output_objective(self):
        objective = IdentityMCMultiOutputObjective()
        with self.assertRaises(BotorchTensorDimensionError):
            IdentityMCMultiOutputObjective(outcomes=[0])
        # test negative outcome without specifying num_outcomes
        with self.assertRaises(BotorchError):
            IdentityMCMultiOutputObjective(outcomes=[0, -1])
        for batch_shape, m, dtype in itertools.product(
            ([], [3]), (2, 3), (torch.float, torch.double)
        ):
            samples = torch.rand(*batch_shape, 2, m, device=self.device, dtype=dtype)
            self.assertTrue(torch.equal(objective(samples), samples))


class TestWeightedMCMultiOutputObjective(BotorchTestCase):
    def test_weighted_mc_multi_output_objective(self):
        with self.assertRaises(BotorchTensorDimensionError):
            WeightedMCMultiOutputObjective(weights=torch.rand(3, 1))
        with self.assertRaises(BotorchTensorDimensionError):
            WeightedMCMultiOutputObjective(
                weights=torch.rand(3), outcomes=[0, 1], num_outcomes=3
            )
        for batch_shape, m, dtype in itertools.product(
            ([], [3]), (2, 3), (torch.float, torch.double)
        ):
            weights = torch.rand(m, device=self.device, dtype=dtype)
            objective = WeightedMCMultiOutputObjective(weights=weights)
            samples = torch.rand(*batch_shape, 2, m, device=self.device, dtype=dtype)
            self.assertTrue(torch.equal(objective(samples), samples * weights))


class TestFeasibilityWeightedMCMultiOutputObjective(BotorchTestCase):
    def test_feasibility_weighted_mc_multi_output_objective(self):
        for dtype in (torch.float, torch.double):
            tkwargs = {"dtype": dtype, "device": self.device}
            X = torch.zeros(5, 1, **tkwargs)
            # The infeasible cost will be 0.0.
            means = torch.tensor(
                [
                    [1.0, 0.5],
                    [2.0, -1.0],
                    [3.0, -0.5],
                    [4.0, 1.0],
                    [5.0, 1.0],
                ],
                **tkwargs,
            )
            variances = torch.zeros(5, 2, **tkwargs)
            mm = MockModel(MockPosterior(mean=means, variance=variances))
            feas_obj = FeasibilityWeightedMCMultiOutputObjective(
                model=mm,
                X_baseline=X,
                constraint_idcs=[-1],
                objective=None,
            )
            feas_samples = feas_obj(means)
            expected = torch.tensor([[1.0], [0.0], [0.0], [4.0], [5.0]], **tkwargs)
            self.assertTrue(torch.allclose(feas_samples, expected))
            self.assertTrue(feas_obj._verify_output_shape)

            # With an objective.
            dummy_obj = MultiOutputExpectation(n_w=1, weights=[2.0])
            dummy_obj._verify_output_shape = False  # for testing
            feas_obj = FeasibilityWeightedMCMultiOutputObjective(
                model=mm,
                X_baseline=X,
                constraint_idcs=[1],
                objective=dummy_obj,
            )
            feas_samples = feas_obj(means)
            self.assertTrue(torch.allclose(feas_samples, expected * 2.0))
            self.assertFalse(feas_obj._verify_output_shape)

            # No constraints.
            feas_obj = FeasibilityWeightedMCMultiOutputObjective(
                model=mm,
                X_baseline=X,
                constraint_idcs=[],
                objective=None,
            )
            feas_samples = feas_obj(means)
            self.assertIs(feas_samples, means)

            # With a single-output objective.
            feas_obj = FeasibilityWeightedMCMultiOutputObjective(
                model=mm,
                X_baseline=X,
                constraint_idcs=[1],
                objective=IdentityMCObjective(),
            )
            feas_samples = feas_obj(means)
            self.assertTrue(torch.allclose(feas_samples, expected.squeeze(-1)))

            # Error with duplicate idcs.
            with self.assertRaisesRegex(ValueError, "duplicate"):
                FeasibilityWeightedMCMultiOutputObjective(
                    model=mm,
                    X_baseline=X,
                    constraint_idcs=[1, -1],
                )


class TestUnstandardizeMultiOutputObjective(BotorchTestCase):
    def test_unstandardize_mo_objective(self):
        Y_mean = torch.ones(2)
        Y_std = torch.ones(2)
        with self.assertRaises(BotorchTensorDimensionError):
            UnstandardizeMCMultiOutputObjective(
                Y_mean=Y_mean, Y_std=Y_std, outcomes=[0, 1, 2]
            )
        for objective_class in (
            UnstandardizeMCMultiOutputObjective,
            UnstandardizeAnalyticMultiOutputObjective,
        ):
            with self.assertRaises(BotorchTensorDimensionError):
                objective_class(Y_mean=Y_mean.unsqueeze(0), Y_std=Y_std)
            with self.assertRaises(BotorchTensorDimensionError):
                objective_class(Y_mean=Y_mean, Y_std=Y_std.unsqueeze(0))
            objective = objective_class(Y_mean=Y_mean, Y_std=Y_std)
            for batch_shape, m, outcomes, dtype in itertools.product(
                ([], [3]), (2, 3), (None, [-2, -1]), (torch.float, torch.double)
            ):
                Y_mean = torch.rand(m, dtype=dtype, device=self.device)
                Y_std = torch.rand(m, dtype=dtype, device=self.device).clamp_min(1e-3)
                kwargs = {}
                if objective_class == UnstandardizeMCMultiOutputObjective:
                    kwargs["outcomes"] = outcomes
                objective = objective_class(Y_mean=Y_mean, Y_std=Y_std, **kwargs)
                if objective_class == UnstandardizeAnalyticMultiOutputObjective:
                    if outcomes is None:
                        # passing outcomes is not currently supported
                        mean = torch.rand(2, m, dtype=dtype, device=self.device)
                        variance = variance = torch.rand(
                            2, m, dtype=dtype, device=self.device
                        )
                        mock_posterior = MockPosterior(mean=mean, variance=variance)
                        tf_posterior = objective(mock_posterior)
                        tf = Standardize(m=m)
                        tf.means = Y_mean
                        tf.stdvs = Y_std
                        tf._stdvs_sq = Y_std.pow(2)
                        tf.eval()
                        expected_posterior = tf.untransform_posterior(mock_posterior)
                        self.assertTrue(
                            torch.equal(tf_posterior.mean, expected_posterior.mean)
                        )
                        self.assertTrue(
                            torch.equal(
                                tf_posterior.variance, expected_posterior.variance
                            )
                        )
                else:

                    samples = torch.rand(
                        *batch_shape, 2, m, dtype=dtype, device=self.device
                    )
                    obj_expected = samples * Y_std.to(dtype=dtype) + Y_mean.to(
                        dtype=dtype
                    )
                    if outcomes is not None:
                        obj_expected = obj_expected[..., outcomes]
                    self.assertTrue(torch.equal(objective(samples), obj_expected))
