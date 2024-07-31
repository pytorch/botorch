#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch
from botorch.acquisition.input_constructors import get_acqf_input_constructor
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.utils.datasets import SupervisedDataset
from botorch.utils.testing import BotorchTestCase, MockModel, MockPosterior
from botorch_community.acquisition.bayesian_active_learning import (
    qBayesianQueryByComittee,
    qBayesianVarianceReduction,
    qStatisticalDistanceActiveLearning,
)
from botorch_community.acquisition.scorebo import qSelfCorrectingBayesianOptimization


class InputConstructorBaseTestCase(BotorchTestCase):
    def setUp(self, suppress_input_warnings: bool = True) -> None:
        super().setUp(suppress_input_warnings=suppress_input_warnings)
        self.mock_model = MockModel(
            posterior=MockPosterior(mean=None, variance=None, base_shape=(1,))
        )

        X1 = torch.rand(3, 2)
        X2 = torch.rand(3, 2)
        Y1 = torch.rand(3, 1)
        Y2 = torch.rand(3, 1)
        feature_names = ["X1", "X2"]
        outcome_names = ["Y"]

        self.blockX_blockY = {
            0: SupervisedDataset(
                X1, Y1, feature_names=feature_names, outcome_names=outcome_names
            )
        }
        self.blockX_multiY = {
            0: SupervisedDataset(
                X1, Y1, feature_names=feature_names, outcome_names=outcome_names
            ),
            1: SupervisedDataset(
                X1, Y2, feature_names=feature_names, outcome_names=outcome_names
            ),
        }
        self.multiX_multiY = {
            0: SupervisedDataset(
                X1, Y1, feature_names=feature_names, outcome_names=outcome_names
            ),
            1: SupervisedDataset(
                X2, Y2, feature_names=feature_names, outcome_names=outcome_names
            ),
        }
        self.bounds = 2 * [(0.0, 1.0)]


class TestFullyBayesianAcquisitionFunctionInputConstructors(
    InputConstructorBaseTestCase
):
    def test_construct_inputs_scorebo(self) -> None:
        func = get_acqf_input_constructor(qSelfCorrectingBayesianOptimization)
        num_samples, num_optima = 3, 7
        model = SaasFullyBayesianSingleTaskGP(
            self.blockX_blockY[0].X, self.blockX_blockY[0].Y
        )

        model.load_mcmc_samples(
            {
                "lengthscale": torch.rand(
                    num_samples,
                    1,
                    self.blockX_blockY[0].X.shape[-1],
                    dtype=torch.double,
                ),
                "outputscale": torch.rand(num_samples, dtype=torch.double),
                "mean": torch.randn(num_samples, dtype=torch.double),
                "noise": torch.rand(num_samples, 1, dtype=torch.double),
            }
        )

        kwargs = func(
            model=model,
            training_data=self.blockX_blockY,
            bounds=self.bounds,
            num_optima=num_optima,
            maximize=False,
            distance_metric="kl_divergence",
        )
        self.assertFalse(kwargs["maximize"])
        self.assertEqual(self.blockX_blockY[0].X.dtype, kwargs["optimal_inputs"].dtype)
        self.assertEqual(len(kwargs["optimal_inputs"]), num_optima)
        self.assertEqual(len(kwargs["optimal_outputs"]), num_optima)
        # asserting that, for the non-batch case, the optimal inputs are
        # of shape num_models x N x D and outputs are num_models x N x 1
        self.assertEqual(len(kwargs["optimal_inputs"].shape), 3)
        self.assertEqual(len(kwargs["optimal_outputs"].shape), 3)
        self.assertEqual(kwargs["distance_metric"], "kl_divergence")
        qSelfCorrectingBayesianOptimization(**kwargs)

    def test_construct_inputs_sal(self) -> None:
        func = get_acqf_input_constructor(qStatisticalDistanceActiveLearning)
        num_samples = 3
        model = SaasFullyBayesianSingleTaskGP(
            self.blockX_blockY[0].X, self.blockX_blockY[0].Y
        )

        model.load_mcmc_samples(
            {
                "lengthscale": torch.rand(
                    num_samples,
                    1,
                    self.blockX_blockY[0].X.shape[-1],
                    dtype=torch.double,
                ),
                "outputscale": torch.rand(num_samples, dtype=torch.double),
                "mean": torch.randn(num_samples, dtype=torch.double),
                "noise": torch.rand(num_samples, 1, dtype=torch.double),
            }
        )

        kwargs = func(
            model=model,
            training_data=self.blockX_blockY,
            bounds=self.bounds,
            distance_metric="kl_divergence",
        )

        self.assertEqual(kwargs["distance_metric"], "kl_divergence")
        qStatisticalDistanceActiveLearning(**kwargs)

    def test_construct_inputs_bqbc(self) -> None:
        func = get_acqf_input_constructor(qBayesianQueryByComittee)
        num_samples = 3
        model = SaasFullyBayesianSingleTaskGP(
            self.blockX_blockY[0].X, self.blockX_blockY[0].Y
        )

        model.load_mcmc_samples(
            {
                "lengthscale": torch.rand(
                    num_samples,
                    1,
                    self.blockX_blockY[0].X.shape[-1],
                    dtype=torch.double,
                ),
                "outputscale": torch.rand(num_samples, dtype=torch.double),
                "mean": torch.randn(num_samples, dtype=torch.double),
                "noise": torch.rand(num_samples, 1, dtype=torch.double),
            }
        )

        kwargs = func(
            model=model,
            training_data=self.blockX_blockY,
            bounds=self.bounds,
        )

        qBayesianQueryByComittee(**kwargs)

    def test_construct_inputs_bayesian_variance_reduction(self) -> None:
        func = get_acqf_input_constructor(qBayesianVarianceReduction)
        num_samples = 3
        model = SaasFullyBayesianSingleTaskGP(
            self.blockX_blockY[0].X, self.blockX_blockY[0].Y
        )

        model.load_mcmc_samples(
            {
                "lengthscale": torch.rand(
                    num_samples,
                    1,
                    self.blockX_blockY[0].X.shape[-1],
                    dtype=torch.double,
                ),
                "outputscale": torch.rand(num_samples, dtype=torch.double),
                "mean": torch.randn(num_samples, dtype=torch.double),
                "noise": torch.rand(num_samples, 1, dtype=torch.double),
            }
        )

        kwargs = func(
            model=model,
            training_data=self.blockX_blockY,
            bounds=self.bounds,
        )

        qBayesianVarianceReduction(**kwargs)
