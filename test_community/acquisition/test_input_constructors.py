#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch
from botorch.acquisition.input_constructors import get_acqf_input_constructor
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch_community.acquisition.bayesian_active_learning import (
    qBayesianActiveLearningByDisagreement,
    qBayesianQueryByComittee,
    qBayesianVarianceReduction,
    qStatisticalDistanceActiveLearning,
)
from botorch_community.acquisition.scorebo import qSelfCorrectingBayesianOptimization

from test.acquisition.test_input_constructors import InputConstructorBaseTestCase


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

    def test_construct_inputs_bald(self) -> None:
        func = get_acqf_input_constructor(qBayesianActiveLearningByDisagreement)
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

        qBayesianActiveLearningByDisagreement(**kwargs)

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
