#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import warnings

import torch
from botorch.cross_validation import batch_cross_validation, gen_loo_cv_folds
from botorch.exceptions.warnings import OptimizationWarning
from botorch.models.gp_regression import FixedNoiseGP, SingleTaskGP
from botorch.utils.testing import _get_random_data, BotorchTestCase
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood


class TestFitBatchCrossValidation(BotorchTestCase):
    def test_single_task_batch_cv(self):
        n = 10
        for batch_shape, m, dtype in itertools.product(
            (torch.Size(), torch.Size([2])), (1, 2), (torch.float, torch.double)
        ):
            tkwargs = {"device": self.device, "dtype": dtype}
            train_X, train_Y = _get_random_data(
                batch_shape=batch_shape, m=m, n=n, **tkwargs
            )
            if m == 1:
                train_Y = train_Y.squeeze(-1)
            train_Yvar = torch.full_like(train_Y, 0.01)
            noiseless_cv_folds = gen_loo_cv_folds(train_X=train_X, train_Y=train_Y)
            # check shapes
            expected_shape_train_X = batch_shape + torch.Size(
                [n, n - 1, train_X.shape[-1]]
            )
            expected_shape_test_X = batch_shape + torch.Size([n, 1, train_X.shape[-1]])
            self.assertEqual(noiseless_cv_folds.train_X.shape, expected_shape_train_X)
            self.assertEqual(noiseless_cv_folds.test_X.shape, expected_shape_test_X)

            expected_shape_train_Y = batch_shape + torch.Size([n, n - 1, m])
            expected_shape_test_Y = batch_shape + torch.Size([n, 1, m])

            self.assertEqual(noiseless_cv_folds.train_Y.shape, expected_shape_train_Y)
            self.assertEqual(noiseless_cv_folds.test_Y.shape, expected_shape_test_Y)
            self.assertIsNone(noiseless_cv_folds.train_Yvar)
            self.assertIsNone(noiseless_cv_folds.test_Yvar)
            # Test SingleTaskGP
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=OptimizationWarning)
                cv_results = batch_cross_validation(
                    model_cls=SingleTaskGP,
                    mll_cls=ExactMarginalLogLikelihood,
                    cv_folds=noiseless_cv_folds,
                    fit_args={"optimizer_kwargs": {"options": {"maxiter": 1}}},
                )
            expected_shape = batch_shape + torch.Size([n, 1, m])
            self.assertEqual(cv_results.posterior.mean.shape, expected_shape)
            self.assertEqual(cv_results.observed_Y.shape, expected_shape)

            # Test FixedNoiseGP
            noisy_cv_folds = gen_loo_cv_folds(
                train_X=train_X, train_Y=train_Y, train_Yvar=train_Yvar
            )
            # check shapes
            self.assertEqual(noisy_cv_folds.train_X.shape, expected_shape_train_X)
            self.assertEqual(noisy_cv_folds.test_X.shape, expected_shape_test_X)
            self.assertEqual(noisy_cv_folds.train_Y.shape, expected_shape_train_Y)
            self.assertEqual(noisy_cv_folds.test_Y.shape, expected_shape_test_Y)
            self.assertEqual(noisy_cv_folds.train_Yvar.shape, expected_shape_train_Y)
            self.assertEqual(noisy_cv_folds.test_Yvar.shape, expected_shape_test_Y)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=OptimizationWarning)
                cv_results = batch_cross_validation(
                    model_cls=FixedNoiseGP,
                    mll_cls=ExactMarginalLogLikelihood,
                    cv_folds=noisy_cv_folds,
                    fit_args={"optimizer_kwargs": {"options": {"maxiter": 1}}},
                )
            self.assertEqual(cv_results.posterior.mean.shape, expected_shape)
            self.assertEqual(cv_results.observed_Y.shape, expected_shape)
            self.assertEqual(cv_results.observed_Y.shape, expected_shape)
