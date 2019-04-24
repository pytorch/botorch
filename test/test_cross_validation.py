#! /usr/bin/env python3

import math
import unittest

import torch
from botorch.cross_validation import batch_cross_validation, gen_loo_cv_folds
from botorch.models.gp_regression import FixedNoiseGP, SingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood


def _get_random_data(batch_shape, num_outputs, n=10, **tkwargs):
    train_x = torch.linspace(0, 0.95, n, **tkwargs).unsqueeze(-1) + 0.05 * torch.rand(
        n, 1, **tkwargs
    ).repeat(batch_shape + torch.Size([1, 1]))
    train_y = torch.sin(train_x * (2 * math.pi)) + 0.2 * torch.randn(
        n, num_outputs, **tkwargs
    ).repeat(batch_shape + torch.Size([1, 1]))

    if num_outputs == 1:
        train_y = train_y.squeeze(-1)
    return train_x, train_y


class TestFitBatchCrossValidation(unittest.TestCase):
    def test_single_task_batch_cv(self, cuda=False):
        n = 10
        for batch_shape in (torch.Size([]), torch.Size([2])):
            for num_outputs in (1, 2):
                for double in (False, True):
                    tkwargs = {
                        "device": torch.device("cuda") if cuda else torch.device("cpu"),
                        "dtype": torch.double if double else torch.float,
                    }
                    train_X, train_Y = _get_random_data(
                        batch_shape=batch_shape, num_outputs=num_outputs, n=n, **tkwargs
                    )
                    train_Yvar = torch.full_like(train_Y, 0.01)
                    noiseless_cv_folds = gen_loo_cv_folds(
                        train_X=train_X, train_Y=train_Y
                    )
                    # Test SingleTaskGP
                    cv_results = batch_cross_validation(
                        model_cls=SingleTaskGP,
                        mll_cls=ExactMarginalLogLikelihood,
                        cv_folds=noiseless_cv_folds,
                        fit_args={"options": {"maxiter": 1}},
                    )
                    expected_shape = batch_shape + torch.Size([n, 1, num_outputs])
                    self.assertEqual(cv_results.posterior.mean.shape, expected_shape)
                    self.assertEqual(cv_results.observed_Y.shape, expected_shape)

                    # Test FixedNoiseGP
                    noisy_cv_folds = gen_loo_cv_folds(
                        train_X=train_X, train_Y=train_Y, train_Yvar=train_Yvar
                    )
                    cv_results = batch_cross_validation(
                        model_cls=FixedNoiseGP,
                        mll_cls=ExactMarginalLogLikelihood,
                        cv_folds=noisy_cv_folds,
                        fit_args={"options": {"maxiter": 1}},
                    )
                    self.assertEqual(cv_results.posterior.mean.shape, expected_shape)
                    self.assertEqual(cv_results.observed_Y.shape, expected_shape)
                    self.assertEqual(cv_results.observed_Y.shape, expected_shape)

    def test_single_task_batch_cv_cuda(self):
        if torch.cuda.is_available():
            self.test_single_task_batch_cv(cuda=True)
