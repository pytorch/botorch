#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import warnings

import torch
from botorch.exceptions.warnings import OptimizationWarning
from botorch.fit import fit_gpytorch_model
from botorch.models.multitask import FixedNoiseMultiTaskGP, MultiTaskGP
from botorch.posteriors import GPyTorchPosterior
from botorch.utils.testing import BotorchTestCase
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from gpytorch.kernels import IndexKernel, MaternKernel, ScaleKernel
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood, GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.priors import GammaPrior


def _get_random_mt_data(**tkwargs):
    train_x = torch.linspace(0, 0.95, 10, **tkwargs) + 0.05 * torch.rand(10, **tkwargs)
    train_y1 = torch.sin(train_x * (2 * math.pi)) + torch.randn_like(train_x) * 0.2
    train_y2 = torch.cos(train_x * (2 * math.pi)) + torch.randn_like(train_x) * 0.2
    train_i_task1 = torch.full_like(train_x, dtype=torch.long, fill_value=0)
    train_i_task2 = torch.full_like(train_x, dtype=torch.long, fill_value=1)
    full_train_x = torch.cat([train_x, train_x])
    full_train_i = torch.cat([train_i_task1, train_i_task2])
    full_train_y = torch.cat([train_y1, train_y2])
    train_X = torch.stack([full_train_x, full_train_i.type_as(full_train_x)], dim=-1)
    train_Y = full_train_y.unsqueeze(-1)  # add output dim
    return train_X, train_Y


def _get_model(**tkwargs):
    train_X, train_Y = _get_random_mt_data(**tkwargs)
    model = MultiTaskGP(train_X, train_Y, task_feature=1)
    return model.to(**tkwargs)


def _get_model_single_output(**tkwargs):
    train_X, train_Y = _get_random_mt_data(**tkwargs)
    model = MultiTaskGP(train_X, train_Y, task_feature=1, output_tasks=[1])
    return model.to(**tkwargs)


def _get_fixed_noise_model(**tkwargs):
    train_X, train_Y = _get_random_mt_data(**tkwargs)
    train_Yvar = torch.full_like(train_Y, 0.05)
    model = FixedNoiseMultiTaskGP(train_X, train_Y, train_Yvar, task_feature=1)
    return model.to(**tkwargs)


def _get_fixed_noise_model_single_output(**tkwargs):
    train_X, train_Y = _get_random_mt_data(**tkwargs)
    train_Yvar = torch.full_like(train_Y, 0.05)
    model = FixedNoiseMultiTaskGP(
        train_X, train_Y, train_Yvar, task_feature=1, output_tasks=[1]
    )
    return model.to(**tkwargs)


class TestMultiTaskGP(BotorchTestCase):
    def test_MultiTaskGP(self):
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            model = _get_model(**tkwargs)
            self.assertIsInstance(model, MultiTaskGP)
            self.assertEqual(model.num_outputs, 2)
            self.assertIsInstance(model.likelihood, GaussianLikelihood)
            self.assertIsInstance(model.mean_module, ConstantMean)
            self.assertIsInstance(model.covar_module, ScaleKernel)
            matern_kernel = model.covar_module.base_kernel
            self.assertIsInstance(matern_kernel, MaternKernel)
            self.assertIsInstance(matern_kernel.lengthscale_prior, GammaPrior)
            self.assertIsInstance(model.task_covar_module, IndexKernel)
            self.assertEqual(model._rank, 2)
            self.assertEqual(
                model.task_covar_module.covar_factor.shape[-1], model._rank
            )

            # test model fitting
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=OptimizationWarning)
                mll = fit_gpytorch_model(mll, options={"maxiter": 1}, max_retries=1)

            # test posterior
            test_x = torch.rand(2, 1, **tkwargs)
            posterior_f = model.posterior(test_x)
            self.assertIsInstance(posterior_f, GPyTorchPosterior)
            self.assertIsInstance(posterior_f.mvn, MultitaskMultivariateNormal)
            self.assertEqual(posterior_f.mean.shape, torch.Size([2, 2]))
            self.assertEqual(posterior_f.variance.shape, torch.Size([2, 2]))

            # test that posterior w/ observation noise raises appropriate error
            with self.assertRaises(NotImplementedError):
                model.posterior(test_x, observation_noise=True)
            with self.assertRaises(NotImplementedError):
                model.posterior(test_x, observation_noise=torch.rand(2, **tkwargs))

            # test posterior w/ single output index
            posterior_f = model.posterior(test_x, output_indices=[0])
            self.assertIsInstance(posterior_f, GPyTorchPosterior)
            self.assertIsInstance(posterior_f.mvn, MultivariateNormal)
            self.assertEqual(posterior_f.mean.shape, torch.Size([2, 1]))
            self.assertEqual(posterior_f.variance.shape, torch.Size([2, 1]))

            # test posterior w/ bad output index
            with self.assertRaises(ValueError):
                model.posterior(test_x, output_indices=[2])

            # test posterior (batch eval)
            test_x = torch.rand(3, 2, 1, **tkwargs)
            posterior_f = model.posterior(test_x)
            self.assertIsInstance(posterior_f, GPyTorchPosterior)
            self.assertIsInstance(posterior_f.mvn, MultitaskMultivariateNormal)

            # test that unsupported batch shape MTGPs throw correct error
            with self.assertRaises(ValueError):
                MultiTaskGP(torch.rand(2, 2, 2), torch.rand(2, 2, 1), 0)

            # test that bad feature index throws correct error
            train_X, train_Y = _get_random_mt_data(**tkwargs)
            with self.assertRaises(ValueError):
                MultiTaskGP(train_X, train_Y, 2)

            # test that bad output task throws correct error
            with self.assertRaises(RuntimeError):
                MultiTaskGP(train_X, train_Y, 0, output_tasks=[2])

            # test error if outcome_transform attribute is present
            model.outcome_transform = None
            with self.assertRaises(NotImplementedError):
                model.posterior(test_x)

    def test_MultiTaskGP_single_output(self):
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            model = _get_model_single_output(**tkwargs)
            self.assertIsInstance(model, MultiTaskGP)
            self.assertEqual(model.num_outputs, 1)
            self.assertIsInstance(model.likelihood, GaussianLikelihood)
            self.assertIsInstance(model.mean_module, ConstantMean)
            self.assertIsInstance(model.covar_module, ScaleKernel)
            matern_kernel = model.covar_module.base_kernel
            self.assertIsInstance(matern_kernel, MaternKernel)
            self.assertIsInstance(matern_kernel.lengthscale_prior, GammaPrior)
            self.assertIsInstance(model.task_covar_module, IndexKernel)
            self.assertEqual(model._rank, 2)
            self.assertEqual(
                model.task_covar_module.covar_factor.shape[-1], model._rank
            )

            # test model fitting
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=OptimizationWarning)
                mll = fit_gpytorch_model(mll, options={"maxiter": 1}, max_retries=1)

            # test posterior
            test_x = torch.rand(2, 1, **tkwargs)
            posterior_f = model.posterior(test_x)
            self.assertIsInstance(posterior_f, GPyTorchPosterior)
            self.assertIsInstance(posterior_f.mvn, MultivariateNormal)

            # test posterior (batch eval)
            test_x = torch.rand(3, 2, 1, **tkwargs)
            posterior_f = model.posterior(test_x)
            self.assertIsInstance(posterior_f, GPyTorchPosterior)
            self.assertIsInstance(posterior_f.mvn, MultivariateNormal)


class TestFixedNoiseMultiTaskGP(BotorchTestCase):
    def test_FixedNoiseMultiTaskGP(self):
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            model = _get_fixed_noise_model(**tkwargs)
            self.assertIsInstance(model, FixedNoiseMultiTaskGP)
            self.assertEqual(model.num_outputs, 2)
            self.assertIsInstance(model.likelihood, FixedNoiseGaussianLikelihood)
            self.assertIsInstance(model.mean_module, ConstantMean)
            self.assertIsInstance(model.covar_module, ScaleKernel)
            matern_kernel = model.covar_module.base_kernel
            self.assertIsInstance(matern_kernel, MaternKernel)
            self.assertIsInstance(matern_kernel.lengthscale_prior, GammaPrior)
            self.assertIsInstance(model.task_covar_module, IndexKernel)
            self.assertEqual(model._rank, 2)
            self.assertEqual(
                model.task_covar_module.covar_factor.shape[-1], model._rank
            )

            # test model fitting
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=OptimizationWarning)
                mll = fit_gpytorch_model(mll, options={"maxiter": 1}, max_retries=1)

            # test posterior
            test_x = torch.rand(2, 1, **tkwargs)
            posterior_f = model.posterior(test_x)
            self.assertIsInstance(posterior_f, GPyTorchPosterior)
            self.assertIsInstance(posterior_f.mvn, MultitaskMultivariateNormal)
            self.assertEqual(posterior_f.mean.shape, torch.Size([2, 2]))
            self.assertEqual(posterior_f.variance.shape, torch.Size([2, 2]))

            # test that posterior w/ observation noise raises appropriate error
            with self.assertRaises(NotImplementedError):
                model.posterior(test_x, observation_noise=True)
            with self.assertRaises(NotImplementedError):
                model.posterior(test_x, observation_noise=torch.rand(2, **tkwargs))

            # test posterior w/ single output index
            posterior_f = model.posterior(test_x, output_indices=[0])
            self.assertIsInstance(posterior_f, GPyTorchPosterior)
            self.assertIsInstance(posterior_f.mvn, MultivariateNormal)
            self.assertEqual(posterior_f.mean.shape, torch.Size([2, 1]))
            self.assertEqual(posterior_f.variance.shape, torch.Size([2, 1]))

            # test posterior w/ bad output index
            with self.assertRaises(ValueError):
                model.posterior(test_x, output_indices=[2])

            # test posterior (batch eval)
            test_x = torch.rand(3, 2, 1, **tkwargs)
            posterior_f = model.posterior(test_x)
            self.assertIsInstance(posterior_f, GPyTorchPosterior)
            self.assertIsInstance(posterior_f.mvn, MultitaskMultivariateNormal)

            # test that unsupported batch shape MTGPs throw correct error
            with self.assertRaises(ValueError):
                FixedNoiseMultiTaskGP(
                    torch.rand(2, 2, 2), torch.rand(2, 2, 1), torch.rand(2, 2, 1), 0
                )

            # test that bad feature index throws correct error
            train_X, train_Y = _get_random_mt_data(**tkwargs)
            train_Yvar = torch.full_like(train_Y, 0.05)
            with self.assertRaises(ValueError):
                FixedNoiseMultiTaskGP(train_X, train_Y, train_Yvar, 2)

            # test that bad output task throws correct error
            with self.assertRaises(RuntimeError):
                FixedNoiseMultiTaskGP(train_X, train_Y, train_Yvar, 0, output_tasks=[2])

    def test_FixedNoiseMultiTaskGP_single_output(self):
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            model = _get_fixed_noise_model_single_output(**tkwargs)
            self.assertIsInstance(model, FixedNoiseMultiTaskGP)
            self.assertEqual(model.num_outputs, 1)
            self.assertIsInstance(model.likelihood, FixedNoiseGaussianLikelihood)
            self.assertIsInstance(model.mean_module, ConstantMean)
            self.assertIsInstance(model.covar_module, ScaleKernel)
            matern_kernel = model.covar_module.base_kernel
            self.assertIsInstance(matern_kernel, MaternKernel)
            self.assertIsInstance(matern_kernel.lengthscale_prior, GammaPrior)
            self.assertIsInstance(model.task_covar_module, IndexKernel)
            self.assertEqual(model._rank, 2)
            self.assertEqual(
                model.task_covar_module.covar_factor.shape[-1], model._rank
            )

            # test model fitting
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=OptimizationWarning)
                mll = fit_gpytorch_model(mll, options={"maxiter": 1}, max_retries=1)

            # test posterior
            test_x = torch.rand(2, 1, **tkwargs)
            posterior_f = model.posterior(test_x)
            self.assertIsInstance(posterior_f, GPyTorchPosterior)
            self.assertIsInstance(posterior_f.mvn, MultivariateNormal)

            # test posterior (batch eval)
            test_x = torch.rand(3, 2, 1, **tkwargs)
            posterior_f = model.posterior(test_x)
            self.assertIsInstance(posterior_f, GPyTorchPosterior)
            self.assertIsInstance(posterior_f.mvn, MultivariateNormal)
