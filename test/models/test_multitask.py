#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import math
import warnings

import torch
from botorch.exceptions.warnings import OptimizationWarning
from botorch.fit import fit_gpytorch_model
from botorch.models.multitask import FixedNoiseMultiTaskGP, MultiTaskGP
from botorch.models.transforms.input import Normalize
from botorch.posteriors import GPyTorchPosterior
from botorch.utils.containers import TrainingData
from botorch.utils.testing import BotorchTestCase
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from gpytorch.kernels import IndexKernel, MaternKernel, ScaleKernel
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood, GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.priors import GammaPrior
from gpytorch.priors.lkj_prior import LKJCovariancePrior


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


def _get_model(input_transform=None, **tkwargs):
    return _get_model_and_training_data(input_transform=input_transform, **tkwargs)[0]


def _get_model_and_training_data(input_transform=None, **tkwargs):
    train_X, train_Y = _get_random_mt_data(**tkwargs)
    model = MultiTaskGP(
        train_X, train_Y, task_feature=1, input_transform=input_transform
    )
    return model.to(**tkwargs), train_X, train_Y


def _get_model_single_output(**tkwargs):
    train_X, train_Y = _get_random_mt_data(**tkwargs)
    model = MultiTaskGP(train_X, train_Y, task_feature=1, output_tasks=[1])
    return model.to(**tkwargs)


def _get_fixed_noise_model(input_transform=None, **tkwargs):
    return _get_fixed_noise_model_and_training_data(
        input_transform=input_transform, **tkwargs
    )[0]


def _get_fixed_noise_model_and_training_data(input_transform=None, **tkwargs):
    train_X, train_Y = _get_random_mt_data(**tkwargs)
    train_Yvar = torch.full_like(train_Y, 0.05)
    model = FixedNoiseMultiTaskGP(
        train_X, train_Y, train_Yvar, task_feature=1, input_transform=input_transform
    )
    return model.to(**tkwargs), train_X, train_Y, train_Yvar


def _get_fixed_noise_model_single_output(**tkwargs):
    train_X, train_Y = _get_random_mt_data(**tkwargs)
    train_Yvar = torch.full_like(train_Y, 0.05)
    model = FixedNoiseMultiTaskGP(
        train_X, train_Y, train_Yvar, task_feature=1, output_tasks=[1]
    )
    return model.to(**tkwargs)


def _get_fixed_prior_model(**tkwargs):
    train_X, train_Y = _get_random_mt_data(**tkwargs)
    sd_prior = GammaPrior(2.0, 0.15)
    sd_prior._event_shape = torch.Size([2])
    model = MultiTaskGP(
        train_X,
        train_Y,
        task_feature=1,
        task_covar_prior=LKJCovariancePrior(2, 0.6, sd_prior),
    )
    return model.to(**tkwargs)


def _get_fixed_noise_and_prior_model(**tkwargs):
    train_X, train_Y = _get_random_mt_data(**tkwargs)
    train_Yvar = torch.full_like(train_Y, 0.05)
    sd_prior = GammaPrior(2.0, 0.15)
    sd_prior._event_shape = torch.Size([2])
    model = FixedNoiseMultiTaskGP(
        train_X,
        train_Y,
        train_Yvar,
        task_feature=1,
        task_covar_prior=LKJCovariancePrior(2, 0.6, sd_prior),
    )
    return model.to(**tkwargs)


class TestMultiTaskGP(BotorchTestCase):
    def test_MultiTaskGP(self):
        bounds = torch.tensor([[-1.0, 0.0], [1.0, 1.0]])
        for dtype, use_intf in itertools.product(
            (torch.float, torch.double), (False, True)
        ):
            tkwargs = {"device": self.device, "dtype": dtype}
            intf = (
                Normalize(
                    d=2, bounds=bounds.to(**tkwargs), transform_on_preprocess=True
                )
                if use_intf
                else None
            )
            model, train_X, _ = _get_model_and_training_data(
                input_transform=intf, **tkwargs
            )
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
            if use_intf:
                self.assertIsInstance(model.input_transform, Normalize)

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

            # check that training data has input transform applied
            # check that the train inputs have been transformed and set on the model
            if use_intf:
                self.assertTrue(
                    torch.equal(model.train_inputs[0], model.input_transform(train_X))
                )

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

    def test_MultiTaskGP_fixed_prior(self):
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            model = _get_fixed_prior_model(**tkwargs)
            self.assertIsInstance(model, MultiTaskGP)
            self.assertIsInstance(model.task_covar_module, IndexKernel)
            self.assertIsInstance(
                model.task_covar_module.IndexKernelPrior, LKJCovariancePrior
            )


class TestFixedNoiseMultiTaskGP(BotorchTestCase):
    def test_FixedNoiseMultiTaskGP(self):
        bounds = torch.tensor([[-1.0, 0.0], [1.0, 1.0]])
        for dtype, use_intf in itertools.product(
            (torch.float, torch.double), (False, True)
        ):
            tkwargs = {"device": self.device, "dtype": dtype}
            intf = (
                Normalize(
                    d=2, bounds=bounds.to(**tkwargs), transform_on_preprocess=True
                )
                if use_intf
                else None
            )
            model, train_X, _, _ = _get_fixed_noise_model_and_training_data(
                input_transform=intf, **tkwargs
            )
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
            if use_intf:
                self.assertIsInstance(model.input_transform, Normalize)

            # test model fitting
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=OptimizationWarning)
                mll = fit_gpytorch_model(mll, options={"maxiter": 1}, max_retries=1)

            # check that training data has input transform applied
            # check that the train inputs have been transformed and set on the model
            if use_intf:
                self.assertTrue(
                    torch.equal(model.train_inputs[0], model.input_transform(train_X))
                )

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

            # test input transform

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

    def test_FixedNoiseMultiTaskGP_fixed_prior(self):
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            model = _get_fixed_noise_and_prior_model(**tkwargs)
            self.assertIsInstance(model, FixedNoiseMultiTaskGP)
            self.assertIsInstance(model.task_covar_module, IndexKernel)
            self.assertIsInstance(
                model.task_covar_module.IndexKernelPrior, LKJCovariancePrior
            )

    def test_MultiTaskGP_construct_inputs(self):
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            model, train_X, train_Y = _get_model_and_training_data(**tkwargs)
            training_data = TrainingData(X=train_X, Y=train_Y)
            # Test that task features are required.
            with self.assertRaisesRegex(ValueError, "`task_features` required"):
                model.construct_inputs(training_data)
            # Validate prior config.
            with self.assertRaisesRegex(
                ValueError, ".* only config for LKJ prior is supported"
            ):
                data_dict = model.construct_inputs(
                    training_data,
                    task_features=[0],
                    prior_config={"use_LKJ_prior": False},
                )
            # Validate eta.
            with self.assertRaisesRegex(ValueError, "eta must be a real number"):
                data_dict = model.construct_inputs(
                    training_data,
                    task_features=[0],
                    prior_config={"use_LKJ_prior": True, "eta": "not_number"},
                )
            # Test that presence of `prior` and `prior_config` kwargs at the
            # same time causes error.
            with self.assertRaisesRegex(ValueError, ".* one of `prior` and `prior_"):
                data_dict = model.construct_inputs(
                    training_data,
                    task_features=[0],
                    task_covar_prior=1,
                    prior_config={"use_LKJ_prior": True, "eta": "not_number"},
                )
            data_dict = model.construct_inputs(
                training_data,
                task_features=[0],
                prior_config={"use_LKJ_prior": True, "eta": 0.6},
            )
            self.assertTrue(torch.equal(data_dict["train_X"], train_X))
            self.assertTrue(torch.equal(data_dict["train_Y"], train_Y))
            self.assertEqual(data_dict["task_feature"], 0)
            self.assertIsInstance(data_dict["task_covar_prior"], LKJCovariancePrior)

    def test_FixedNoiseMultiTaskGP_construct_inputs(self):
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            (
                model,
                train_X,
                train_Y,
                train_Yvar,
            ) = _get_fixed_noise_model_and_training_data(**tkwargs)
            td_no_Yvar = TrainingData(X=train_X, Y=train_Y)
            # Test that Yvar is required.
            with self.assertRaisesRegex(ValueError, "Yvar required"):
                model.construct_inputs(td_no_Yvar)
            training_data = TrainingData(X=train_X, Y=train_Y, Yvar=train_Yvar)
            # Test that task features are required.
            with self.assertRaisesRegex(ValueError, "`task_features` required"):
                model.construct_inputs(training_data)
            # Validate prior config.
            with self.assertRaisesRegex(
                ValueError, ".* only config for LKJ prior is supported"
            ):
                data_dict = model.construct_inputs(
                    training_data,
                    task_features=[0],
                    prior_config={"use_LKJ_prior": False},
                )
            data_dict = model.construct_inputs(
                training_data,
                task_features=[0],
                prior_config={"use_LKJ_prior": True, "eta": 0.6},
            )
            self.assertTrue(torch.equal(data_dict["train_X"], train_X))
            self.assertTrue(torch.equal(data_dict["train_Y"], train_Y))
            self.assertTrue(torch.equal(data_dict["train_Yvar"], train_Yvar))
            self.assertEqual(data_dict["task_feature"], 0)
            self.assertIsInstance(data_dict["task_covar_prior"], LKJCovariancePrior)
