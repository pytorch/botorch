#! /usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import math
import unittest
import warnings

import torch
from botorch.exceptions.warnings import OptimizationWarning
from botorch.fit import fit_gpytorch_model
from botorch.models import ModelListGP
from botorch.models.gp_regression import FixedNoiseGP, SingleTaskGP
from botorch.posteriors import GPyTorchPosterior
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import LikelihoodList
from gpytorch.means import ConstantMean
from gpytorch.mlls import SumMarginalLogLikelihood
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.priors import GammaPrior


def _get_random_data(n, **tkwargs):
    train_x1 = torch.linspace(0, 0.95, n + 1, **tkwargs) + 0.05 * torch.rand(
        n + 1, **tkwargs
    )
    train_x2 = torch.linspace(0, 0.95, n, **tkwargs) + 0.05 * torch.rand(n, **tkwargs)
    train_y1 = torch.sin(train_x1 * (2 * math.pi)) + 0.2 * torch.randn_like(train_x1)
    train_y2 = torch.cos(train_x2 * (2 * math.pi)) + 0.2 * torch.randn_like(train_x2)
    return train_x1.unsqueeze(-1), train_x2.unsqueeze(-1), train_y1, train_y2


def _get_model(n, fixed_noise=False, **tkwargs):
    train_x1, train_x2, train_y1, train_y2 = _get_random_data(n=n, **tkwargs)
    if fixed_noise:
        train_y1_var = 0.1 + 0.1 * torch.rand_like(train_y1, **tkwargs)
        train_y2_var = 0.1 + 0.1 * torch.rand_like(train_y2, **tkwargs)
        model1 = FixedNoiseGP(
            train_X=train_x1, train_Y=train_y1, train_Yvar=train_y1_var
        )
        model2 = FixedNoiseGP(
            train_X=train_x2, train_Y=train_y2, train_Yvar=train_y2_var
        )
    else:
        model1 = SingleTaskGP(train_X=train_x1, train_Y=train_y1)
        model2 = SingleTaskGP(train_X=train_x2, train_Y=train_y2)
    model = ModelListGP(model1, model2)
    return model.to(**tkwargs)


class TestModelListGP(unittest.TestCase):
    def test_ModelListGP(self, cuda=False):
        for double in (False, True):
            tkwargs = {
                "device": torch.device("cuda") if cuda else torch.device("cpu"),
                "dtype": torch.double if double else torch.float,
            }
            model = _get_model(n=10, **tkwargs)
            self.assertIsInstance(model, ModelListGP)
            self.assertIsInstance(model.likelihood, LikelihoodList)
            for m in model.models:
                self.assertIsInstance(m.mean_module, ConstantMean)
                self.assertIsInstance(m.covar_module, ScaleKernel)
                matern_kernel = m.covar_module.base_kernel
                self.assertIsInstance(matern_kernel, MaternKernel)
                self.assertIsInstance(matern_kernel.lengthscale_prior, GammaPrior)

            # test constructing likelihood wrapper
            mll = SumMarginalLogLikelihood(model.likelihood, model)
            for mll_ in mll.mlls:
                self.assertIsInstance(mll_, ExactMarginalLogLikelihood)

            # test model fitting (sequential)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=OptimizationWarning)
                mll = fit_gpytorch_model(mll, options={"maxiter": 1}, max_retries=1)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=OptimizationWarning)
                # test model fitting (joint)
                mll = fit_gpytorch_model(
                    mll, options={"maxiter": 1}, max_retries=1, sequential=False
                )

            # test posterior
            test_x = torch.tensor([[0.25], [0.75]], **tkwargs)
            posterior = model.posterior(test_x)
            self.assertIsInstance(posterior, GPyTorchPosterior)
            self.assertIsInstance(posterior.mvn, MultitaskMultivariateNormal)

            # test observation_noise
            posterior = model.posterior(test_x, observation_noise=True)
            self.assertIsInstance(posterior, GPyTorchPosterior)
            self.assertIsInstance(posterior.mvn, MultitaskMultivariateNormal)

            # test output_indices
            posterior = model.posterior(
                test_x, output_indices=[0], observation_noise=True
            )
            self.assertIsInstance(posterior, GPyTorchPosterior)
            self.assertIsInstance(posterior.mvn, MultivariateNormal)

            # test condition_on_observations
            f_x = torch.rand(2, 1, **tkwargs)
            f_y = torch.rand(2, 2, **tkwargs)
            cm = model.condition_on_observations(f_x, f_y)
            self.assertIsInstance(cm, ModelListGP)

            # test condition_on_observations batched
            f_x = torch.rand(3, 2, 1, **tkwargs)
            f_y = torch.rand(3, 2, 2, **tkwargs)
            cm = model.condition_on_observations(f_x, f_y)
            self.assertIsInstance(cm, ModelListGP)

            # test condition_on_observations batched (fast fantasies)
            f_x = torch.rand(2, 1, **tkwargs)
            f_y = torch.rand(3, 2, 2, **tkwargs)
            cm = model.condition_on_observations(f_x, f_y)
            self.assertIsInstance(cm, ModelListGP)

            # test condition_on_observations (incorrect input shape error)
            with self.assertRaises(ValueError):
                model.condition_on_observations(f_x, torch.rand(3, 2, 3, **tkwargs))

    def test_ModelListGP_cuda(self):
        if torch.cuda.is_available():
            self.test_ModelListGP(cuda=True)

    def test_ModelListGP_fixed_noise(self, cuda=False):
        for double in (False, True):
            tkwargs = {
                "device": torch.device("cuda") if cuda else torch.device("cpu"),
                "dtype": torch.double if double else torch.float,
            }
            model = _get_model(n=10, fixed_noise=True, **tkwargs)
            self.assertIsInstance(model, ModelListGP)
            self.assertIsInstance(model.likelihood, LikelihoodList)
            for m in model.models:
                self.assertIsInstance(m.mean_module, ConstantMean)
                self.assertIsInstance(m.covar_module, ScaleKernel)
                matern_kernel = m.covar_module.base_kernel
                self.assertIsInstance(matern_kernel, MaternKernel)
                self.assertIsInstance(matern_kernel.lengthscale_prior, GammaPrior)

            # test model fitting
            mll = SumMarginalLogLikelihood(model.likelihood, model)
            for mll_ in mll.mlls:
                self.assertIsInstance(mll_, ExactMarginalLogLikelihood)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=OptimizationWarning)
                mll = fit_gpytorch_model(mll, options={"maxiter": 1}, max_retries=1)

            # test posterior
            test_x = torch.tensor([[0.25], [0.75]], **tkwargs)
            posterior = model.posterior(test_x)
            self.assertIsInstance(posterior, GPyTorchPosterior)
            self.assertIsInstance(posterior.mvn, MultitaskMultivariateNormal)

            # test output_indices
            posterior = model.posterior(
                test_x, output_indices=[0], observation_noise=True
            )
            self.assertIsInstance(posterior, GPyTorchPosterior)
            self.assertIsInstance(posterior.mvn, MultivariateNormal)

            # test condition_on_observations
            f_x = torch.rand(2, 1, **tkwargs)
            f_y = torch.rand(2, 2, **tkwargs)
            noise = 0.1 + 0.1 * torch.rand_like(f_y)
            cm = model.condition_on_observations(f_x, f_y, noise=noise)
            self.assertIsInstance(cm, ModelListGP)

            # test condition_on_observations batched
            f_x = torch.rand(3, 2, 1, **tkwargs)
            f_y = torch.rand(3, 2, 2, **tkwargs)
            noise = 0.1 + 0.1 * torch.rand_like(f_y)
            cm = model.condition_on_observations(f_x, f_y, noise=noise)
            self.assertIsInstance(cm, ModelListGP)

            # test condition_on_observations batched (fast fantasies)
            f_x = torch.rand(2, 1, **tkwargs)
            f_y = torch.rand(3, 2, 2, **tkwargs)
            noise = 0.1 + 0.1 * torch.rand(2, 2, **tkwargs)
            cm = model.condition_on_observations(f_x, f_y, noise=noise)
            self.assertIsInstance(cm, ModelListGP)

            # test condition_on_observations (incorrect input shape error)
            with self.assertRaises(ValueError):
                model.condition_on_observations(
                    f_x, torch.rand(3, 2, 3, **tkwargs), noise=noise
                )
            # test condition_on_observations (incorrect noise shape error)
            with self.assertRaises(ValueError):
                model.condition_on_observations(
                    f_x, f_y, noise=torch.rand(2, 3, **tkwargs)
                )

    def test_ModelListGP_fixed_noise_cuda(self):
        if torch.cuda.is_available():
            self.test_ModelListGP_fixed_noise(cuda=True)

    def test_ModelListGP_single(self, cuda=False):
        tkwargs = {
            "device": torch.device("cuda") if cuda else torch.device("cpu"),
            "dtype": torch.float,
        }
        train_x1, train_x2, train_y1, train_y2 = _get_random_data(n=10, **tkwargs)
        model1 = SingleTaskGP(train_X=train_x1, train_Y=train_y1)
        model = ModelListGP(model1)
        model.to(**tkwargs)
        test_x = (torch.tensor([0.25, 0.75]).type_as(model.train_targets[0]),)
        posterior = model.posterior(test_x)
        self.assertIsInstance(posterior, GPyTorchPosterior)
        self.assertIsInstance(posterior.mvn, MultivariateNormal)

    def test_ModelListGP_single_cuda(self):
        if torch.cuda.is_available():
            self.test_ModelListGP_single(cuda=True)
