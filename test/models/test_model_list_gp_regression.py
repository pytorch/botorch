#! /usr/bin/env python3

import math
import unittest

import torch
from botorch import fit_gpytorch_model
from botorch.models import ModelListGP
from botorch.models.gp_regression import SingleTaskGP
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


def _get_model(n, **tkwargs):
    train_x1, train_x2, train_y1, train_y2 = _get_random_data(n=n, **tkwargs)
    model1 = SingleTaskGP(train_X=train_x1, train_Y=train_y1)
    model2 = SingleTaskGP(train_X=train_x2, train_Y=train_y2)
    model = ModelListGP(gp_models=[model1, model2])
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

            # test model fitting
            mll = SumMarginalLogLikelihood(model.likelihood, model)
            for mll_ in mll.mlls:
                self.assertIsInstance(mll_, ExactMarginalLogLikelihood)
            mll = fit_gpytorch_model(mll, options={"maxiter": 1})

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

    def test_ModelListGP_cuda(self):
        if torch.cuda.is_available():
            self.test_ModelListGP(cuda=True)

    def test_ModelListGPSingle(self, cuda=False):
        tkwargs = {
            "device": torch.device("cuda") if cuda else torch.device("cpu"),
            "dtype": torch.float,
        }
        train_x1, train_x2, train_y1, train_y2 = _get_random_data(n=10, **tkwargs)
        model1 = SingleTaskGP(train_X=train_x1, train_Y=train_y1)
        model = ModelListGP(gp_models=[model1])
        model.to(**tkwargs)
        test_x = (torch.tensor([0.25, 0.75]).type_as(model.train_targets[0]),)
        posterior = model.posterior(test_x)
        self.assertIsInstance(posterior, GPyTorchPosterior)
        self.assertIsInstance(posterior.mvn, MultivariateNormal)

    def test_ModelListGPSingle_cuda(self):
        if torch.cuda.is_available():
            self.test_ModelListGPSingle(cuda=True)
