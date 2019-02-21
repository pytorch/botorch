#! /usr/bin/env python3

import math
import unittest

import torch
from botorch import fit_model
from botorch.models.gp_regression import HeteroskedasticSingleTaskGP, SingleTaskGP
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import (
    GaussianLikelihood,
    HeteroskedasticNoise,
    _GaussianLikelihoodBase,
)
from gpytorch.means import ConstantMean
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.priors import GammaPrior

from ..test_fit import NOISE


class SingleTaskGPTest(unittest.TestCase):
    def setUp(self, cuda=False):
        train_x = torch.linspace(0, 1, 10).unsqueeze(1)
        train_y = torch.sin(train_x * (2 * math.pi)).view(-1) + torch.tensor(NOISE)
        model = SingleTaskGP(
            train_x.cuda() if cuda else train_x, train_y.cuda() if cuda else train_y
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_model(mll, options={"maxiter": 1})
        self.model = model

    def testInit(self):
        self.assertIsInstance(self.model.mean_module, ConstantMean)
        self.assertIsInstance(self.model.covar_module, ScaleKernel)
        matern_kernel = self.model.covar_module.base_kernel
        self.assertIsInstance(matern_kernel, MaternKernel)
        self.assertIsInstance(matern_kernel.lengthscale_prior, GammaPrior)

    def testForward(self):
        test_x = torch.tensor([6.0, 7.0, 8.0]).view(-1, 1)
        posterior = self.model(test_x)
        self.assertIsInstance(posterior, MultivariateNormal)

    def testReinitialize(self):
        train_x = torch.linspace(0, 1, 11).unsqueeze(1)
        noise = torch.tensor(NOISE + [0.1])
        train_y = torch.sin(train_x * (2 * math.pi)).view(-1) + noise

        model = self.model

        # check reinitializing while keeping param values
        old_params = dict(model.named_parameters())
        model.reinitialize(train_x, train_y, keep_params=True)
        params = dict(model.named_parameters())
        for p in params:
            self.assertEqual(params[p].item(), old_params[p].item())

        # check reinitializing, resetting param values
        model.reinitialize(train_x, train_y, keep_params=False)
        params = dict(model.named_parameters())
        for p in params:
            self.assertEqual(params[p].item(), 0.0)
        mll = ExactMarginalLogLikelihood(model.likelihood, self.model)
        fit_model(mll)
        # check that some of the parameters changed
        self.assertFalse(all(params[p].item() == 0.0 for p in params))


class HeteroskedasticSingleTaskGPTest(unittest.TestCase):
    def setUp(self, cuda=False):
        train_x = torch.linspace(0, 1, 10).unsqueeze(1)
        train_y = torch.sin(train_x * (2 * math.pi)).view(-1) + torch.tensor(NOISE)
        train_y_sem = 0.1 + 0.1 * torch.rand_like(train_y)
        self.model = HeteroskedasticSingleTaskGP(
            train_x.cuda() if cuda else train_x,
            train_y.cuda() if cuda else train_y,
            train_y_sem.cuda() if cuda else train_y_sem,
        )
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_model(mll, options={"maxiter": 1})

    def testInit(self):
        self.assertIsInstance(self.model.mean_module, ConstantMean)
        self.assertIsInstance(self.model.covar_module, ScaleKernel)
        matern_kernel = self.model.covar_module.base_kernel
        self.assertIsInstance(matern_kernel, MaternKernel)
        self.assertIsInstance(matern_kernel.lengthscale_prior, GammaPrior)
        likelihood = self.model.likelihood
        self.assertIsInstance(likelihood, _GaussianLikelihoodBase)
        self.assertFalse(isinstance(likelihood, GaussianLikelihood))
        self.assertIsInstance(likelihood.noise_covar, HeteroskedasticNoise)

    def testForward(self):
        test_x = torch.tensor([6.0, 7.0, 8.0]).view(-1, 1)
        posterior = self.model(test_x)
        self.assertIsInstance(posterior, MultivariateNormal)

    def testReinitialize(self):
        train_x = torch.linspace(0, 1, 11).unsqueeze(1)
        noise = torch.tensor(NOISE + [0.1])
        train_y = torch.sin(train_x * (2 * math.pi)).view(-1) + noise
        train_y_sem = 0.1 + 0.1 * torch.rand_like(train_y)

        model = self.model

        # check reinitializing while keeping param values
        old_params = dict(model.named_parameters())
        model.reinitialize(train_x, train_y, train_y_sem, keep_params=True)
        params = dict(model.named_parameters())
        for p in params:
            self.assertEqual(params[p].item(), old_params[p].item())

        # check reinitializing, resetting param values
        model.reinitialize(train_x, train_y, train_y_sem, keep_params=False)
        params = dict(model.named_parameters())
        for p in params:
            self.assertEqual(params[p].item(), 0.0)
        mll = ExactMarginalLogLikelihood(model.likelihood, self.model)
        fit_model(mll)
        # check that some of the parameters changed
        self.assertFalse(all(params[p].item() == 0.0 for p in params))
