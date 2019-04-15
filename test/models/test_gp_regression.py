#! /usr/bin/env python3

import math
import unittest
from copy import deepcopy

import torch
from botorch import fit_gpytorch_model
from botorch.models.gp_regression import (
    FixedNoiseGP,
    HeteroskedasticSingleTaskGP,
    SingleTaskGP,
)
from botorch.posteriors import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import (
    FixedNoiseGaussianLikelihood,
    GaussianLikelihood,
    HeteroskedasticNoise,
    _GaussianLikelihoodBase,
)
from gpytorch.means import ConstantMean
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.priors import GammaPrior

from ..test_fit import NOISE


class TestSingleTaskGP(unittest.TestCase):
    def setUp(self, cuda=False):
        train_x = torch.linspace(0, 1, 10).unsqueeze(1)
        train_y = torch.sin(train_x * (2 * math.pi)).view(-1) + torch.tensor(NOISE)
        model = SingleTaskGP(
            train_x.cuda() if cuda else train_x, train_y.cuda() if cuda else train_y
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll, options={"maxiter": 1})
        self.model = model

    def test_Init(self):
        self.assertIsInstance(self.model.mean_module, ConstantMean)
        self.assertIsInstance(self.model.covar_module, ScaleKernel)
        matern_kernel = self.model.covar_module.base_kernel
        self.assertIsInstance(matern_kernel, MaternKernel)
        self.assertIsInstance(matern_kernel.lengthscale_prior, GammaPrior)

    def test_Forward(self):
        test_x = torch.tensor([6.0, 7.0, 8.0]).view(-1, 1)
        posterior = self.model(test_x)
        self.assertIsInstance(posterior, MultivariateNormal)

    def test_Reinitialize(self):
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
        fit_gpytorch_model(mll, options={"maxiter": 1})
        # check that some of the parameters changed
        self.assertFalse(all(params[p].item() == 0.0 for p in params))


class TestFixedNoiseGP(unittest.TestCase):
    def _get_random_data(self, **tkwargs):
        train_x = torch.linspace(0, 0.95, 10, **tkwargs) + 0.05 * torch.rand(
            10, **tkwargs
        )
        train_y = torch.sin(train_x * (2 * math.pi)) + 0.2 * torch.randn_like(train_x)
        train_yvar = torch.full_like(train_y, 0.01)
        return train_x.view(-1, 1), train_y, train_yvar

    def _get_model(self, **tkwargs):
        train_x, train_y, train_yvar = self._get_random_data(**tkwargs)
        model = FixedNoiseGP(train_X=train_x, train_Y=train_y, train_Yvar=train_yvar)
        return model.to(**tkwargs)

    def test_FixedNoiseGP(self, cuda=False):
        for double in (False, True):
            tkwargs = {
                "device": torch.device("cuda") if cuda else torch.device("cpu"),
                "dtype": torch.double if double else torch.float,
            }
            model = self._get_model(**tkwargs)
            self.assertIsInstance(model, FixedNoiseGP)
            self.assertIsInstance(model.likelihood, FixedNoiseGaussianLikelihood)
            self.assertIsInstance(model.mean_module, ConstantMean)
            self.assertIsInstance(model.covar_module, ScaleKernel)
            matern_kernel = model.covar_module.base_kernel
            self.assertIsInstance(matern_kernel, MaternKernel)
            self.assertIsInstance(matern_kernel.lengthscale_prior, GammaPrior)

            # test model fitting
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            mll = fit_gpytorch_model(mll, options={"maxiter": 1})

            # test posterior
            test_x = torch.tensor([[0.25], [0.75]]).to(**tkwargs)
            posterior_f = model.posterior(test_x)
            self.assertIsInstance(posterior_f, GPyTorchPosterior)
            self.assertIsInstance(posterior_f.mvn, MultivariateNormal)
            # TODO: Pass observation noise into posterior
            # posterior_obs = model.posterior(test_x, observation_noise=True)
            # self.assertTrue(
            #     torch.allclose(posterior_f.variance + 0.01, posterior_obs.variance)
            # )

            # test reinitialization
            train_x_, train_y_, train_yvar_ = self._get_random_data(**tkwargs)
            old_state_dict = deepcopy(model.state_dict())
            model.reinitialize(
                train_X=train_x_,
                train_Y=train_y_,
                train_Yvar=train_yvar_,
                keep_params=True,
            )
            for key, val in model.state_dict().items():
                self.assertTrue(torch.equal(val, old_state_dict[key]))
            model.reinitialize(
                train_X=train_x_,
                train_Y=train_y_,
                train_Yvar=train_yvar_,
                keep_params=False,
            )
            self.assertFalse(
                all(
                    torch.equal(val, old_state_dict[key])
                    for key, val in model.state_dict().items()
                )
            )

    def test_FixedNoiseGP_cuda(self):
        if torch.cuda.is_available():
            self.test_FixedNoiseGP(cuda=True)


class TestHeteroskedasticSingleTaskGP(unittest.TestCase):
    def setUp(self, cuda=False):
        train_x = torch.linspace(0, 1, 10).unsqueeze(1)
        train_y = torch.sin(train_x * (2 * math.pi)).view(-1) + torch.tensor(NOISE)
        train_yvar = (0.1 + 0.1 * torch.rand_like(train_y)) ** 2
        self.model = HeteroskedasticSingleTaskGP(
            train_x.cuda() if cuda else train_x,
            train_y.cuda() if cuda else train_y,
            train_yvar.cuda() if cuda else train_yvar,
        )
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_model(mll, options={"maxiter": 1})

    def test_Init(self):
        self.assertIsInstance(self.model.mean_module, ConstantMean)
        self.assertIsInstance(self.model.covar_module, ScaleKernel)
        matern_kernel = self.model.covar_module.base_kernel
        self.assertIsInstance(matern_kernel, MaternKernel)
        self.assertIsInstance(matern_kernel.lengthscale_prior, GammaPrior)
        likelihood = self.model.likelihood
        self.assertIsInstance(likelihood, _GaussianLikelihoodBase)
        self.assertFalse(isinstance(likelihood, GaussianLikelihood))
        self.assertIsInstance(likelihood.noise_covar, HeteroskedasticNoise)

    def test_Forward(self):
        test_x = torch.tensor([6.0, 7.0, 8.0]).view(-1, 1)
        posterior = self.model(test_x)
        self.assertIsInstance(posterior, MultivariateNormal)

    def test_Reinitialize(self):
        train_x = torch.linspace(0, 1, 11).unsqueeze(1)
        noise = torch.tensor(NOISE + [0.1])
        train_y = torch.sin(train_x * (2 * math.pi)).view(-1) + noise
        train_yvar = (0.1 + 0.1 * torch.rand_like(train_y)) ** 2

        model = self.model

        # check reinitializing while keeping param values
        old_params = dict(model.named_parameters())
        model.reinitialize(train_x, train_y, train_yvar, keep_params=True)
        params = dict(model.named_parameters())
        for p in params:
            self.assertEqual(params[p].item(), old_params[p].item())

        # check reinitializing, resetting param values
        model.reinitialize(train_x, train_y, train_yvar, keep_params=False)
        params = dict(model.named_parameters())
        for p in params:
            self.assertEqual(params[p].item(), 0.0)
        mll = ExactMarginalLogLikelihood(model.likelihood, self.model)
        fit_gpytorch_model(mll, options={"maxiter": 1})
        # check that some of the parameters changed
        self.assertFalse(all(params[p].item() == 0.0 for p in params))
