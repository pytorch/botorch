#! /usr/bin/env python3

import math
import unittest

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


class TestSingleTaskGP(unittest.TestCase):
    def _get_model(self, batch_shape, num_outputs, likelihood=None, **tkwargs):
        train_x, train_y = _get_random_data(
            batch_shape=batch_shape, num_outputs=num_outputs, **tkwargs
        )
        model = SingleTaskGP(train_X=train_x, train_Y=train_y, likelihood=likelihood)
        mll = ExactMarginalLogLikelihood(model.likelihood, model).to(**tkwargs)
        fit_gpytorch_model(mll, options={"maxiter": 1})
        return model

    def test_SingleTaskGP(self, cuda=False):
        for batch_shape in (torch.Size([]), torch.Size([2])):
            for num_outputs in (1, 2):
                for double in (False, True):
                    tkwargs = {
                        "device": torch.device("cuda") if cuda else torch.device("cpu"),
                        "dtype": torch.double if double else torch.float,
                    }
                    model = self._get_model(
                        batch_shape=batch_shape, num_outputs=num_outputs, **tkwargs
                    )
                    # test init
                    self.assertIsInstance(model.mean_module, ConstantMean)
                    self.assertIsInstance(model.covar_module, ScaleKernel)
                    matern_kernel = model.covar_module.base_kernel
                    self.assertIsInstance(matern_kernel, MaternKernel)
                    self.assertIsInstance(matern_kernel.lengthscale_prior, GammaPrior)

                    # Test forward
                    test_x = torch.rand(batch_shape + torch.Size([3, 1]), **tkwargs)
                    posterior = model(test_x)
                    self.assertIsInstance(posterior, MultivariateNormal)

                    # test param sizes
                    params = dict(model.named_parameters())
                    for p in params:
                        self.assertEqual(
                            params[p].numel(),
                            num_outputs * torch.tensor(batch_shape).prod().item(),
                        )

                    # test posterior
                    # test non batch evaluation
                    X = torch.rand(batch_shape + torch.Size([3, 1]), **tkwargs)
                    posterior = model.posterior(X)
                    self.assertIsInstance(posterior, GPyTorchPosterior)
                    self.assertEqual(
                        posterior.mean.shape, batch_shape + torch.Size([3, num_outputs])
                    )
                    # test batch evaluation
                    X = torch.rand(
                        torch.Size([2]) + batch_shape + torch.Size([3, 1]), **tkwargs
                    )
                    posterior = model.posterior(X)
                    self.assertIsInstance(posterior, GPyTorchPosterior)
                    self.assertEqual(
                        posterior.mean.shape,
                        torch.Size([2]) + batch_shape + torch.Size([3, num_outputs]),
                    )

    def test_SingleTaskGP_cuda(self):
        if torch.cuda.is_available():
            self.test_SingleTaskGP(cuda=True)


class TestFixedNoiseGP(unittest.TestCase):
    def _get_model(self, batch_shape, num_outputs, n, **tkwargs):
        train_x, train_y = _get_random_data(
            batch_shape=batch_shape, num_outputs=num_outputs, n=n, **tkwargs
        )
        train_yvar = torch.full_like(train_y, 0.01)
        model = FixedNoiseGP(train_X=train_x, train_Y=train_y, train_Yvar=train_yvar)
        return model.to(**tkwargs)

    def test_FixedNoiseGP(self, cuda=False):
        for batch_shape in (torch.Size([]), torch.Size([2])):
            for num_outputs in (1, 2):
                for double in (False, True):
                    tkwargs = {
                        "device": torch.device("cuda") if cuda else torch.device("cpu"),
                        "dtype": torch.double if double else torch.float,
                    }
                    model = self._get_model(
                        batch_shape=batch_shape,
                        num_outputs=num_outputs,
                        n=10,
                        **tkwargs
                    )
                    self.assertIsInstance(model, FixedNoiseGP)
                    self.assertIsInstance(
                        model.likelihood, FixedNoiseGaussianLikelihood
                    )
                    self.assertIsInstance(model.mean_module, ConstantMean)
                    self.assertIsInstance(model.covar_module, ScaleKernel)
                    matern_kernel = model.covar_module.base_kernel
                    self.assertIsInstance(matern_kernel, MaternKernel)
                    self.assertIsInstance(matern_kernel.lengthscale_prior, GammaPrior)

                    # test model fitting
                    mll = ExactMarginalLogLikelihood(model.likelihood, model)
                    mll = fit_gpytorch_model(mll, options={"maxiter": 1})

                    # Test forward
                    test_x = torch.rand(batch_shape + torch.Size([3, 1]), **tkwargs)
                    posterior = model(test_x)
                    self.assertIsInstance(posterior, MultivariateNormal)

                    # TODO: Pass observation noise into posterior
                    # posterior_obs = model.posterior(test_x, observation_noise=True)
                    # self.assertTrue(
                    #     torch.allclose(
                    #         posterior_f.variance + 0.01,
                    #         posterior_obs.variance
                    #     )
                    # )

                    # test posterior
                    # test non batch evaluation
                    X = torch.rand(batch_shape + torch.Size([3, 1]), **tkwargs)
                    posterior = model.posterior(X)
                    self.assertIsInstance(posterior, GPyTorchPosterior)
                    self.assertEqual(
                        posterior.mean.shape, batch_shape + torch.Size([3, num_outputs])
                    )
                    # test batch evaluation
                    X = torch.rand(
                        torch.Size([2]) + batch_shape + torch.Size([3, 1]), **tkwargs
                    )
                    posterior = model.posterior(X)
                    self.assertIsInstance(posterior, GPyTorchPosterior)
                    self.assertEqual(
                        posterior.mean.shape,
                        torch.Size([2]) + batch_shape + torch.Size([3, num_outputs]),
                    )

    def test_FixedNoiseGP_cuda(self):
        if torch.cuda.is_available():
            self.test_FixedNoiseGP(cuda=True)


class TestHeteroskedasticSingleTaskGP(unittest.TestCase):
    def _get_model(self, batch_shape, num_outputs, **tkwargs):
        train_x, train_y = _get_random_data(
            batch_shape=batch_shape, num_outputs=num_outputs, **tkwargs
        )
        train_yvar = (0.1 + 0.1 * torch.rand_like(train_y)) ** 2
        model = HeteroskedasticSingleTaskGP(
            train_X=train_x, train_Y=train_y, train_Yvar=train_yvar
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model).to(**tkwargs)
        fit_gpytorch_model(mll, options={"maxiter": 1})
        return model

    def test_HeterskedasticSingleTaskGP(self, cuda=False):
        for batch_shape in (torch.Size([]), torch.Size([2])):
            for num_outputs in (1, 2):
                for double in (False, True):
                    tkwargs = {
                        "device": torch.device("cuda") if cuda else torch.device("cpu"),
                        "dtype": torch.double if double else torch.float,
                    }
                    model = self._get_model(
                        batch_shape=batch_shape, num_outputs=num_outputs, **tkwargs
                    )
                    # test init
                    self.assertIsInstance(model.mean_module, ConstantMean)
                    self.assertIsInstance(model.covar_module, ScaleKernel)
                    matern_kernel = model.covar_module.base_kernel
                    self.assertIsInstance(matern_kernel, MaternKernel)
                    self.assertIsInstance(matern_kernel.lengthscale_prior, GammaPrior)
                    likelihood = model.likelihood
                    self.assertIsInstance(likelihood, _GaussianLikelihoodBase)
                    self.assertFalse(isinstance(likelihood, GaussianLikelihood))
                    self.assertIsInstance(likelihood.noise_covar, HeteroskedasticNoise)

                    # test forward
                    test_x = torch.rand(batch_shape + torch.Size([3, 1]), **tkwargs)
                    posterior = model(test_x)
                    self.assertIsInstance(posterior, MultivariateNormal)

                    # check param sizes
                    params = dict(model.named_parameters())
                    for p in params:
                        self.assertEqual(
                            params[p].numel(),
                            num_outputs * torch.tensor(batch_shape).prod().item(),
                        )

                    # test posterior
                    # test non batch evaluation
                    X = torch.rand(batch_shape + torch.Size([3, 1]), **tkwargs)
                    posterior = model.posterior(X)
                    self.assertIsInstance(posterior, GPyTorchPosterior)
                    self.assertEqual(
                        posterior.mean.shape, batch_shape + torch.Size([3, num_outputs])
                    )
                    # test batch evaluation
                    X = torch.rand(
                        torch.Size([2]) + batch_shape + torch.Size([3, 1]), **tkwargs
                    )
                    posterior = model.posterior(X)
                    self.assertIsInstance(posterior, GPyTorchPosterior)
                    self.assertEqual(
                        posterior.mean.shape,
                        torch.Size([2]) + batch_shape + torch.Size([3, num_outputs]),
                    )

    def test_HeterskedasticSingleTaskGP_cuda(self):
        if torch.cuda.is_available():
            self.test_HeterskedasticSingleTaskGP(cuda=True)
