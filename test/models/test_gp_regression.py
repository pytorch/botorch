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


def _get_random_data(num_outputs, **tkwargs):
    train_x = torch.linspace(0, 0.95, 10, **tkwargs).unsqueeze(-1) + 0.05 * torch.rand(
        10, 1, **tkwargs
    )
    train_y = torch.sin(train_x * (2 * math.pi)) + 0.2 * torch.randn(
        10, num_outputs, **tkwargs
    )

    if num_outputs == 1:
        train_y = train_y.squeeze(-1)
    return train_x, train_y


class TestSingleTaskGP(unittest.TestCase):
    def _get_model(self, num_outputs, likelihood=None, **tkwargs):
        train_x, train_y = _get_random_data(num_outputs=num_outputs, **tkwargs)
        model = SingleTaskGP(train_X=train_x, train_Y=train_y, likelihood=likelihood)
        mll = ExactMarginalLogLikelihood(model.likelihood, model).to(**tkwargs)
        fit_gpytorch_model(mll, options={"maxiter": 1})
        return model

    def test_SingleTaskGP(self, cuda=False):
        for num_outputs in (1, 2):
            for double in (False, True):
                tkwargs = {
                    "device": torch.device("cuda") if cuda else torch.device("cpu"),
                    "dtype": torch.double if double else torch.float,
                }
                model = self._get_model(num_outputs=num_outputs, **tkwargs)
                # test init
                self.assertIsInstance(model.mean_module, ConstantMean)
                self.assertIsInstance(model.covar_module, ScaleKernel)
                matern_kernel = model.covar_module.base_kernel
                self.assertIsInstance(matern_kernel, MaternKernel)
                self.assertIsInstance(matern_kernel.lengthscale_prior, GammaPrior)

                # Test forward
                test_x = torch.tensor([6.0, 7.0, 8.0], **tkwargs).view(-1, 1)
                posterior = model(test_x)
                self.assertIsInstance(posterior, MultivariateNormal)

                # Test reinitialize
                train_x, train_y = _get_random_data(num_outputs=num_outputs, **tkwargs)

                # check reinitializing while keeping param values
                old_params = deepcopy(dict(model.named_parameters()))
                model.reinitialize(train_x, train_y, keep_params=True)
                params = dict(model.named_parameters())
                for p in params:
                    self.assertTrue(torch.equal(params[p], old_params[p]))

                # check reinitializing, resetting param values
                model.reinitialize(train_x, train_y, keep_params=False)
                params = dict(model.named_parameters())
                for p in params:
                    if p == "likelihood.noise_covar.raw_noise":
                        self.assertTrue(
                            torch.allclose(
                                params[p].detach(), torch.tensor([22.0], **tkwargs)
                            )
                        )
                    else:
                        self.assertTrue(
                            torch.allclose(
                                params[p].detach(), torch.tensor([0.0], **tkwargs)
                            )
                        )
                    self.assertEqual(params[p].numel(), num_outputs)

                # check reinitializing while reseting param values and using custom
                # likelihood
                batch_shape = (
                    torch.Size([num_outputs]) if num_outputs > 1 else torch.Size()
                )
                model = SingleTaskGP(
                    train_X=train_x,
                    train_Y=train_y,
                    likelihood=GaussianLikelihood(batch_shape=batch_shape),
                )
                old_params = deepcopy(dict(model.named_parameters()))
                mll = ExactMarginalLogLikelihood(model.likelihood, model)
                fit_gpytorch_model(mll, options={"maxiter": 1})
                train_x, train_y = _get_random_data(num_outputs=num_outputs, **tkwargs)
                model.reinitialize(train_x, train_y, keep_params=False)
                params = dict(model.named_parameters())
                for p in params:
                    self.assertTrue(torch.equal(params[p], old_params[p]))

                # test posterior
                # test non batch evaluation
                X = torch.rand(3, 1, **tkwargs)
                posterior = model.posterior(X)
                self.assertIsInstance(posterior, GPyTorchPosterior)
                self.assertEqual(posterior.mean.shape, (3, num_outputs))
                # test batch evaluation
                X = torch.rand(2, 3, 1, **tkwargs)
                posterior = model.posterior(X)
                self.assertIsInstance(posterior, GPyTorchPosterior)
                self.assertEqual(posterior.mean.shape, (2, 3, num_outputs))

    def test_SingleTaskGP_cuda(self):
        if torch.cuda.is_available():
            self.test_SingleTaskGP(cuda=True)


class TestFixedNoiseGP(unittest.TestCase):
    def _get_model(self, num_outputs, **tkwargs):
        train_x, train_y = _get_random_data(num_outputs=num_outputs, **tkwargs)
        train_yvar = torch.full_like(train_y, 0.01)
        model = FixedNoiseGP(train_X=train_x, train_Y=train_y, train_Yvar=train_yvar)
        return model.to(**tkwargs)

    def test_FixedNoiseGP(self, cuda=False):
        for num_outputs in (1, 2):
            for double in (False, True):
                tkwargs = {
                    "device": torch.device("cuda") if cuda else torch.device("cpu"),
                    "dtype": torch.double if double else torch.float,
                }
                model = self._get_model(num_outputs=num_outputs, **tkwargs)
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
                #     torch.allclose(
                #         posterior_f.variance + 0.01,
                #         posterior_obs.variance
                #     )
                # )

                # test reinitialization
                train_x_, train_y_ = _get_random_data(
                    num_outputs=num_outputs, **tkwargs
                )
                train_yvar_ = torch.full_like(train_y_, 0.01)
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

                # test posterior
                # test non batch
                X = torch.rand(3, 1, **tkwargs)
                posterior = model.posterior(X)
                self.assertIsInstance(posterior, GPyTorchPosterior)
                self.assertEqual(posterior.mean.shape, (3, num_outputs))
                # test batch
                X = torch.rand(2, 3, 1, **tkwargs)
                posterior = model.posterior(X)
                self.assertIsInstance(posterior, GPyTorchPosterior)
                self.assertEqual(posterior.mean.shape, (2, 3, num_outputs))

    def test_FixedNoiseGP_cuda(self):
        if torch.cuda.is_available():
            self.test_FixedNoiseGP(cuda=True)


class TestHeteroskedasticSingleTaskGP(unittest.TestCase):
    def _get_model(self, num_outputs, **tkwargs):
        train_x, train_y = _get_random_data(num_outputs=num_outputs, **tkwargs)
        train_yvar = (0.1 + 0.1 * torch.rand_like(train_y)) ** 2
        model = HeteroskedasticSingleTaskGP(
            train_X=train_x, train_Y=train_y, train_Yvar=train_yvar
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model).to(**tkwargs)
        fit_gpytorch_model(mll, options={"maxiter": 1})
        return model

    def test_HeterskedasticSingleTaskGP(self, cuda=False):
        for num_outputs in (1, 2):
            for double in (False, True):
                tkwargs = {
                    "device": torch.device("cuda") if cuda else torch.device("cpu"),
                    "dtype": torch.double if double else torch.float,
                }
                model = self._get_model(num_outputs=num_outputs, **tkwargs)
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
                test_x = torch.tensor([6.0, 7.0, 8.0], **tkwargs).view(-1, 1)
                posterior = model(test_x)
                self.assertIsInstance(posterior, MultivariateNormal)

                # test reinitialize
                train_x, train_y = _get_random_data(num_outputs=num_outputs, **tkwargs)
                train_yvar = (0.1 + 0.1 * torch.rand_like(train_y)) ** 2

                # check reinitializing while keeping param values
                old_params = dict(model.named_parameters())
                model.reinitialize(train_x, train_y, train_yvar, keep_params=True)
                params = dict(model.named_parameters())
                for p in params:
                    self.assertTrue(torch.equal(params[p], old_params[p]))

                # check reinitializing, resetting param values
                model.reinitialize(train_x, train_y, train_yvar, keep_params=False)
                params = dict(model.named_parameters())
                for p in params:
                    self.assertTrue(
                        torch.allclose(params[p], torch.tensor([0.0], **tkwargs))
                    )
                    self.assertEqual(params[p].numel(), num_outputs)
                mll = ExactMarginalLogLikelihood(model.likelihood, model).to(**tkwargs)
                fit_gpytorch_model(mll, options={"maxiter": 1})
                # check that some of the parameters changed
                self.assertFalse(
                    all(
                        torch.allclose(params[p], torch.tensor([0.0], **tkwargs))
                        for p in params
                    )
                )

                # test posterior
                # test non batch
                X = torch.rand(3, 1, **tkwargs)
                posterior = model.posterior(X)
                self.assertIsInstance(posterior, GPyTorchPosterior)
                self.assertEqual(posterior.mean.shape, (3, num_outputs))
                # test batch
                X = torch.rand(2, 3, 1, **tkwargs)
                posterior = model.posterior(X)
                self.assertIsInstance(posterior, GPyTorchPosterior)
                self.assertEqual(posterior.mean.shape, (2, 3, num_outputs))

    def test_HeterskedasticSingleTaskGP_cuda(self):
        if torch.cuda.is_available():
            self.test_HeterskedasticSingleTaskGP(cuda=True)
