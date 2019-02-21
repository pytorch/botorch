#! /usr/bin/env python3

import math
import unittest
from copy import deepcopy

import torch
from botorch import fit_model
from botorch.models.constant_noise import ConstantNoiseGP
from botorch.posteriors import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import _GaussianLikelihoodBase
from gpytorch.means import ConstantMean
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.priors import GammaPrior


def _get_random_data(**tkwargs):
    train_x = torch.linspace(0, 0.95, 10, **tkwargs) + 0.05 * torch.rand(10, **tkwargs)
    train_y = torch.sin(train_x * (2 * math.pi)) + 0.2 * torch.randn_like(train_x)
    train_y_se = torch.tensor(0.01, **tkwargs)
    return train_x.view(-1, 1), train_y, train_y_se


def _get_model(**tkwargs):
    train_x, train_y, train_y_se = _get_random_data(**tkwargs)
    model = ConstantNoiseGP(train_X=train_x, train_Y=train_y, train_Y_se=train_y_se)
    return model.to(**tkwargs)


class ConstantNoiseGPTest(unittest.TestCase):
    def testConstantNoiseGP(self, cuda=False):
        for double in (False, True):
            tkwargs = {
                "device": torch.device("cuda") if cuda else torch.device("cpu"),
                "dtype": torch.double if double else torch.float,
            }
            model = _get_model(**tkwargs)
            self.assertIsInstance(model, ConstantNoiseGP)
            self.assertIsInstance(model.likelihood, _GaussianLikelihoodBase)
            self.assertIsInstance(model.mean_module, ConstantMean)
            self.assertIsInstance(model.covar_module, ScaleKernel)
            matern_kernel = model.covar_module.base_kernel
            self.assertIsInstance(matern_kernel, MaternKernel)
            self.assertIsInstance(matern_kernel.lengthscale_prior, GammaPrior)

            # test model fitting
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            mll = fit_model(mll, options={"maxiter": 1})

            # test posterior
            test_x = torch.tensor([[0.25], [0.75]]).type_as(model.train_targets[0])
            posterior_f = model.posterior(test_x)
            self.assertIsInstance(posterior_f, GPyTorchPosterior)
            self.assertIsInstance(posterior_f.mvn, MultivariateNormal)
            posterior_obs = model.posterior(test_x, observation_noise=True)
            self.assertTrue(
                torch.allclose(posterior_f.variance + 0.01, posterior_obs.variance)
            )

            # test reinitialization
            train_x_, train_y_, train_y_se_ = _get_random_data(**tkwargs)
            old_state_dict = deepcopy(model.state_dict())
            model.reinitialize(
                train_X=train_x_,
                train_Y=train_y_,
                train_Y_se=train_y_se_,
                keep_params=True,
            )
            for key, val in model.state_dict().items():
                self.assertEqual(val, old_state_dict[key])
            model.reinitialize(
                train_X=train_x_,
                train_Y=train_y_,
                train_Y_se=train_y_se_,
                keep_params=False,
            )
            model.to(**tkwargs)  # TODO: Fix incorporate this into reinitialize
            self.assertFalse(
                all(
                    val == old_state_dict[key]
                    for key, val in model.state_dict().items()
                )
            )

    def testConstantNoiseGP_cuda(self):
        if torch.cuda.is_available():
            self.testConstantNoiseGP(cuda=True)
