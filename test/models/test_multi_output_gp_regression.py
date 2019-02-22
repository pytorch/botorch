#! /usr/bin/env python3

import math
import unittest
from copy import deepcopy

import torch
from botorch import fit_model
from botorch.models import MultiOutputGP
from botorch.models.gp_regression import SingleTaskGP
from botorch.posteriors import GPyTorchPosterior
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import LikelihoodList
from gpytorch.means import ConstantMean
from gpytorch.mlls import SumMarginalLogLikelihood
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.priors import GammaPrior


def _get_random_data(**tkwargs):
    train_x1 = torch.linspace(0, 0.95, 10, **tkwargs) + 0.05 * torch.rand(10, **tkwargs)
    train_x2 = torch.linspace(0, 0.95, 5, **tkwargs) + 0.05 * torch.rand(5, **tkwargs)
    train_y1 = torch.sin(train_x1 * (2 * math.pi)) + 0.2 * torch.randn_like(train_x1)
    train_y2 = torch.cos(train_x2 * (2 * math.pi)) + 0.2 * torch.randn_like(train_x2)
    return train_x1, train_x2, train_y1, train_y2


def _get_model(**tkwargs):
    train_x1, train_x2, train_y1, train_y2 = _get_random_data(**tkwargs)
    model1 = SingleTaskGP(train_X=train_x1, train_Y=train_y1)
    model2 = SingleTaskGP(train_X=train_x2, train_Y=train_y2)
    model = MultiOutputGP(gp_models=[model1, model2])
    return model.to(**tkwargs)


class MultiOutputGPTest(unittest.TestCase):
    def testMultiOutputGP(self, cuda=False):
        for double in (False, True):
            tkwargs = {
                "device": torch.device("cuda") if cuda else torch.device("cpu"),
                "dtype": torch.double if double else torch.float,
            }
            model = _get_model(**tkwargs)
            self.assertIsInstance(model, MultiOutputGP)
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
            mll = fit_model(mll, options={"maxiter": 1})

            # test posterior
            test_x = (torch.tensor([0.25, 0.75]).type_as(model.train_targets[0]),)
            posterior = model.posterior(test_x)
            self.assertIsInstance(posterior, GPyTorchPosterior)
            self.assertIsInstance(posterior.mvn, MultitaskMultivariateNormal)

            # test reinitialization
            train_x1_, train_x2_, train_y1_, train_y2_ = _get_random_data(**tkwargs)
            old_state_dict = deepcopy(model.state_dict())
            model.reinitialize(
                train_Xs=[train_x1_, train_x2_],
                train_Ys=[train_y1_, train_y2_],
                keep_params=True,
            )
            for key, val in model.state_dict().items():
                self.assertEqual(val, old_state_dict[key])
            model.reinitialize(
                train_Xs=[train_x1_, train_x2_],
                train_Ys=[train_y1_, train_y2_],
                keep_params=False,
            )
            model.to(**tkwargs)  # TODO: Fix incorporate this into reinitialize
            self.assertFalse(
                all(
                    val == old_state_dict[key]
                    for key, val in model.state_dict().items()
                )
            )

    def testMultiOutputGPSingle(self, cuda=False):
        tkwargs = {
            "device": torch.device("cuda") if cuda else torch.device("cpu"),
            "dtype": torch.float,
        }
        train_x1, train_x2, train_y1, train_y2 = _get_random_data(**tkwargs)
        model1 = SingleTaskGP(train_X=train_x1, train_Y=train_y1)
        model = MultiOutputGP(gp_models=[model1])
        model.to(**tkwargs)
        test_x = (torch.tensor([0.25, 0.75]).type_as(model.train_targets[0]),)
        posterior = model.posterior(test_x)
        self.assertIsInstance(posterior, GPyTorchPosterior)
        self.assertIsInstance(posterior.mvn, MultivariateNormal)
