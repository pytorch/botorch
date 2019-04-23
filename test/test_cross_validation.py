#! /usr/bin/env python3

import math
import unittest

import torch
from botorch.cross_validation import batch_cross_validation, gen_loo_cv_folds
from botorch.models.gpytorch import GPyTorchModel
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from gpytorch.kernels import MultitaskKernel, RBFKernel, ScaleKernel
from gpytorch.likelihoods import (
    GaussianLikelihood,
    MultitaskGaussianLikelihoodKronecker,
)
from gpytorch.means import ConstantMean, MultitaskMean
from gpytorch.models import ExactGP
from gpytorch.priors import NormalPrior, SmoothedBoxPrior


class CVExactGPModel(ExactGP, GPyTorchModel):
    def __init__(self, train_inputs, train_targets, likelihood, batch_shape=None):
        super().__init__(train_inputs, train_targets, likelihood)
        if batch_shape is None:
            batch_shape = torch.Size([1])
        self.mean_module = ConstantMean(
            batch_shape=batch_shape, prior=SmoothedBoxPrior(-10, 10)
        )
        self.covar_module = ScaleKernel(
            RBFKernel(
                batch_shape=batch_shape,
                lengthscale_prior=NormalPrior(
                    loc=torch.zeros(batch_shape + torch.Size([1, 1])),
                    scale=torch.ones(batch_shape + torch.Size([1, 1])),
                ),
            ),
            batch_shape=batch_shape,
            outputscale_prior=SmoothedBoxPrior(-2, 2),
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class CVGaussianLikelihood(GaussianLikelihood):
    def __init__(self, batch_shape=None):
        if batch_shape is None:
            batch_shape = torch.Size([1])
        super().__init__(
            noise_prior=NormalPrior(
                loc=torch.zeros(batch_shape), scale=torch.ones(batch_shape)
            ),
            batch_shape=batch_shape,
        )


class CVMultitaskExactGPModel(ExactGP, GPyTorchModel):
    def __init__(self, train_inputs, train_targets, likelihood, batch_shape=None):
        super().__init__(train_inputs, train_targets, likelihood)
        if batch_shape is None:
            batch_shape = torch.Size([1])
        self.mean_module = MultitaskMean(
            ConstantMean(batch_shape=batch_shape, prior=SmoothedBoxPrior(-10, 10)),
            num_tasks=2,
        )
        self.covar_module = MultitaskKernel(
            RBFKernel(
                batch_shape=batch_shape,
                lengthscale_prior=NormalPrior(
                    loc=torch.zeros(batch_shape + torch.Size([1, 1])),
                    scale=torch.ones(batch_shape + torch.Size([1, 1])),
                ),
            ),
            num_tasks=2,
            rank=1,
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultitaskMultivariateNormal(mean_x, covar_x)


class CVMultitaskGaussianLikelihood(MultitaskGaussianLikelihoodKronecker):
    def __init__(self, batch_shape=None):
        if batch_shape is None:
            batch_shape = torch.Size([1])
        super().__init__(
            noise_prior=NormalPrior(
                loc=torch.zeros(batch_shape), scale=torch.ones(batch_shape)
            ),
            num_tasks=2,
            batch_shape=batch_shape,
        )


class TestFitBatchCrossValidation(unittest.TestCase):
    def test_single_task_batch_cv(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        train_x = torch.linspace(0, 1, 5, device=device).view(-1, 1)
        train_y = torch.sin(train_x * (2 * math.pi)).view(-1)
        train_y += torch.randn_like(train_y).mul_(0.05)
        cv_folds = gen_loo_cv_folds(train_x, train_y)
        cv_results = batch_cross_validation(
            model_cls=CVExactGPModel,
            likelihood_cls=CVGaussianLikelihood,
            cv_folds=cv_folds,
            fit_args={"options": {"maxiter": 1}},
        )
        # compute MSE
        ((cv_results.observed - cv_results.posterior.mean) ** 2).mean()
        self.assertTrue(cv_results.posterior.mean.shape == torch.Size([5, 1, 1]))

    def test_single_task_batch_cv_cuda(self):
        if torch.cuda.is_available():
            self.test_single_task_batch_cv(cuda=True)

    def test_multi_task_batch_cv(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        train_x = torch.linspace(0, 1, 5, device=device).view(-1, 1)
        train_y = torch.cat(
            [torch.sin(train_x * (2 * math.pi)), torch.cos(train_x * (2 * math.pi))], -1
        )
        train_y += torch.randn_like(train_y).mul_(0.05)
        cv_folds = gen_loo_cv_folds(train_x, train_y)
        cv_results = batch_cross_validation(
            model_cls=CVMultitaskExactGPModel,
            likelihood_cls=CVMultitaskGaussianLikelihood,
            cv_folds=cv_folds,
            fit_args={"options": {"maxiter": 1}},
        )
        # compute MSE
        ((cv_results.observed - cv_results.posterior.mean) ** 2).sum(-1).mean()
        self.assertTrue(cv_results.posterior.mean.shape == torch.Size([5, 1, 2]))
        # compute predicton errors
        (cv_results.posterior.mean - cv_folds.test_y).detach().squeeze(1)
        # compute predictive sems
        torch.diagonal(
            cv_results.posterior.mvn.covariance_matrix, dim1=-2, dim2=-1
        ).sqrt().detach()

    def test_multi_task_batch_cv_cuda(self):
        if torch.cuda.is_available():
            self.test_multi_task_batch_cv(cuda=True)
