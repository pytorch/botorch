#! /usr/bin/env python3

import unittest

import torch
from botorch.models.gpytorch import GPyTorchModel
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP


class SimpleGPyTorchModel(ExactGP, GPyTorchModel):
    def __init__(self, train_X, train_Y):
        likelihood = GaussianLikelihood()
        super().__init__(train_X, train_Y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class TestGPyTorchModel(unittest.TestCase):
    def test_gpytorch_model(self):
        train_X = torch.rand(5, 1)
        train_Y = torch.sin(train_X.squeeze())
        # basic test
        model = SimpleGPyTorchModel(train_X, train_Y)
        test_X = torch.rand(2, 1)
        posterior = model.posterior(test_X)
        self.assertIsInstance(posterior, GPyTorchPosterior)
        self.assertEqual(posterior.mean.shape, torch.Size([2, 1]))
        # test observation noise
        posterior = model.posterior(test_X, observation_noise=True)
        self.assertIsInstance(posterior, GPyTorchPosterior)
        self.assertEqual(posterior.mean.shape, torch.Size([2, 1]))
