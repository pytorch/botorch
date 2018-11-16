#! /usr/bin/env python3

import math
import unittest

import torch
from botorch import fit_model
from botorch.models.gp_regression import SingleTaskGP
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.priors import GammaPrior

from ..test_fit import NOISE


class SingleTaskGPTest(unittest.TestCase):
    def setUp(self, cuda=False):
        train_x = torch.linspace(0, 1, 10).unsqueeze(1)
        train_y = torch.sin(train_x * (2 * math.pi)).view(-1) + torch.tensor(NOISE)
        likelihood = GaussianLikelihood()
        self.model = SingleTaskGP(
            train_x.cuda() if cuda else train_x,
            train_y.cuda() if cuda else train_y,
            likelihood,
        )
        mll = ExactMarginalLogLikelihood(likelihood, self.model)
        fit_model(mll, options={"maxiter": 1})

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


if __name__ == "__main__":
    unittest.main()
