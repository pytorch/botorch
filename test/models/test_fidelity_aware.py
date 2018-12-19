#! /usr/bin/env python3

import math
import unittest

import torch
from botorch import fit_model
from botorch.models.fidelity_aware import (
    FidelityAwareHeteroskedasticNoise,
    FidelityAwareSingleTaskGP,
)
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood, _GaussianLikelihoodBase
from gpytorch.means import ConstantMean
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood

from ..test_fit import NOISE


class FidelityAwareSingleTaskGPTest(unittest.TestCase):
    def setUp(self, cuda=False):
        train_x = torch.stack([torch.linspace(0, 1, 10), torch.ones(10)], -1)
        train_y = torch.sin(train_x[:, 0] * (2 * math.pi)) + torch.tensor(NOISE)
        train_y_sem = 0.1 + 0.1 * torch.rand_like(train_y)
        self.model = FidelityAwareSingleTaskGP(
            train_x.cuda() if cuda else train_x,
            train_y.cuda() if cuda else train_y,
            train_y_sem.cuda() if cuda else train_y_sem,
            phi_idcs=1,
        )
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_model(mll, options={"maxiter": 1})

    def testInit(self):
        self.assertIsInstance(self.model.mean_module, ConstantMean)
        self.assertIsInstance(self.model.covar_module, ScaleKernel)
        kernel = self.model.covar_module.base_kernel
        self.assertIsInstance(kernel, MaternKernel)
        likelihood = self.model.likelihood
        self.assertIsInstance(likelihood, _GaussianLikelihoodBase)
        self.assertFalse(isinstance(likelihood, GaussianLikelihood))
        self.assertIsInstance(likelihood.noise_covar, FidelityAwareHeteroskedasticNoise)

    def testForward(self):
        test_x = torch.rand(3, 2)
        posterior = self.model(test_x)
        self.assertIsInstance(posterior, MultivariateNormal)

    def testReinitialize(self):
        train_x = torch.stack([torch.linspace(0, 1, 11), torch.ones(11)], -1)
        noise = torch.tensor(NOISE + [0.1])
        train_y = torch.sin(train_x[:, 0] * (2 * math.pi)) + noise
        train_y_sem = 0.1 + 0.1 * torch.rand_like(train_y)
        self.model.reinitialize(train_x, train_y, train_y_sem)
        params = dict(self.model.named_parameters())
        for p in params:
            self.assertEqual(params[p].item(), 0.0)
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_model(mll)
        # check that some of the parameters changed
        self.assertFalse(all(params[p].item() == 0.0 for p in params))


if __name__ == "__main__":
    unittest.main()
