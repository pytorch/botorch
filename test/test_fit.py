#! /usr/bin/env python3

import math
import unittest

import torch
from botorch import fit_model
from botorch.models import GPRegressionModel
from gpytorch.likelihoods import GaussianLikelihood


NOISE = [0.127, -0.113, -0.345, -0.034, -0.069, -0.272, 0.013, 0.056, 0.087, -0.081]


class TestFitModel(unittest.TestCase):
    def setUp(self):
        self.train_x = torch.linspace(0, 1, 10)
        self.train_y = torch.sin(self.train_x * (2 * math.pi)) + torch.tensor(NOISE)

    def test_fit_model(self, cuda=False):
        train_x = self.train_x.cuda() if cuda else self.train_x
        train_y = self.train_y.cuda() if cuda else self.train_y
        likelihood = GaussianLikelihood()
        model = fit_model(
            gp_model=GPRegressionModel,
            likelihood=likelihood,
            train_x=train_x,
            train_y=train_y,
            max_iter=5,
        )
        self.assertLess((model.likelihood.log_noise + 0.25).abs(), 0.05)
        self.assertLess(model.mean_module.constant.abs(), 0.1)
        self.assertLess((model.covar_module.base_kernel.log_lengthscale).abs(), 0.5)
        self.assertLess((model.covar_module.log_outputscale + 0.235).abs(), 0.1)

    def test_fit_model_cuda(self):
        if torch.cuda.is_available():
            self.test_fit_model(cuda=True)


if __name__ == "__main__":
    unittest.main()
