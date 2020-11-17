#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
from botorch import fit_gpytorch_model
from botorch.models.contextual import LCEAGP, SACGP
from botorch.models.gp_regression import FixedNoiseGP
from botorch.models.kernels.contextual_lcea import LCEAKernel
from botorch.models.kernels.contextual_sac import SACKernel
from botorch.utils.testing import BotorchTestCase
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.means import ConstantMean
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood


class ContextualGPTest(BotorchTestCase):
    def test_SACGP(self):
        train_X = torch.tensor(
            [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]]
        )
        train_Y = torch.tensor([[1.0], [2.0], [3.0]])
        train_Yvar = 0.01 * torch.ones(3, 1, dtype=torch.double)
        self.decomposition = {"1": [0, 3], "2": [1, 2]}

        self.model = SACGP(train_X, train_Y, train_Yvar, self.decomposition)
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_model(mll, options={"maxiter": 1})

        self.assertIsInstance(self.model, FixedNoiseGP)
        self.assertDictEqual(self.model.decomposition, self.decomposition)
        self.assertIsInstance(self.model.mean_module, ConstantMean)
        self.assertIsInstance(self.model.covar_module, SACKernel)

        # test number of named parameters
        num_of_mean = 0
        num_of_lengthscales = 0
        num_of_outputscales = 0
        for param_name, param in self.model.named_parameters():
            if param_name == "mean_module.constant":
                num_of_mean += param.data.shape.numel()
            elif "raw_lengthscale" in param_name:
                num_of_lengthscales += param.data.shape.numel()
            elif "raw_outputscale" in param_name:
                num_of_outputscales += param.data.shape.numel()
        self.assertEqual(num_of_mean, 1)
        self.assertEqual(num_of_lengthscales, 2)
        self.assertEqual(num_of_outputscales, 2)

        test_x = torch.rand(5, 4)
        posterior = self.model(test_x)
        self.assertIsInstance(posterior, MultivariateNormal)

    def testLCEAGP(self):
        train_X = torch.tensor(
            [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]]
        )
        train_Y = torch.tensor([[1.0], [2.0], [3.0]])
        train_Yvar = 0.01 * torch.ones(3, 1, dtype=torch.double)
        # Test setting attributes
        decomposition = {"1": [0, 1], "2": [2, 3]}

        # test instantiate model
        model = LCEAGP(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            decomposition=decomposition,
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll, options={"maxiter": 1})

        self.assertIsInstance(model, LCEAGP)
        self.assertIsInstance(model.covar_module, LCEAKernel)
        self.assertDictEqual(model.decomposition, decomposition)
