# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from itertools import product
from math import log, pi

import torch
from botorch.fit import fit_fully_bayesian_model_nuts, fit_gpytorch_mll
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.gp_regression import SingleTaskGP
from botorch.test_utils.mock import mock_optimize
from botorch.utils.evaluation import AIC, BIC, compute_in_sample_model_fit_metric, MLL
from botorch.utils.testing import BotorchTestCase
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood


class TestEvaluation(BotorchTestCase):
    @mock_optimize
    def test_compute_in_sample_model_fit_metric(self):
        torch.manual_seed(0)
        for dtype, model_cls in product(
            (torch.float, torch.double), (SingleTaskGP, SaasFullyBayesianSingleTaskGP)
        ):
            train_X = torch.linspace(
                0, 1, 10, dtype=dtype, device=self.device
            ).unsqueeze(-1)
            train_Y = torch.sin(2 * pi * train_X)
            model = model_cls(train_X=train_X, train_Y=train_Y)
            if model_cls is SingleTaskGP:
                fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
            else:
                fit_fully_bayesian_model_nuts(
                    model,
                    warmup_steps=8,
                    num_samples=6,
                    thinning=2,
                    disable_progbar=True,
                )
            num_params = sum(p.numel() for p in model.parameters())
            if model_cls is SaasFullyBayesianSingleTaskGP:
                num_params /= 3  # divide by number of MCMC samples
            mll = compute_in_sample_model_fit_metric(model=model, criterion=MLL)
            aic = compute_in_sample_model_fit_metric(model=model, criterion=AIC)
            bic = compute_in_sample_model_fit_metric(model=model, criterion=BIC)
            self.assertEqual(aic, 2 * num_params - 2 * mll)
            self.assertEqual(bic, log(10) * num_params - 2 * mll)
        # test invalid criterion
        with self.assertRaisesRegex(
            ValueError, "Invalid evaluation criterion invalid."
        ):
            compute_in_sample_model_fit_metric(model=model, criterion="invalid")
