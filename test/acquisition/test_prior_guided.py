#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from itertools import product

import torch
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.acquisition.prior_guided import PriorGuidedAcquisitionFunction
from botorch.models import SingleTaskGP
from botorch.utils.testing import BotorchTestCase
from torch.nn import Module


class DummyPrior(Module):
    def forward(self, X):
        p = torch.distributions.Normal(0, 1)
        # sum over d and q dimensions
        return p.log_prob(X).sum(dim=-1).sum(dim=-1).exp()


class TestPriorGuidedAcquisitionFunction(BotorchTestCase):
    def test_prior_guided_acquisition_function(self):
        prior = DummyPrior()
        for dtype in (torch.float, torch.double):
            train_X = torch.rand(5, 3, dtype=dtype, device=self.device)
            train_Y = train_X.norm(dim=-1, keepdim=True)
            model = SingleTaskGP(train_X, train_Y).eval()
            qEI = qExpectedImprovement(model, best_f=0.0)
            for batch_shape, q, use_log, exponent in product(
                ([], [2]), (1, 2), (False, True), (1.0, 2.0)
            ):
                af = PriorGuidedAcquisitionFunction(
                    acq_function=qEI,
                    prior_module=prior,
                    log=use_log,
                    prior_exponent=exponent,
                )
                test_X = torch.rand(*batch_shape, q, 3, dtype=dtype, device=self.device)
                with torch.no_grad():
                    val = af(test_X)
                    prob = prior(test_X)
                    ei = qEI(test_X)
                if use_log:
                    expected_val = prob * exponent + ei
                else:
                    expected_val = prob.pow(exponent) * ei
                self.assertTrue(torch.equal(val, expected_val))
                # test set_X_pending
                X_pending = torch.rand(2, 3, dtype=dtype, device=self.device)
                af.X_pending = X_pending
                self.assertTrue(torch.equal(X_pending, af.acq_func.X_pending))
                self.assertTrue(torch.equal(X_pending, af.X_pending))
        # test exception when base AF does not support X_pending
        ei = ExpectedImprovement(model, best_f=0.0)
        af = PriorGuidedAcquisitionFunction(
            acq_function=ei,
            prior_module=prior,
        )
        msg = (
            "Base acquisition function ExpectedImprovement "
            "does not have an `X_pending` attribute."
        )
        with self.assertRaisesRegex(ValueError, msg):
            af.X_pending
