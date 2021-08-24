#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.acquisition.proximal import ProximalAcquisitionFunction
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.models import SingleTaskGP
from botorch.utils.testing import BotorchTestCase
from torch.distributions.multivariate_normal import MultivariateNormal


class TestProximalAcquisitionFunction(BotorchTestCase):
    def test_proximal(self):
        train_X = torch.rand(5, 3, device=self.device)
        train_Y = train_X.norm(dim=-1, keepdim=True)
        model = SingleTaskGP(train_X, train_Y).to(device=self.device).eval()
        EI = ExpectedImprovement(model, best_f=0.0)

        # test single point
        proximal_weights = torch.ones(3)
        test_X = torch.rand(1, 3, device=self.device)
        EI_prox = ProximalAcquisitionFunction(EI, proximal_weights=proximal_weights)

        ei = EI(test_X)
        mv_normal = MultivariateNormal(train_X[-1], torch.diag(proximal_weights))
        test_prox_weight = torch.exp(mv_normal.log_prob(test_X)) / torch.exp(
            mv_normal.log_prob(train_X[-1]))

        ei_prox = EI_prox(test_X)
        self.assertAlmostEqual(float(ei_prox), float(ei*test_prox_weight),
                               places=5)
        self.assertTrue(ei_prox.shape == torch.Size([1]))

        # test t-batch with broadcasting
        test_X = torch.rand(4, 1, 3, device=self.device)
        ei_prox = EI_prox(test_X)

        ei = EI(test_X)
        mv_normal = MultivariateNormal(train_X[-1], torch.diag(proximal_weights))
        test_prox_weight = torch.exp(mv_normal.log_prob(test_X)) / torch.exp(
            mv_normal.log_prob(train_X[-1]))

        ei_prox = EI_prox(test_X)
        for a, b in zip(ei_prox, ei * test_prox_weight.flatten()):
            self.assertAlmostEqual(float(a), float(b),
                                   places=5)
        self.assertTrue(ei_prox.shape == torch.Size([4]))

        # test MC acquisition function
        qEI = qExpectedImprovement(model, best_f=0.0)
        test_X = torch.rand(4, 1, 3, device=self.device)
        qEI_prox = ProximalAcquisitionFunction(qEI, proximal_weights=proximal_weights)

        qei = qEI(test_X)
        mv_normal = MultivariateNormal(train_X[-1], torch.diag(proximal_weights))
        test_prox_weight = torch.exp(mv_normal.log_prob(test_X)) / torch.exp(
            mv_normal.log_prob(train_X[-1]))

        qei_prox = qEI_prox(test_X)
        for a, b in zip(qei_prox, qei * test_prox_weight.flatten()):
            self.assertAlmostEqual(float(a), float(b),
                                   places=5)
        self.assertTrue(qei_prox.shape == torch.Size([4]))

        # test gradient
        test_X = torch.rand(1, 3, device=self.device, requires_grad=True)
        ei_prox = EI_prox(test_X)
        ei_prox.backward()
