#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from botorch.acquisition.proximal import ProximalAcquisitionFunction
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.models import SingleTaskGP
from botorch.utils.testing import BotorchTestCase


class TestFixedFeatureAcquisitionFunction(BotorchTestCase):
    def test_fixed_features(self):
        train_X = torch.rand(5, 3, device=self.device)
        train_Y = train_X.norm(dim=-1, keepdim=True)
        model = SingleTaskGP(train_X, train_Y).to(device=self.device).eval()
        EI = ExpectedImprovement(model, best_f=0.0)

        # test single point
        proximal_weights = torch.ones(3)
        test_X = torch.rand(1, 3, device=self.device)
        EI_prox = ProximalAcquisitionFunction(
            EI, proximal_weights=proximal_weights)

        ei_prox = EI_prox(test_X)

        # test t-batch with broadcasting
        test_X = torch.rand(1, 3, device=self.device).expand(4, 1, 3)
        ei_prox = EI_prox(test_X)

        # test gradient
        test_X = torch.rand(1, 3, device=self.device, requires_grad=True)
        ei_prox = EI_prox(test_X)
        ei_prox.backward()
