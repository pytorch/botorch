#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import List, Any, Dict

import torch
from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.acquisition.proximal import ProximalAcquisitionFunction
from botorch.exceptions.errors import UnsupportedError
from botorch.models import SingleTaskGP
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.model import Model
from botorch.utils.containers import TrainingData
from botorch.utils.testing import BotorchTestCase
from torch.distributions.multivariate_normal import MultivariateNormal


class DummyModel(GPyTorchModel):
    num_outputs = 1

    def __init__(self):
        super(GPyTorchModel, self).__init__()

    @classmethod
    def construct_inputs(
        cls, training_data: TrainingData, **kwargs: Any
    ) -> Dict[str, Any]:
        pass

    def subset_output(self, idcs: List[int]) -> Model:
        pass


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
            mv_normal.log_prob(train_X[-1])
        )

        ei_prox = EI_prox(test_X)
        self.assertTrue(torch.allclose(ei_prox, ei * test_prox_weight))
        self.assertTrue(ei_prox.shape == torch.Size([1]))

        # test t-batch with broadcasting
        test_X = torch.rand(4, 1, 3, device=self.device)
        ei_prox = EI_prox(test_X)

        ei = EI(test_X)
        mv_normal = MultivariateNormal(train_X[-1], torch.diag(proximal_weights))
        test_prox_weight = torch.exp(mv_normal.log_prob(test_X)) / torch.exp(
            mv_normal.log_prob(train_X[-1])
        )

        ei_prox = EI_prox(test_X)
        self.assertTrue(torch.allclose(ei_prox, ei * test_prox_weight.flatten()))
        self.assertTrue(ei_prox.shape == torch.Size([4]))

        # test MC acquisition function
        qEI = qExpectedImprovement(model, best_f=0.0)
        test_X = torch.rand(4, 1, 3, device=self.device)
        qEI_prox = ProximalAcquisitionFunction(qEI, proximal_weights=proximal_weights)

        qei = qEI(test_X)
        mv_normal = MultivariateNormal(train_X[-1], torch.diag(proximal_weights))
        test_prox_weight = torch.exp(mv_normal.log_prob(test_X)) / torch.exp(
            mv_normal.log_prob(train_X[-1])
        )

        qei_prox = qEI_prox(test_X)
        self.assertTrue(torch.allclose(qei_prox, qei * test_prox_weight.flatten()))
        self.assertTrue(qei_prox.shape == torch.Size([4]))

        # test gradient
        test_X = torch.rand(1, 3, device=self.device, requires_grad=True)
        ei_prox = EI_prox(test_X)
        ei_prox.backward()

        # test model without train_inputs
        bad_model = DummyModel()
        with self.assertRaises(UnsupportedError):
            ProximalAcquisitionFunction(
                ExpectedImprovement(bad_model, 0.0), proximal_weights
            )

        # test proximal weights that do not match training_inputs
        train_X = torch.rand(5, 1, 3, device=self.device)
        train_Y = train_X.norm(dim=-1, keepdim=True)
        model = SingleTaskGP(train_X, train_Y).to(device=self.device).eval()
        with self.assertRaises(UnsupportedError):
            ProximalAcquisitionFunction(
                ExpectedImprovement(model, 0.0), proximal_weights[:1]
            )

        # test model with multi-batch training inputs
        train_X = torch.rand(5, 2, 3, device=self.device)
        train_Y = train_X.norm(dim=-1, keepdim=True)
        bad_single_task = SingleTaskGP(train_X, train_Y).to(device=self.device).eval()
        with self.assertRaises(UnsupportedError):
            ProximalAcquisitionFunction(
                ExpectedImprovement(bad_single_task, 0.0), proximal_weights
            )
