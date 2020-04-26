#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.models import SingleTaskGP
from botorch.utils.testing import BotorchTestCase


class TestFixedFeatureAcquisitionFunction(BotorchTestCase):
    def test_fixed_features(self):
        train_X = torch.rand(5, 3, device=self.device)
        train_Y = train_X.norm(dim=-1, keepdim=True)
        model = SingleTaskGP(train_X, train_Y).to(device=self.device).eval()
        qEI = qExpectedImprovement(model, best_f=0.0)
        # test single point
        test_X = torch.rand(1, 3, device=self.device)
        qEI_ff = FixedFeatureAcquisitionFunction(
            qEI, d=3, columns=[2], values=test_X[..., -1:]
        )
        qei = qEI(test_X)
        qei_ff = qEI_ff(test_X[..., :-1])
        self.assertTrue(torch.allclose(qei, qei_ff))
        # test list input
        qEI_ff = FixedFeatureAcquisitionFunction(qEI, d=3, columns=[2], values=[0.5])
        qei_ff = qEI_ff(test_X[..., :-1])
        # test q-batch
        test_X = torch.rand(2, 3, device=self.device)
        qEI_ff = FixedFeatureAcquisitionFunction(
            qEI, d=3, columns=[1], values=test_X[..., [1]]
        )
        qei = qEI(test_X)
        qei_ff = qEI_ff(test_X[..., [0, 2]])
        self.assertTrue(torch.allclose(qei, qei_ff))
        # test t-batch with broadcasting
        test_X = torch.rand(2, 3, device=self.device).expand(4, 2, 3)
        qEI_ff = FixedFeatureAcquisitionFunction(
            qEI, d=3, columns=[2], values=test_X[0, :, -1:]
        )
        qei = qEI(test_X)
        qei_ff = qEI_ff(test_X[..., :-1])
        self.assertTrue(torch.allclose(qei, qei_ff))
        # test gradient
        test_X = torch.rand(1, 3, device=self.device, requires_grad=True)
        test_X_ff = test_X[..., :-1].detach().clone().requires_grad_(True)
        qei = qEI(test_X)
        qEI_ff = FixedFeatureAcquisitionFunction(
            qEI, d=3, columns=[2], values=test_X[..., [2]].detach()
        )
        qei_ff = qEI_ff(test_X_ff)
        self.assertTrue(torch.allclose(qei, qei_ff))
        qei.backward()
        qei_ff.backward()
        self.assertTrue(torch.allclose(test_X.grad[..., :-1], test_X_ff.grad))
        # test error b/c of incompatible input shapes
        with self.assertRaises(ValueError):
            qEI_ff(test_X)
