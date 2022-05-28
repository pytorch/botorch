#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.models import SingleTaskGP
from botorch.utils.testing import BotorchTestCase


class TestFixedFeatureAcquisitionFunction(BotorchTestCase):
    def test_fixed_features(self):
        train_X = torch.rand(5, 3, device=self.device)
        train_Y = train_X.norm(dim=-1, keepdim=True)
        model = SingleTaskGP(train_X, train_Y).to(device=self.device).eval()
        for q in [1, 2]:
            qEI = qExpectedImprovement(model, best_f=0.0)

            # test single point
            test_X = torch.rand(q, 3, device=self.device)
            qEI_ff = FixedFeatureAcquisitionFunction(
                qEI, d=3, columns=[2], values=test_X[..., -1:]
            )
            qei = qEI(test_X)
            qei_ff = qEI_ff(test_X[..., :-1])
            self.assertTrue(torch.allclose(qei, qei_ff))

            # test list input with float
            qEI_ff = FixedFeatureAcquisitionFunction(
                qEI, d=3, columns=[2], values=[0.5]
            )
            qei_ff = qEI_ff(test_X[..., :-1])
            test_X_clone = test_X.clone()
            test_X_clone[..., 2] = 0.5
            qei = qEI(test_X_clone)
            self.assertTrue(torch.allclose(qei, qei_ff))

            # test list input with Tensor and float
            qEI_ff = FixedFeatureAcquisitionFunction(
                qEI, d=3, columns=[0, 2], values=[test_X[..., [0]], 0.5]
            )
            qei_ff = qEI_ff(test_X[..., [1]])
            self.assertTrue(torch.allclose(qei, qei_ff))

            # test t-batch with broadcasting and list of floats
            test_X = torch.rand(q, 3, device=self.device).expand(4, q, 3)
            qEI_ff = FixedFeatureAcquisitionFunction(
                qEI, d=3, columns=[2], values=test_X[0, :, -1:]
            )
            qei = qEI(test_X)
            qei_ff = qEI_ff(test_X[..., :-1])
            self.assertTrue(torch.allclose(qei, qei_ff))

            # test t-batch with broadcasting and list of floats and Tensor
            qEI_ff = FixedFeatureAcquisitionFunction(
                qEI, d=3, columns=[0, 2], values=[test_X[0, :, [0]], 0.5]
            )
            test_X_clone = test_X.clone()
            test_X_clone[..., 2] = 0.5
            qei = qEI(test_X_clone)
            qei_ff = qEI_ff(test_X[..., [1]])
            self.assertTrue(torch.allclose(qei, qei_ff))

            # test X_pending
            X_pending = torch.rand(2, 3, device=self.device)
            qEI.set_X_pending(X_pending)
            qEI_ff = FixedFeatureAcquisitionFunction(
                qEI, d=3, columns=[2], values=test_X[..., -1:]
            )
            self.assertTrue(torch.allclose(qEI.X_pending, qEI_ff.X_pending))

            # test setting X_pending from qEI_ff
            # (set target value to be last dim of X_pending and check if the
            # constructed X_pending on qEI is the full X_pending)
            X_pending = torch.rand(2, 3, device=self.device)
            qEI.X_pending = None
            qEI_ff = FixedFeatureAcquisitionFunction(
                qEI, d=3, columns=[2], values=X_pending[..., -1:]
            )
            qEI_ff.set_X_pending(X_pending[..., :-1])
            self.assertTrue(torch.allclose(qEI.X_pending, X_pending))
            # test setting to None
            qEI_ff.X_pending = None
            self.assertIsNone(qEI_ff.X_pending)

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

        test_X = test_X.detach().clone()
        test_X_ff = test_X[..., [1]].detach().clone().requires_grad_(True)
        test_X[..., 2] = 0.5
        test_X.requires_grad_(True)
        qei = qEI(test_X)
        qEI_ff = FixedFeatureAcquisitionFunction(
            qEI, d=3, columns=[0, 2], values=[test_X[..., [0]].detach(), 0.5]
        )
        qei_ff = qEI_ff(test_X_ff)
        qei.backward()
        qei_ff.backward()
        self.assertTrue(torch.allclose(test_X.grad[..., [1]], test_X_ff.grad))

        # test error b/c of incompatible input shapes
        with self.assertRaises(ValueError):
            qEI_ff(test_X)

        # test error when there is no X_pending (analytic EI)
        test_X = torch.rand(q, 3, device=self.device)
        analytic_EI = ExpectedImprovement(model, best_f=0.0)
        EI_ff = FixedFeatureAcquisitionFunction(
            analytic_EI, d=3, columns=[2], values=test_X[..., -1:]
        )
        with self.assertRaises(ValueError):
            EI_ff.X_pending
