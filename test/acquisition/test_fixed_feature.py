#! /usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

import torch
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.models import SingleTaskGP


class TestFixedFeatureAcquisitionFunction(unittest.TestCase):
    def test_fixed_features(self, cuda=False):
        device = torch.device("cuda" if cuda else "cpu")
        train_X = torch.rand(5, 3, device=device)
        train_Y = train_X.norm(dim=-1)
        model = SingleTaskGP(train_X, train_Y).to(device=device).eval()
        qEI = qExpectedImprovement(model, best_f=0.0)
        # test single point
        test_X = torch.rand(1, 3, device=device)
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
        test_X = torch.rand(2, 3, device=device)
        qEI_ff = FixedFeatureAcquisitionFunction(
            qEI, d=3, columns=[1], values=test_X[..., [1]]
        )
        qei = qEI(test_X)
        qei_ff = qEI_ff(test_X[..., [0, 2]])
        self.assertTrue(torch.allclose(qei, qei_ff))
        # test t-batch with broadcasting
        test_X = torch.rand(2, 3, device=device).expand(4, 2, 3)
        qEI_ff = FixedFeatureAcquisitionFunction(
            qEI, d=3, columns=[2], values=test_X[0, :, -1:]
        )
        qei = qEI(test_X)
        qei_ff = qEI_ff(test_X[..., :-1])
        self.assertTrue(torch.allclose(qei, qei_ff))
        # test gradient
        test_X = torch.rand(1, 3, device=device, requires_grad=True)
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

    def test_fix_features_cuda(self):
        if torch.cuda.is_available():
            self.test_fix_features(cuda=True)
