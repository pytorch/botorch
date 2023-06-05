#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from botorch.test_functions.sensitivity_analysis import Gsobol, Ishigami, Morris
from botorch.utils.testing import BotorchTestCase


class TestIshigami(BotorchTestCase):
    def testFunction(self):
        with self.assertRaises(ValueError):
            Ishigami(b=0.33)
        f = Ishigami(b=0.1)
        self.assertEqual(f.b, 0.1)
        f = Ishigami(b=0.05)
        self.assertEqual(f.b, 0.05)
        X = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        m1, m2, m3 = f.compute_dgsm(X)
        for m in [m1, m2, m3]:
            self.assertEqual(len(m), 3)
        Z = f.evaluate_true(X)
        Ztrue = torch.tensor([5.8401, 7.4245])
        self.assertAllClose(Z, Ztrue, atol=1e-3)
        self.assertIsNone(f._optimizers)
        with self.assertRaises(NotImplementedError):
            f.optimal_value


class TestGsobol(BotorchTestCase):
    def testFunction(self):
        for dim in [6, 8, 15]:
            f = Gsobol(dim=dim)
            self.assertIsNotNone(f.a)
            self.assertEqual(len(f.a), dim)
        f = Gsobol(dim=3, a=[1, 2, 3])
        self.assertEqual(f.a, [1, 2, 3])
        X = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        Z = f.evaluate_true(X)
        Ztrue = torch.tensor([2.5, 21.0])
        self.assertAllClose(Z, Ztrue, atol=1e-3)
        self.assertIsNone(f._optimizers)
        with self.assertRaises(NotImplementedError):
            f.optimal_value


class TestMorris(BotorchTestCase):
    def testFunction(self):
        f = Morris()
        X = torch.stack((torch.zeros(20), torch.ones(20)))
        Z = f.evaluate_true(X)
        Ztrue = torch.tensor([5163.0, -8137.0])
        self.assertAllClose(Z, Ztrue, atol=1e-3)
        self.assertIsNone(f._optimizers)
        with self.assertRaises(NotImplementedError):
            f.optimal_value
