#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from botorch.utils.containers import TrainingData
from botorch.utils.testing import BotorchTestCase


class TestConstructContainers(BotorchTestCase):
    def test_TrainingData(self):
        X = torch.tensor([[-1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
        Y = torch.tensor([[-1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
        Yvar = torch.tensor([[-1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])

        training_data = TrainingData(X, Y)
        self.assertTrue(torch.equal(training_data.X, X))
        self.assertTrue(torch.equal(training_data.Y, Y))
        self.assertEqual(training_data.Yvar, None)

        training_data = TrainingData(X, Y, Yvar)
        self.assertTrue(torch.equal(training_data.X, X))
        self.assertTrue(torch.equal(training_data.Y, Y))
        self.assertTrue(torch.equal(training_data.Yvar, Yvar))
