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
        Xs = torch.tensor([[-1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
        Ys = torch.tensor([[-1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
        Yvars = torch.tensor([[-1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])

        training_data = TrainingData(Xs, Ys, Yvars)
        self.assertTrue(torch.equal(training_data.Xs, Xs))
        self.assertTrue(torch.equal(training_data.Ys, Ys))
        self.assertTrue(torch.equal(training_data.Yvars, Yvars))
