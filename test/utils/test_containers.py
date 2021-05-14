#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from botorch.exceptions.errors import UnsupportedError
from botorch.utils.containers import TrainingData
from botorch.utils.testing import BotorchTestCase


class TestContainers(BotorchTestCase):
    def test_TrainingData(self):

        # block design, without variance observations
        X_bd = torch.rand(2, 4, 3)
        Y_bd = torch.rand(2, 4, 2)
        training_data = TrainingData.from_block_design(X_bd, Y_bd)
        self.assertTrue(training_data.is_block_design)
        self.assertTrue(torch.equal(training_data.X, X_bd))
        self.assertTrue(torch.equal(training_data.Y, Y_bd))
        self.assertIsNone(training_data.Yvar)
        self.assertTrue(torch.equal(Xi, X_bd) for Xi in training_data.Xs)
        self.assertTrue(torch.equal(training_data.Ys[0], Y_bd[..., :1]))
        self.assertTrue(torch.equal(training_data.Ys[1], Y_bd[..., 1:]))
        self.assertIsNone(training_data.Yvars)

        # block design, with variance observations
        Yvar_bd = torch.rand(2, 4, 2)
        training_data = TrainingData.from_block_design(X_bd, Y_bd, Yvar_bd)
        self.assertTrue(training_data.is_block_design)
        self.assertTrue(torch.equal(training_data.X, X_bd))
        self.assertTrue(torch.equal(training_data.Y, Y_bd))
        self.assertTrue(torch.equal(training_data.Yvar, Yvar_bd))
        self.assertTrue(torch.equal(Xi, X_bd) for Xi in training_data.Xs)
        self.assertTrue(torch.equal(training_data.Ys[0], Y_bd[..., :1]))
        self.assertTrue(torch.equal(training_data.Ys[1], Y_bd[..., 1:]))
        self.assertTrue(torch.equal(training_data.Yvars[0], Yvar_bd[..., :1]))
        self.assertTrue(torch.equal(training_data.Yvars[1], Yvar_bd[..., 1:]))

        # non-block design, without variance observations
        Xs = [torch.rand(2, 4, 3), torch.rand(2, 3, 3)]
        Ys = [torch.rand(2, 4, 2), torch.rand(2, 3, 2)]
        training_data = TrainingData(Xs, Ys)
        self.assertFalse(training_data.is_block_design)
        self.assertTrue(torch.equal(training_data.Xs[0], Xs[0]))
        self.assertTrue(torch.equal(training_data.Xs[1], Xs[1]))
        self.assertTrue(torch.equal(training_data.Ys[0], Ys[0]))
        self.assertTrue(torch.equal(training_data.Ys[1], Ys[1]))
        self.assertIsNone(training_data.Yvars)
        with self.assertRaises(UnsupportedError):
            training_data.X
        with self.assertRaises(UnsupportedError):
            training_data.Y
        self.assertIsNone(training_data.Yvar)

        # non-block design, with variance observations
        Yvars = [torch.rand(2, 4, 2), torch.rand(2, 3, 2)]
        training_data = TrainingData(Xs, Ys, Yvars)
        self.assertFalse(training_data.is_block_design)
        self.assertTrue(torch.equal(training_data.Xs[0], Xs[0]))
        self.assertTrue(torch.equal(training_data.Xs[1], Xs[1]))
        self.assertTrue(torch.equal(training_data.Ys[0], Ys[0]))
        self.assertTrue(torch.equal(training_data.Ys[1], Ys[1]))
        self.assertTrue(torch.equal(training_data.Yvars[0], Yvars[0]))
        self.assertTrue(torch.equal(training_data.Yvars[1], Yvars[1]))
        with self.assertRaises(UnsupportedError):
            training_data.X
        with self.assertRaises(UnsupportedError):
            training_data.Y
        with self.assertRaises(UnsupportedError):
            training_data.Yvar

        # implicit block design, without variance observations
        X = torch.rand(2, 4, 3)
        Xs = [X] * 2
        Ys = [torch.rand(2, 4, 2), torch.rand(2, 4, 2)]
        training_data = TrainingData(Xs, Ys)
        self.assertTrue(training_data.is_block_design)
        self.assertTrue(torch.equal(training_data.X, X))
        self.assertTrue(torch.equal(training_data.Y, torch.cat(Ys, dim=-1)))
        self.assertIsNone(training_data.Yvar)
        self.assertTrue(torch.equal(training_data.Xs[0], X))
        self.assertTrue(torch.equal(training_data.Xs[1], X))
        self.assertTrue(torch.equal(training_data.Ys[0], Ys[0]))
        self.assertTrue(torch.equal(training_data.Ys[1], Ys[1]))
        self.assertIsNone(training_data.Yvars)

        # implicit block design, with variance observations
        Yvars = [torch.rand(2, 4, 2), torch.rand(2, 4, 2)]
        training_data = TrainingData(Xs, Ys, Yvars)
        self.assertTrue(training_data.is_block_design)
        self.assertTrue(torch.equal(training_data.X, X))
        self.assertTrue(torch.equal(training_data.Y, torch.cat(Ys, dim=-1)))
        self.assertTrue(torch.equal(training_data.Yvar, torch.cat(Yvars, dim=-1)))
        self.assertTrue(torch.equal(training_data.Xs[0], X))
        self.assertTrue(torch.equal(training_data.Xs[1], X))
        self.assertTrue(torch.equal(training_data.Ys[0], Ys[0]))
        self.assertTrue(torch.equal(training_data.Ys[1], Ys[1]))
        self.assertTrue(torch.equal(training_data.Yvars[0], Yvars[0]))
        self.assertTrue(torch.equal(training_data.Yvars[1], Yvars[1]))
