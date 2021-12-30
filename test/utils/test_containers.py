#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
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
        training_data_bd = TrainingData.from_block_design(X_bd, Y_bd)
        self.assertTrue(training_data_bd.is_block_design)
        self.assertTrue(torch.equal(training_data_bd.X, X_bd))
        self.assertTrue(torch.equal(training_data_bd.Y, Y_bd))
        self.assertIsNone(training_data_bd.Yvar)
        self.assertTrue(torch.equal(Xi, X_bd) for Xi in training_data_bd.Xs)
        self.assertTrue(torch.equal(training_data_bd.Ys[0], Y_bd[..., :1]))
        self.assertTrue(torch.equal(training_data_bd.Ys[1], Y_bd[..., 1:]))
        self.assertIsNone(training_data_bd.Yvars)
        # test equality check with null Yvars and one-element Xs ans Ys
        self.assertEqual(
            training_data_bd,
            TrainingData(Xs=[X_bd] * 2, Ys=list(torch.split(Y_bd, 1, dim=-1))),
        )

        # block design, with variance observations
        Yvar_bd = torch.rand(2, 4, 2)
        training_data_bd = TrainingData.from_block_design(X_bd, Y_bd, Yvar_bd)
        self.assertTrue(training_data_bd.is_block_design)
        self.assertTrue(torch.equal(training_data_bd.X, X_bd))
        self.assertTrue(torch.equal(training_data_bd.Y, Y_bd))
        self.assertTrue(torch.equal(training_data_bd.Yvar, Yvar_bd))
        self.assertTrue(torch.equal(Xi, X_bd) for Xi in training_data_bd.Xs)
        self.assertTrue(torch.equal(training_data_bd.Ys[0], Y_bd[..., :1]))
        self.assertTrue(torch.equal(training_data_bd.Ys[1], Y_bd[..., 1:]))
        self.assertTrue(torch.equal(training_data_bd.Yvars[0], Yvar_bd[..., :1]))
        self.assertTrue(torch.equal(training_data_bd.Yvars[1], Yvar_bd[..., 1:]))

        # test equality check with non-null Yvars and one-element Xs ans Ys
        self.assertEqual(
            training_data_bd,
            TrainingData(
                Xs=[X_bd] * 2,
                Ys=list(torch.split(Y_bd, 1, dim=-1)),
                Yvars=list(torch.split(Yvar_bd, 1, dim=-1)),
            ),
        )

        # non-block design, without variance observations
        Xs = [torch.rand(2, 4, 3), torch.rand(2, 3, 3)]
        Ys = [torch.rand(2, 4, 2), torch.rand(2, 3, 2)]
        training_data_nbd = TrainingData(Xs, Ys)
        self.assertFalse(training_data_nbd.is_block_design)
        self.assertTrue(torch.equal(training_data_nbd.Xs[0], Xs[0]))
        self.assertTrue(torch.equal(training_data_nbd.Xs[1], Xs[1]))
        self.assertTrue(torch.equal(training_data_nbd.Ys[0], Ys[0]))
        self.assertTrue(torch.equal(training_data_nbd.Ys[1], Ys[1]))
        self.assertIsNone(training_data_nbd.Yvars)
        with self.assertRaises(UnsupportedError):
            training_data_nbd.X
        with self.assertRaises(UnsupportedError):
            training_data_nbd.Y
        self.assertIsNone(training_data_nbd.Yvar)

        # test equality check with different length Xs and Ys in two training data
        # and only one training data including non-null Yvars
        self.assertNotEqual(training_data_nbd, training_data_bd)
        # test equality of two training datas with different legth Xs/Ys
        training_data_nbd_X = TrainingData(
            Xs=Xs + [torch.rand(2, 2, 3)],
            Ys=Ys,
        )
        self.assertNotEqual(training_data_nbd, training_data_nbd_X)
        training_data_nbd_Y = TrainingData(
            Xs=Xs,
            Ys=Ys + [torch.rand(2, 2, 2)],
        )
        self.assertNotEqual(training_data_nbd, training_data_nbd_Y)

        # non-block design, with variance observations
        Yvars = [torch.rand(2, 4, 2), torch.rand(2, 3, 2)]
        training_data_nbd_yvar = TrainingData(Xs, Ys, Yvars)
        self.assertFalse(training_data_nbd_yvar.is_block_design)
        self.assertTrue(torch.equal(training_data_nbd_yvar.Xs[0], Xs[0]))
        self.assertTrue(torch.equal(training_data_nbd_yvar.Xs[1], Xs[1]))
        self.assertTrue(torch.equal(training_data_nbd_yvar.Ys[0], Ys[0]))
        self.assertTrue(torch.equal(training_data_nbd_yvar.Ys[1], Ys[1]))
        self.assertTrue(torch.equal(training_data_nbd_yvar.Yvars[0], Yvars[0]))
        self.assertTrue(torch.equal(training_data_nbd_yvar.Yvars[1], Yvars[1]))
        with self.assertRaises(UnsupportedError):
            training_data_nbd_yvar.X
        with self.assertRaises(UnsupportedError):
            training_data_nbd_yvar.Y
        with self.assertRaises(UnsupportedError):
            training_data_nbd_yvar.Yvar

        # test equality check with same length Xs and Ys in two training data but
        # with variance observations only in one
        self.assertNotEqual(training_data_nbd, training_data_nbd_yvar)
        # test equality check with different length Xs and Ys in two training data
        self.assertNotEqual(training_data_nbd_yvar, training_data_bd)

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

        # test equality with same Xs and Ys but different-length Yvars
        self.assertNotEqual(
            TrainingData(Xs, Ys, Yvars),
            TrainingData(Xs, Ys, Yvars[:1]),
        )
