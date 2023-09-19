#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from botorch.utils.containers import DenseContainer, SliceContainer
from botorch.utils.datasets import FixedNoiseDataset, RankingDataset, SupervisedDataset
from botorch.utils.testing import BotorchTestCase
from torch import rand, randperm, Size, stack, Tensor, tensor


class TestDatasets(BotorchTestCase):
    def test_supervised(self):
        # Generate some data
        X = rand(3, 2)
        Y = rand(3, 1)
        feature_names = ["x1", "x2"]
        outcome_names = ["y"]

        # Test `__init__`
        dataset = SupervisedDataset(
            X=X, Y=Y, feature_names=feature_names, outcome_names=outcome_names
        )
        self.assertIsInstance(dataset.X, Tensor)
        self.assertIsInstance(dataset._X, Tensor)
        self.assertIsInstance(dataset.Y, Tensor)
        self.assertIsInstance(dataset._Y, Tensor)
        self.assertEqual(dataset.feature_names, feature_names)
        self.assertEqual(dataset.outcome_names, outcome_names)

        dataset2 = SupervisedDataset(
            X=DenseContainer(X, X.shape[-1:]),
            Y=DenseContainer(Y, Y.shape[-1:]),
            feature_names=feature_names,
            outcome_names=outcome_names,
        )
        self.assertIsInstance(dataset2.X, Tensor)
        self.assertIsInstance(dataset2._X, DenseContainer)
        self.assertIsInstance(dataset2.Y, Tensor)
        self.assertIsInstance(dataset2._Y, DenseContainer)
        self.assertEqual(dataset, dataset2)

        # Test `_validate`
        with self.assertRaisesRegex(ValueError, "Batch dimensions .* incompatible."):
            SupervisedDataset(
                X=rand(1, 2),
                Y=rand(2, 1),
                feature_names=feature_names,
                outcome_names=outcome_names,
            )
        with self.assertRaisesRegex(ValueError, "`Y` and `Yvar`"):
            SupervisedDataset(
                X=rand(2, 2),
                Y=rand(2, 1),
                Yvar=rand(2),
                feature_names=feature_names,
                outcome_names=outcome_names,
            )
        with self.assertRaisesRegex(ValueError, "feature_names"):
            SupervisedDataset(
                X=rand(2, 2),
                Y=rand(2, 1),
                feature_names=[],
                outcome_names=outcome_names,
            )
        with self.assertRaisesRegex(ValueError, "outcome_names"):
            SupervisedDataset(
                X=rand(2, 2),
                Y=rand(2, 1),
                feature_names=feature_names,
                outcome_names=[],
            )

        # Test with Yvar.
        dataset = SupervisedDataset(
            X=X,
            Y=Y,
            Yvar=DenseContainer(Y, Y.shape[-1:]),
            feature_names=feature_names,
            outcome_names=outcome_names,
        )
        self.assertIsInstance(dataset.X, Tensor)
        self.assertIsInstance(dataset._X, Tensor)
        self.assertIsInstance(dataset.Y, Tensor)
        self.assertIsInstance(dataset._Y, Tensor)
        self.assertIsInstance(dataset.Yvar, Tensor)
        self.assertIsInstance(dataset._Yvar, DenseContainer)

    def test_fixedNoise(self):
        # Generate some data
        X = rand(3, 2)
        Y = rand(3, 1)
        Yvar = rand(3, 1)
        feature_names = ["x1", "x2"]
        outcome_names = ["y"]
        dataset = FixedNoiseDataset(
            X=X,
            Y=Y,
            Yvar=Yvar,
            feature_names=feature_names,
            outcome_names=outcome_names,
        )
        self.assertTrue(torch.equal(dataset.X, X))
        self.assertTrue(torch.equal(dataset.Y, Y))
        self.assertTrue(torch.equal(dataset.Yvar, Yvar))
        self.assertEqual(dataset.feature_names, feature_names)
        self.assertEqual(dataset.outcome_names, outcome_names)

        with self.assertRaisesRegex(
            ValueError, "`Y` and `Yvar`"
        ), self.assertWarnsRegex(DeprecationWarning, "SupervisedDataset"):
            FixedNoiseDataset(
                X=X,
                Y=Y,
                Yvar=Yvar.squeeze(),
                feature_names=feature_names,
                outcome_names=outcome_names,
            )

    def test_ranking(self):
        # Test `_validate`
        X_val = rand(16, 2)
        X_idx = stack([randperm(len(X_val))[:3] for _ in range(1)])
        X = SliceContainer(X_val, X_idx, event_shape=Size([3 * X_val.shape[-1]]))
        feature_names = ["x1", "x2"]
        outcome_names = ["ranking indices"]

        with self.assertRaisesRegex(ValueError, "The `values` field of `X`"):
            RankingDataset(
                X=X,
                Y=tensor([[-1, 0, 1]]),
                feature_names=feature_names[:1],
                outcome_names=outcome_names,
            )

        with self.assertRaisesRegex(ValueError, "out-of-bounds"):
            RankingDataset(
                X=X,
                Y=tensor([[-1, 0, 1]]),
                feature_names=feature_names,
                outcome_names=outcome_names,
            )
        RankingDataset(
            X=X,
            Y=tensor([[2, 0, 1]]),
            feature_names=feature_names,
            outcome_names=outcome_names,
        )

        with self.assertRaisesRegex(ValueError, "out-of-bounds"):
            RankingDataset(
                X=X,
                Y=tensor([[0, 1, 3]]),
                feature_names=feature_names,
                outcome_names=outcome_names,
            )
        RankingDataset(
            X=X,
            Y=tensor([[0, 1, 2]]),
            feature_names=feature_names,
            outcome_names=outcome_names,
        )

        with self.assertRaisesRegex(ValueError, "missing zero-th rank."):
            RankingDataset(
                X=X,
                Y=tensor([[1, 2, 2]]),
                feature_names=feature_names,
                outcome_names=outcome_names,
            )
        RankingDataset(
            X=X,
            Y=tensor([[0, 1, 1]]),
            feature_names=feature_names,
            outcome_names=outcome_names,
        )

        with self.assertRaisesRegex(ValueError, "ranks not skipped after ties."):
            RankingDataset(
                X=X,
                Y=tensor([[0, 0, 1]]),
                feature_names=feature_names,
                outcome_names=outcome_names,
            )
        RankingDataset(
            X=X,
            Y=tensor([[0, 0, 2]]),
            feature_names=feature_names,
            outcome_names=outcome_names,
        )
