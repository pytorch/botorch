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
        Xs = rand(4, 3, 2)
        Ys = rand(4, 3, 1)

        # Test `__init__`
        dataset = SupervisedDataset(X=Xs[0], Y=Ys[0])
        self.assertIsInstance(dataset.X, Tensor)
        self.assertIsInstance(dataset._X, Tensor)
        self.assertIsInstance(dataset.Y, Tensor)
        self.assertIsInstance(dataset._Y, Tensor)

        dataset = SupervisedDataset(
            X=DenseContainer(Xs[0], Xs[0].shape[-1:]),
            Y=DenseContainer(Ys[0], Ys[0].shape[-1:]),
        )
        self.assertIsInstance(dataset.X, Tensor)
        self.assertIsInstance(dataset._X, DenseContainer)
        self.assertIsInstance(dataset.Y, Tensor)
        self.assertIsInstance(dataset._Y, DenseContainer)

        # Test `_validate`
        with self.assertRaisesRegex(ValueError, "Batch dimensions .* incompatible."):
            SupervisedDataset(X=rand(1, 2), Y=rand(2, 1))

        # Test `dict_from_iter` and `__eq__`
        datasets = SupervisedDataset.dict_from_iter(X=Xs.unbind(), Y=Ys.unbind())
        self.assertIsInstance(datasets, dict)
        self.assertEqual(tuple(datasets.keys()), tuple(range(len(Xs))))
        for i, dataset in datasets.items():
            self.assertEqual(dataset, SupervisedDataset(Xs[i], Ys[i]))
        self.assertNotEqual(datasets[0], datasets)

        datasets = SupervisedDataset.dict_from_iter(X=Xs[0], Y=Ys.unbind())
        self.assertEqual(len(datasets), len(Xs))
        for i in range(1, len(Xs)):
            self.assertTrue(torch.equal(datasets[0].X, datasets[i].X))

        # Test with Yvar.
        dataset = SupervisedDataset(
            X=Xs[0], Y=Ys[0], Yvar=DenseContainer(Ys[0], Ys[0].shape[-1:])
        )
        self.assertIsInstance(dataset.X, Tensor)
        self.assertIsInstance(dataset._X, Tensor)
        self.assertIsInstance(dataset.Y, Tensor)
        self.assertIsInstance(dataset._Y, Tensor)
        self.assertIsInstance(dataset.Yvar, Tensor)
        self.assertIsInstance(dataset._Yvar, DenseContainer)

    def test_fixedNoise(self):
        # Generate some data
        Xs = rand(4, 3, 2)
        Ys = rand(4, 3, 1)
        Ys_var = rand(4, 3, 1)

        # Test `dict_from_iter`
        datasets = FixedNoiseDataset.dict_from_iter(
            X=Xs.unbind(),
            Y=Ys.unbind(),
            Yvar=Ys_var.unbind(),
        )
        for i, dataset in datasets.items():
            self.assertTrue(dataset.X.equal(Xs[i]))
            self.assertTrue(dataset.Y.equal(Ys[i]))
            self.assertTrue(dataset.Yvar.equal(Ys_var[i]))

        # Test handling of Tensor-valued arguments to `dict_from_iter`
        datasets = FixedNoiseDataset.dict_from_iter(
            X=Xs[0],
            Y=Ys[1],
            Yvar=Ys_var.unbind(),
        )
        for dataset in datasets.values():
            self.assertTrue(Xs[0].equal(dataset.X))
            self.assertTrue(Ys[1].equal(dataset.Y))

        with self.assertRaisesRegex(
            ValueError, "`Y` and `Yvar`"
        ), self.assertWarnsRegex(DeprecationWarning, "SupervisedDataset"):
            FixedNoiseDataset(X=Xs, Y=Ys, Yvar=Ys_var[0])

    def test_ranking(self):
        # Test `_validate`
        X_val = rand(16, 2)
        X_idx = stack([randperm(len(X_val))[:3] for _ in range(1)])
        X = SliceContainer(X_val, X_idx, event_shape=Size([3 * X_val.shape[-1]]))

        with self.assertRaisesRegex(ValueError, "out-of-bounds"):
            RankingDataset(X=X, Y=tensor([[-1, 0, 1]]))
        RankingDataset(X=X, Y=tensor([[2, 0, 1]]))

        with self.assertRaisesRegex(ValueError, "out-of-bounds"):
            RankingDataset(X=X, Y=tensor([[0, 1, 3]]))
        RankingDataset(X=X, Y=tensor([[0, 1, 2]]))

        with self.assertRaisesRegex(ValueError, "missing zero-th rank."):
            RankingDataset(X=X, Y=tensor([[1, 2, 2]]))
        RankingDataset(X=X, Y=tensor([[0, 1, 1]]))

        with self.assertRaisesRegex(ValueError, "ranks not skipped after ties."):
            RankingDataset(X=X, Y=tensor([[0, 0, 1]]))
        RankingDataset(X=X, Y=tensor([[0, 0, 2]]))
