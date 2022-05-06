#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import field, make_dataclass
from unittest.mock import patch

from botorch.utils.containers import DenseContainer, SliceContainer
from botorch.utils.datasets import (
    BotorchDataset,
    FixedNoiseDataset,
    RankingDataset,
    SupervisedDataset,
)
from botorch.utils.testing import BotorchTestCase
from torch import rand, randperm, Size, stack, tensor


class TestDatasets(BotorchTestCase):
    def test_base(self):
        with patch.object(BotorchDataset, "_validate", new=lambda self: 1 / 0):
            with self.assertRaises(ZeroDivisionError):
                BotorchDataset()

        dataset = BotorchDataset()
        self.assertTrue(dataset._validate() is None)

    def test_supervised_meta(self):
        X = rand(3, 2)
        Y = rand(3, 1)
        A = DenseContainer(rand(3, 5), event_shape=Size([5]))

        SupervisedDatasetWithDefaults = make_dataclass(
            cls_name="SupervisedDatasetWithDefaults",
            bases=(SupervisedDataset,),
            fields=[
                ("default", DenseContainer, field(default=A)),
                ("factory", DenseContainer, field(default_factory=lambda: A)),
            ],
        )

        # Check that call signature is property enforced
        with self.assertRaisesRegex(RuntimeError, "Missing .* `X`"):
            SupervisedDatasetWithDefaults(Y=Y)

        with self.assertRaisesRegex(RuntimeError, "Missing .* `Y`"):
            SupervisedDatasetWithDefaults(X=X)

        with self.assertRaisesRegex(TypeError, "Expected <BotorchContainer | Tensor>"):
            SupervisedDatasetWithDefaults(X=X, Y=Y.tolist())

        # Check handling of default values and factories
        dataset = SupervisedDatasetWithDefaults(X=X, Y=Y)
        self.assertEqual(dataset.default, A)
        self.assertEqual(dataset.factory, A)

        # Check type coercion
        dataset = SupervisedDatasetWithDefaults(X=X, Y=Y, default=X, factory=Y)
        self.assertIsInstance(dataset.X, DenseContainer)
        self.assertIsInstance(dataset.Y, DenseContainer)
        self.assertEqual(dataset.default, dataset.X)
        self.assertEqual(dataset.factory, dataset.Y)

    def test_supervised(self):
        # Generate some data
        Xs = rand(4, 3, 2)
        Ys = rand(4, 3, 1)

        # Test `__post_init__`
        dataset = SupervisedDataset(X=Xs[0], Y=Ys[0])
        for name in ("X", "Y"):
            field = getattr(dataset, name)
            self.assertIsInstance(field, DenseContainer)
            self.assertEqual(field.event_shape, field.values.shape[-1:])

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
            self.assertEqual(datasets[0].X, datasets[i].X)

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
            self.assertTrue(dataset.X().equal(Xs[i]))
            self.assertTrue(dataset.Y().equal(Ys[i]))
            self.assertTrue(dataset.Yvar().equal(Ys_var[i]))

        # Test handling of Tensor-valued arguments to `dict_from_iter`
        datasets = FixedNoiseDataset.dict_from_iter(
            X=Xs[0],
            Y=Ys[1],
            Yvar=Ys_var.unbind(),
        )
        for dataset in datasets.values():
            self.assertTrue(Xs[0].equal(dataset.X()))
            self.assertTrue(Ys[1].equal(dataset.Y()))

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
