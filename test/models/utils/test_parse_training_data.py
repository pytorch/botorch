#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from botorch.exceptions import UnsupportedError
from botorch.models.gp_regression import FixedNoiseGP
from botorch.models.model import Model
from botorch.models.multitask import MultiTaskGP
from botorch.models.pairwise_gp import PairwiseGP
from botorch.models.utils.parse_training_data import parse_training_data
from botorch.utils.containers import SliceContainer
from botorch.utils.datasets import FixedNoiseDataset, RankingDataset, SupervisedDataset
from botorch.utils.testing import BotorchTestCase
from torch import cat, long, rand, Size, tensor


class TestParseTrainingData(BotorchTestCase):
    def test_supervised(self):
        with self.assertRaisesRegex(NotImplementedError, "Could not find signature"):
            parse_training_data(Model, None)

        dataset = SupervisedDataset(X=rand(3, 2), Y=rand(3, 1))
        with self.assertRaisesRegex(NotImplementedError, "Could not find signature"):
            parse_training_data(None, dataset)

        parse = parse_training_data(Model, dataset)
        self.assertIsInstance(parse, dict)
        self.assertTrue(dataset.X().equal(parse["train_X"]))
        self.assertTrue(dataset.Y().equal(parse["train_Y"]))

    def test_fixedNoise(self):
        # Test passing a `SupervisedDataset`
        dataset = SupervisedDataset(X=rand(3, 2), Y=rand(3, 1))
        parse = parse_training_data(FixedNoiseGP, dataset)
        self.assertTrue("train_Yvar" not in parse)
        self.assertTrue(dataset.X().equal(parse["train_X"]))
        self.assertTrue(dataset.Y().equal(parse["train_Y"]))

        # Test passing a `FixedNoiseDataset`
        dataset = FixedNoiseDataset(X=rand(3, 2), Y=rand(3, 1), Yvar=rand(3, 1))
        parse = parse_training_data(FixedNoiseGP, dataset)
        self.assertTrue(dataset.X().equal(parse["train_X"]))
        self.assertTrue(dataset.Y().equal(parse["train_Y"]))
        self.assertTrue(dataset.Yvar().equal(parse["train_Yvar"]))

    def test_pairwiseGP_ranking(self):
        # Test parsing Ranking Dataset for PairwiseGP
        datapoints = rand(3, 2)
        indices = tensor([[0, 1], [1, 2]], dtype=long)
        event_shape = Size([2 * datapoints.shape[-1]])
        dataset_X = SliceContainer(datapoints, indices, event_shape=event_shape)
        dataset_Y = tensor([[0, 1], [1, 0]]).expand(indices.shape)
        dataset = RankingDataset(X=dataset_X, Y=dataset_Y)
        parse = parse_training_data(PairwiseGP, dataset)
        self.assertTrue(dataset.X.values.equal(parse["datapoints"]))

        comparisons = tensor([[0, 1], [2, 1]], dtype=long)
        self.assertTrue(comparisons.equal(parse["comparisons"]))

    def test_dict(self):
        n = 3
        m = 2
        datasets = {i: SupervisedDataset(X=rand(n, 2), Y=rand(n, 1)) for i in range(m)}
        parse_training_data(Model, {0: datasets[0]})
        with self.assertRaisesRegex(UnsupportedError, "multiple datasets to single"):
            parse_training_data(Model, datasets)

        with self.assertRaisesRegex(UnsupportedError, "Cannot combine .* hetero"):
            _datasets = datasets.copy()
            _datasets[m] = FixedNoiseDataset(rand(n, 2), rand(n, 1), rand(n, 1))
            parse_training_data(MultiTaskGP, _datasets)

        with self.assertRaisesRegex(ValueError, "Missing required term"):
            parse_training_data(MultiTaskGP, datasets, task_feature_container="foo")

        with self.assertRaisesRegex(ValueError, "out-of-bounds"):
            parse_training_data(MultiTaskGP, datasets, task_feature=-m - 2)

        with self.assertRaisesRegex(ValueError, "out-of-bounds"):
            parse_training_data(MultiTaskGP, datasets, task_feature=m + 1)

        X = cat([dataset.X() for dataset in datasets.values()])
        Y = cat([dataset.Y() for dataset in datasets.values()])
        for i in (0, 1, 2):
            parse = parse_training_data(MultiTaskGP, datasets, task_feature=i)
            self.assertTrue(Y.equal(parse["train_Y"]))

            X2 = cat([parse["train_X"][..., :i], parse["train_X"][..., i + 1 :]], -1)
            self.assertTrue(X.equal(X2))
            for j, task_features in enumerate(parse["train_X"][..., i].split(n)):
                self.assertTrue(task_features.eq(j).all())
