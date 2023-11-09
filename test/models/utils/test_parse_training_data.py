#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from botorch.models.model import Model
from botorch.models.pairwise_gp import PairwiseGP
from botorch.models.utils.parse_training_data import parse_training_data
from botorch.utils.containers import SliceContainer
from botorch.utils.datasets import RankingDataset, SupervisedDataset
from botorch.utils.testing import BotorchTestCase
from torch import long, rand, Size, tensor


class TestParseTrainingData(BotorchTestCase):
    def test_supervised(self):
        with self.assertRaisesRegex(NotImplementedError, "Could not find signature"):
            parse_training_data(Model, None)

        dataset = SupervisedDataset(
            X=rand(3, 2), Y=rand(3, 1), feature_names=["a", "b"], outcome_names=["y"]
        )
        with self.assertRaisesRegex(NotImplementedError, "Could not find signature"):
            parse_training_data(None, dataset)

        parse = parse_training_data(Model, dataset)
        self.assertIsInstance(parse, dict)
        self.assertTrue(torch.equal(dataset.X, parse["train_X"]))
        self.assertTrue(torch.equal(dataset.Y, parse["train_Y"]))
        self.assertTrue("train_Yvar" not in parse)

        # Test with noise
        dataset = SupervisedDataset(
            X=rand(3, 2),
            Y=rand(3, 1),
            Yvar=rand(3, 1),
            feature_names=["a", "b"],
            outcome_names=["y"],
        )
        parse = parse_training_data(Model, dataset)
        self.assertTrue(torch.equal(dataset.X, parse["train_X"]))
        self.assertTrue(torch.equal(dataset.Y, parse["train_Y"]))
        self.assertTrue(torch.equal(dataset.Yvar, parse["train_Yvar"]))

    def test_pairwiseGP_ranking(self):
        # Test parsing Ranking Dataset for PairwiseGP
        datapoints = rand(3, 2)
        indices = tensor([[0, 1], [1, 2]], dtype=long)
        event_shape = Size([2 * datapoints.shape[-1]])
        dataset_X = SliceContainer(datapoints, indices, event_shape=event_shape)
        dataset_Y = tensor([[0, 1], [1, 0]]).expand(indices.shape)
        dataset = RankingDataset(
            X=dataset_X, Y=dataset_Y, feature_names=["a", "b"], outcome_names=["y"]
        )
        parse = parse_training_data(PairwiseGP, dataset)
        self.assertTrue(dataset._X.values.equal(parse["datapoints"]))

        comparisons = tensor([[0, 1], [2, 1]], dtype=long)
        self.assertTrue(comparisons.equal(parse["comparisons"]))
