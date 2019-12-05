#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from botorch.exceptions.errors import UnsupportedError
from botorch.models.deterministic import (
    AffineDeterministicModel,
    DeterministicModel,
    GenericDeterministicModel,
)
from botorch.posteriors.deterministic import DeterministicPosterior
from botorch.utils.testing import BotorchTestCase


class TestDeterministicModels(BotorchTestCase):
    def test_abstract_base_model(self):
        with self.assertRaises(TypeError):
            DeterministicModel()

    def test_GenericDeterministicModel(self):
        def f(X):
            return X.mean(dim=-1, keepdim=True)

        model = GenericDeterministicModel(f)
        self.assertEqual(model.num_outputs, 1)
        X = torch.rand(3, 2)
        # basic test
        p = model.posterior(X)
        self.assertIsInstance(p, DeterministicPosterior)
        self.assertTrue(torch.equal(p.mean, f(X)))
        # check that w/ observation noise this errors properly
        with self.assertRaises(UnsupportedError):
            model.posterior(X, observation_noise=True)
        # check output indices
        model = GenericDeterministicModel(lambda X: X, num_outputs=2)
        self.assertEqual(model.num_outputs, 2)
        p = model.posterior(X, output_indices=[0])
        self.assertTrue(torch.equal(p.mean, X[..., [0]]))
        # test subset output
        subset_model = model.subset_output([0])
        self.assertIsInstance(subset_model, GenericDeterministicModel)
        p_sub = subset_model.posterior(X)
        self.assertTrue(torch.equal(p_sub.mean, X[..., [0]]))

    def test_AffineDeterministicModel(self):
        # test error on bad shape of a
        with self.assertRaises(ValueError):
            AffineDeterministicModel(torch.rand(2))
        # test error on bad shape of b
        with self.assertRaises(ValueError):
            AffineDeterministicModel(torch.rand(2, 1), torch.rand(2, 1))
        # test one-dim output
        a = torch.rand(3, 1)
        model = AffineDeterministicModel(a)
        self.assertEqual(model.num_outputs, 1)
        for shape in ((4, 3), (1, 4, 3)):
            X = torch.rand(*shape)
            p = model.posterior(X)
            mean_exp = model.b + (X.unsqueeze(-1) * a).sum(dim=-2)
            self.assertTrue(torch.equal(p.mean, mean_exp))
        # # test two-dim output
        a = torch.rand(3, 2)
        model = AffineDeterministicModel(a)
        self.assertEqual(model.num_outputs, 2)
        for shape in ((4, 3), (1, 4, 3)):
            X = torch.rand(*shape)
            p = model.posterior(X)
            mean_exp = model.b + (X.unsqueeze(-1) * a).sum(dim=-2)
            self.assertTrue(torch.equal(p.mean, mean_exp))
        # test subset output
        X = torch.rand(4, 3)
        subset_model = model.subset_output([0])
        self.assertIsInstance(subset_model, AffineDeterministicModel)
        p = model.posterior(X)
        p_sub = subset_model.posterior(X)
        self.assertTrue(torch.equal(p_sub.mean, p.mean[..., [0]]))
