#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from itertools import product
from unittest import mock

import torch
from botorch.acquisition.max_value_entropy_search import qMaxValueEntropy
from botorch.acquisition.multi_objective.max_value_entropy_search import (
    qMultiObjectiveMaxValueEntropy,
)
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.utils.testing import BotorchTestCase


def dummy_sample_pareto_frontiers(model):
    m = model.models[0] if isinstance(model, ModelListGP) else model
    return torch.rand(
        3,
        4,
        model.num_outputs,
        dtype=m.train_inputs[0].dtype,
        device=m.train_inputs[0].device,
    )


class TestMultiObjectiveMaxValueEntropy(BotorchTestCase):
    def test_multi_objective_max_value_entropy(self):
        for dtype, m in product((torch.float, torch.double), (2, 3)):
            torch.manual_seed(7)
            # test batched model
            train_X = torch.rand(1, 1, 2, dtype=dtype, device=self.device)
            train_Y = torch.rand(1, 1, m, dtype=dtype, device=self.device)
            model = SingleTaskGP(train_X, train_Y)
            with self.assertRaises(NotImplementedError):
                qMultiObjectiveMaxValueEntropy(model, dummy_sample_pareto_frontiers)
            # test initialization
            train_X = torch.rand(4, 2, dtype=dtype, device=self.device)
            train_Y = torch.rand(4, m, dtype=dtype, device=self.device)
            # test batched MO model
            model = SingleTaskGP(train_X, train_Y)
            mesmo = qMultiObjectiveMaxValueEntropy(model, dummy_sample_pareto_frontiers)
            self.assertEqual(mesmo.num_fantasies, 16)
            self.assertIsInstance(mesmo.sampler, SobolQMCNormalSampler)
            self.assertEqual(mesmo.sampler.sample_shape, torch.Size([128]))
            self.assertIsInstance(mesmo.fantasies_sampler, SobolQMCNormalSampler)
            self.assertEqual(mesmo.posterior_max_values.shape, torch.Size([3, 1, m]))
            # test conversion to single-output model
            self.assertIs(mesmo.mo_model, model)
            self.assertEqual(mesmo.mo_model.num_outputs, m)
            self.assertIsInstance(mesmo.model, SingleTaskGP)
            self.assertEqual(mesmo.model.num_outputs, 1)
            self.assertEqual(
                mesmo.model._aug_batch_shape, mesmo.model._input_batch_shape
            )
            # test ModelListGP
            model = ModelListGP(
                *[SingleTaskGP(train_X, train_Y[:, i : i + 1]) for i in range(m)]
            )
            mock_sample_pfs = mock.Mock()
            mock_sample_pfs.return_value = dummy_sample_pareto_frontiers(model=model)
            mesmo = qMultiObjectiveMaxValueEntropy(model, mock_sample_pfs)
            self.assertEqual(mesmo.num_fantasies, 16)
            self.assertIsInstance(mesmo.sampler, SobolQMCNormalSampler)
            self.assertEqual(mesmo.sampler.sample_shape, torch.Size([128]))
            self.assertIsInstance(mesmo.fantasies_sampler, SobolQMCNormalSampler)
            self.assertEqual(mesmo.posterior_max_values.shape, torch.Size([3, 1, m]))
            # test conversion to batched MO model
            self.assertIsInstance(mesmo.mo_model, SingleTaskGP)
            self.assertEqual(mesmo.mo_model.num_outputs, m)
            self.assertIs(mesmo.mo_model, mesmo._init_model)
            # test conversion to single-output model
            self.assertIsInstance(mesmo.model, SingleTaskGP)
            self.assertEqual(mesmo.model.num_outputs, 1)
            self.assertEqual(
                mesmo.model._aug_batch_shape, mesmo.model._input_batch_shape
            )
            # test that we call sample_pareto_frontiers with the multi-output model
            mock_sample_pfs.assert_called_once_with(mesmo.mo_model)
            # test basic evaluation
            X = torch.rand(1, 2, device=self.device, dtype=dtype)
            with torch.no_grad():
                vals = mesmo(X)
                igs = qMaxValueEntropy.forward(mesmo, X=X.view(1, 1, 1, 2))
            self.assertEqual(vals.shape, torch.Size([1]))
            self.assertTrue(torch.equal(vals, igs.sum(dim=-1)))

            # test batched evaluation
            X = torch.rand(4, 1, 2, device=self.device, dtype=dtype)
            with torch.no_grad():
                vals = mesmo(X)
                igs = qMaxValueEntropy.forward(mesmo, X=X.view(4, 1, 1, 2))
            self.assertEqual(vals.shape, torch.Size([4]))
            self.assertTrue(torch.equal(vals, igs.sum(dim=-1)))

            # test set X pending to None
            mesmo.set_X_pending(None)
            self.assertIs(mesmo.mo_model, mesmo._init_model)
            fant_X = torch.cat(
                [
                    train_X.expand(16, 4, 2),
                    torch.rand(16, 1, 2),
                ],
                dim=1,
            )
            fant_Y = torch.cat(
                [
                    train_Y.expand(16, 4, m),
                    torch.rand(16, 1, m),
                ],
                dim=1,
            )
            fantasy_model = SingleTaskGP(fant_X, fant_Y)

            # test with X_pending is not None
            with mock.patch.object(
                SingleTaskGP, "fantasize", return_value=fantasy_model
            ) as mock_fantasize:
                qMultiObjectiveMaxValueEntropy(
                    model,
                    dummy_sample_pareto_frontiers,
                    X_pending=torch.rand(1, 2, device=self.device, dtype=dtype),
                )
                mock_fantasize.assert_called_once()
