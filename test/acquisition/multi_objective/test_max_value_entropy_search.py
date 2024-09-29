#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from itertools import product
from unittest import mock

import torch
from botorch.acquisition.max_value_entropy_search import qMaxValueEntropy
from botorch.acquisition.multi_objective.max_value_entropy_search import (
    qLowerBoundMultiObjectiveMaxValueEntropySearch,
    qMultiObjectiveMaxValueEntropy,
)
from botorch.acquisition.multi_objective.utils import compute_sample_box_decomposition
from botorch.exceptions.errors import UnsupportedError
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.test_helpers import get_model
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
            model = SingleTaskGP(train_X, train_Y, outcome_transform=None)
            with self.assertRaises(NotImplementedError):
                qMultiObjectiveMaxValueEntropy(
                    model=model, sample_pareto_frontiers=dummy_sample_pareto_frontiers
                )
            # test initialization
            train_X = torch.rand(4, 2, dtype=dtype, device=self.device)
            train_Y = torch.rand(4, m, dtype=dtype, device=self.device)
            # Models with outcome transforms aren't supported.
            model = SingleTaskGP(train_X, train_Y)
            with self.assertRaisesRegex(
                UnsupportedError,
                "Conversion of models with outcome transforms is unsupported. "
                "To fix this error, explicitly pass `outcome_transform=None`.",
            ):
                qMultiObjectiveMaxValueEntropy(
                    model=ModelListGP(model, model),
                    sample_pareto_frontiers=dummy_sample_pareto_frontiers,
                )
            # test batched MO model
            model = SingleTaskGP(train_X, train_Y, outcome_transform=None)
            mesmo = qMultiObjectiveMaxValueEntropy(
                model=model, sample_pareto_frontiers=dummy_sample_pareto_frontiers
            )
            self.assertEqual(mesmo.num_fantasies, 16)
            # Initialize the sampler.
            dummy_post = model.posterior(train_X[:1])
            mesmo.get_posterior_samples(dummy_post)
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
                *[
                    SingleTaskGP(train_X, train_Y[:, i : i + 1], outcome_transform=None)
                    for i in range(m)
                ]
            )
            mock_sample_pfs = mock.Mock()
            mock_sample_pfs.return_value = dummy_sample_pareto_frontiers(model=model)
            mesmo = qMultiObjectiveMaxValueEntropy(
                model=model, sample_pareto_frontiers=mock_sample_pfs
            )
            self.assertEqual(mesmo.num_fantasies, 16)
            # Initialize the sampler.
            dummy_post = model.posterior(train_X[:1])
            mesmo.get_posterior_samples(dummy_post)
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
                    torch.rand(16, 1, 2, device=self.device, dtype=dtype),
                ],
                dim=1,
            )
            fant_Y = torch.cat(
                [
                    train_Y.expand(16, 4, m),
                    torch.rand(16, 1, m, device=self.device, dtype=dtype),
                ],
                dim=1,
            )
            fantasy_model = SingleTaskGP(fant_X, fant_Y, outcome_transform=None)

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


class TestQLowerBoundMultiObjectiveMaxValueEntropySearch(BotorchTestCase):
    def _base_test_lb_moo_max_value_entropy_search(self, estimation_type):
        torch.manual_seed(1)
        tkwargs = {"device": self.device}

        for dtype, num_objectives, use_model_list, standardize_model in product(
            (torch.float, torch.double),
            (1, 2, 3),
            (False, True),
            (False, True),
        ):
            tkwargs["dtype"] = dtype
            input_dim = 2
            train_X = torch.rand(4, input_dim, **tkwargs)
            train_Y = torch.rand(4, num_objectives, **tkwargs)
            model = get_model(
                train_X=train_X,
                train_Y=train_Y,
                use_model_list=use_model_list,
                standardize_model=standardize_model,
            )
            pareto_fronts = dummy_sample_pareto_frontiers(model)
            hypercell_bounds = compute_sample_box_decomposition(pareto_fronts)

            # test acquisition
            X_pending_list = [None, torch.rand(2, input_dim, **tkwargs)]
            for X_pending in X_pending_list:
                acq = qLowerBoundMultiObjectiveMaxValueEntropySearch(
                    model=model,
                    hypercell_bounds=hypercell_bounds,
                    estimation_type=estimation_type,
                    num_samples=64,
                    X_pending=X_pending,
                )
                self.assertIsInstance(acq.sampler, SobolQMCNormalSampler)

                test_Xs = [
                    torch.rand(4, 1, input_dim, **tkwargs),
                    torch.rand(4, 3, input_dim, **tkwargs),
                    torch.rand(4, 5, 1, input_dim, **tkwargs),
                    torch.rand(4, 5, 3, input_dim, **tkwargs),
                ]

                for test_X in test_Xs:
                    acq_X = acq(test_X)
                    # assess shape
                    self.assertTrue(acq_X.shape == test_X.shape[:-2])

    def test_lb_moo_max_value_entropy_search_0(self):
        self._base_test_lb_moo_max_value_entropy_search(estimation_type="0")

    def test_lb_moo_max_value_entropy_search_LB(self):
        self._base_test_lb_moo_max_value_entropy_search(estimation_type="LB")

    def test_lb_moo_max_value_entropy_search_LB2(self):
        self._base_test_lb_moo_max_value_entropy_search(estimation_type="LB2")

    def test_lb_moo_max_value_entropy_search_MC(self):
        self._base_test_lb_moo_max_value_entropy_search(estimation_type="MC")
