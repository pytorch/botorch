#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import warnings

import torch
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.acquisition.monte_carlo import (
    qExpectedImprovement,
    qNoisyExpectedImprovement,
)
from botorch.acquisition.multi_objective.max_value_entropy_search import (
    qMultiObjectiveMaxValueEntropy,
)
from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement,
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.exceptions import BotorchError
from botorch.exceptions.warnings import BotorchWarning
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms.input import Warp
from botorch.optim.utils import columnwise_clamp, fix_features, get_X_baseline
from botorch.sampling.normal import IIDNormalSampler
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from botorch.utils.testing import BotorchTestCase, MockModel, MockPosterior


class TestColumnWiseClamp(BotorchTestCase):
    def setUp(self):
        super().setUp()
        self.X = torch.tensor([[-2, 1], [0.5, -0.5]], device=self.device)
        self.X_expected = torch.tensor([[-1, 0.5], [0.5, -0.5]], device=self.device)

    def test_column_wise_clamp_scalars(self):
        X, X_expected = self.X, self.X_expected
        with self.assertRaises(ValueError):
            X_clmp = columnwise_clamp(X, 1, -1)
        X_clmp = columnwise_clamp(X, -1, 0.5)
        self.assertTrue(torch.equal(X_clmp, X_expected))
        X_clmp = columnwise_clamp(X, -3, 3)
        self.assertTrue(torch.equal(X_clmp, X))

    def test_column_wise_clamp_scalar_tensors(self):
        X, X_expected = self.X, self.X_expected
        with self.assertRaises(ValueError):
            X_clmp = columnwise_clamp(X, torch.tensor(1), torch.tensor(-1))
        X_clmp = columnwise_clamp(X, torch.tensor(-1), torch.tensor(0.5))
        self.assertTrue(torch.equal(X_clmp, X_expected))
        X_clmp = columnwise_clamp(X, torch.tensor(-3), torch.tensor(3))
        self.assertTrue(torch.equal(X_clmp, X))

    def test_column_wise_clamp_tensors(self):
        X, X_expected = self.X, self.X_expected
        with self.assertRaises(ValueError):
            X_clmp = columnwise_clamp(X, torch.ones(2), torch.zeros(2))
        with self.assertRaises(RuntimeError):
            X_clmp = columnwise_clamp(X, torch.zeros(3), torch.ones(3))
        X_clmp = columnwise_clamp(X, torch.tensor([-1, -1]), torch.tensor([0.5, 0.5]))
        self.assertTrue(torch.equal(X_clmp, X_expected))
        X_clmp = columnwise_clamp(X, torch.tensor([-3, -3]), torch.tensor([3, 3]))
        self.assertTrue(torch.equal(X_clmp, X))

    def test_column_wise_clamp_full_dim_tensors(self):
        X = torch.tensor([[[-1, 2, 0.5], [0.5, 3, 1.5]], [[0.5, 1, 0], [2, -2, 3]]])
        lower = torch.tensor([[[0, 0.5, 1], [0, 2, 2]], [[0, 2, 0], [1, -1, 0]]])
        upper = torch.tensor([[[1, 1.5, 1], [1, 4, 3]], [[1, 3, 0.5], [3, 1, 2.5]]])
        X_expected = torch.tensor(
            [[[0, 1.5, 1], [0.5, 3, 2]], [[0.5, 2, 0], [2, -1, 2.5]]]
        )
        X_clmp = columnwise_clamp(X, lower, upper)
        self.assertTrue(torch.equal(X_clmp, X_expected))
        X_clmp = columnwise_clamp(X, lower - 5, upper + 5)
        self.assertTrue(torch.equal(X_clmp, X))
        with self.assertRaises(ValueError):
            X_clmp = columnwise_clamp(X, torch.ones_like(X), torch.zeros_like(X))
        with self.assertRaises(RuntimeError):
            X_clmp = columnwise_clamp(X, lower.unsqueeze(-3), upper.unsqueeze(-3))

    def test_column_wise_clamp_raise_on_violation(self):
        X = self.X
        with self.assertRaises(BotorchError):
            X_clmp = columnwise_clamp(
                X, torch.zeros(2), torch.ones(2), raise_on_violation=True
            )
        X_clmp = columnwise_clamp(
            X, torch.tensor([-3, -3]), torch.tensor([3, 3]), raise_on_violation=True
        )
        self.assertTrue(torch.equal(X_clmp, X))


class TestFixFeatures(BotorchTestCase):
    def _getTensors(self):
        X = torch.tensor([[-2, 1, 3], [0.5, -0.5, 1.0]], device=self.device)
        X_null_two = torch.tensor([[-2, 1, 3], [0.5, -0.5, 1.0]], device=self.device)
        X_expected = torch.tensor([[-1, 1, -2], [-1, -0.5, -2]], device=self.device)
        X_expected_null_two = torch.tensor(
            [[-1, 1, 3], [-1, -0.5, 1.0]], device=self.device
        )
        return X, X_null_two, X_expected, X_expected_null_two

    def test_fix_features(self):
        X, X_null_two, X_expected, X_expected_null_two = self._getTensors()
        X.requires_grad_(True)
        X_null_two.requires_grad_(True)

        X_fix = fix_features(X, {0: -1, 2: -2})
        X_fix_null_two = fix_features(X_null_two, {0: -1, 2: None})

        self.assertTrue(torch.equal(X_fix, X_expected))
        self.assertTrue(torch.equal(X_fix_null_two, X_expected_null_two))

        def f(X):
            return X.sum()

        f(X).backward()
        self.assertTrue(torch.equal(X.grad, torch.ones_like(X)))
        X.grad.zero_()

        f(X_fix).backward()
        self.assertTrue(
            torch.equal(
                X.grad,
                torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]], device=self.device),
            )
        )

        f(X_null_two).backward()
        self.assertTrue(torch.equal(X_null_two.grad, torch.ones_like(X)))
        X_null_two.grad.zero_()
        f(X_fix_null_two).backward()
        self.assertTrue(
            torch.equal(
                X_null_two.grad,
                torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]], device=self.device),
            )
        )


class TestGetXBaseline(BotorchTestCase):
    def test_get_X_baseline(self):
        tkwargs = {"device": self.device}
        for dtype in (torch.float, torch.double):
            tkwargs["dtype"] = dtype
            X_train = torch.rand(20, 2, **tkwargs)
            model = MockModel(
                MockPosterior(mean=(2 * X_train + 1).sum(dim=-1, keepdim=True))
            )
            # test NEI with X_baseline
            acqf = qNoisyExpectedImprovement(
                model, X_baseline=X_train[:2], prune_baseline=False, cache_root=False
            )
            X = get_X_baseline(acq_function=acqf)
            self.assertTrue(torch.equal(X, acqf.X_baseline))
            # test EI without X_baseline
            acqf = qExpectedImprovement(model, best_f=0.0)

            with warnings.catch_warnings(record=True) as w:
                X_rnd = get_X_baseline(
                    acq_function=acqf,
                )
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[-1].category, BotorchWarning))
            self.assertIsNone(X_rnd)

            # set train inputs
            model.train_inputs = (X_train,)
            X = get_X_baseline(acq_function=acqf)
            self.assertTrue(torch.equal(X, X_train))
            # test that we fail back to train_inputs if X_baseline is an empty tensor
            acqf.register_buffer("X_baseline", X_train[:0])
            X = get_X_baseline(acq_function=acqf)
            self.assertTrue(torch.equal(X, X_train))

            # test acquisition function without X_baseline or model
            acqf = FixedFeatureAcquisitionFunction(acqf, d=2, columns=[0], values=[0])
            with warnings.catch_warnings(record=True) as w:
                X_rnd = get_X_baseline(
                    acq_function=acqf,
                )
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[-1].category, BotorchWarning))
            self.assertIsNone(X_rnd)

            Y_train = 2 * X_train[:2] + 1
            moo_model = MockModel(MockPosterior(mean=Y_train, samples=Y_train))
            ref_point = torch.zeros(2, **tkwargs)
            # test NEHVI with X_baseline
            acqf = qNoisyExpectedHypervolumeImprovement(
                moo_model,
                ref_point=ref_point,
                X_baseline=X_train[:2],
                sampler=IIDNormalSampler(sample_shape=torch.Size([2])),
                cache_root=False,
            )
            X = get_X_baseline(acq_function=acqf)
            self.assertTrue(torch.equal(X, acqf.X_baseline))
            # test qEHVI without train_inputs
            acqf = qExpectedHypervolumeImprovement(
                moo_model,
                ref_point=ref_point,
                partitioning=FastNondominatedPartitioning(
                    ref_point=ref_point,
                    Y=Y_train,
                ),
            )
            # test extracting train_inputs from model list GP
            model_list = ModelListGP(
                SingleTaskGP(X_train, Y_train[:, :1]),
                SingleTaskGP(X_train, Y_train[:, 1:]),
            )
            acqf = qExpectedHypervolumeImprovement(
                model_list,
                ref_point=ref_point,
                partitioning=FastNondominatedPartitioning(
                    ref_point=ref_point,
                    Y=Y_train,
                ),
            )
            X = get_X_baseline(acq_function=acqf)
            self.assertTrue(torch.equal(X, X_train))

            # test MESMO for which we need to use
            # `acqf.mo_model`
            batched_mo_model = SingleTaskGP(X_train, Y_train, outcome_transform=None)
            acqf = qMultiObjectiveMaxValueEntropy(
                batched_mo_model,
                sample_pareto_frontiers=lambda model: torch.rand(10, 2, **tkwargs),
            )
            X = get_X_baseline(acq_function=acqf)
            self.assertTrue(torch.equal(X, X_train))
            # test that if there is an input transform that is applied
            # to the train_inputs when the model is in eval mode, we
            # extract the untransformed train_inputs
            model = SingleTaskGP(
                X_train,
                Y_train[:, :1],
                input_transform=Warp(d=X_train.shape[-1], indices=[0, 1]),
            )
            model.eval()
            self.assertFalse(torch.equal(model.train_inputs[0], X_train))
            acqf = qExpectedImprovement(model, best_f=0.0)
            X = get_X_baseline(acq_function=acqf)
            self.assertTrue(torch.equal(X, X_train))
