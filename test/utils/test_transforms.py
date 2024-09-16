#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

import torch
from botorch.models import (
    GenericDeterministicModel,
    HigherOrderGP,
    ModelList,
    PairwiseGP,
    SaasFullyBayesianSingleTaskGP,
    SingleTaskGP,
)
from botorch.models.fully_bayesian_multitask import SaasFullyBayesianMultiTaskGP
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.gp_regression_mixed import MixedSingleTaskGP
from botorch.models.model import Model
from botorch.models.multitask import MultiTaskGP
from botorch.utils.testing import BotorchTestCase, MockModel, MockPosterior
from botorch.utils.transforms import (
    _verify_output_shape,
    concatenate_pending_points,
    is_ensemble,
    is_fully_bayesian,
    match_batch_shape,
    normalize,
    normalize_indices,
    standardize,
    t_batch_mode_transform,
    unnormalize,
)
from torch import Tensor


class TestStandardize(BotorchTestCase):
    def test_standardize(self):
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            Y = torch.tensor([0.0, 0.0], **tkwargs)
            self.assertTrue(torch.equal(Y, standardize(Y)))
            Y2 = torch.tensor([0.0, 1.0, 1.0, 1.0], **tkwargs)
            expected_Y2_stdized = torch.tensor([-1.5, 0.5, 0.5, 0.5], **tkwargs)
            self.assertTrue(torch.equal(expected_Y2_stdized, standardize(Y2)))
            Y3 = torch.tensor(
                [[0.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]], **tkwargs
            ).transpose(1, 0)
            Y3_stdized = standardize(Y3)
            self.assertTrue(torch.equal(Y3_stdized[:, 0], expected_Y2_stdized))
            self.assertTrue(torch.equal(Y3_stdized[:, 1], torch.zeros(4, **tkwargs)))
            Y4 = torch.cat([Y3, Y2.unsqueeze(-1)], dim=-1)
            Y4_stdized = standardize(Y4)
            self.assertTrue(torch.equal(Y4_stdized[:, 0], expected_Y2_stdized))
            self.assertTrue(torch.equal(Y4_stdized[:, 1], torch.zeros(4, **tkwargs)))
            self.assertTrue(torch.equal(Y4_stdized[:, 2], expected_Y2_stdized))


class TestNormalizeAndUnnormalize(BotorchTestCase):
    def test_normalize_unnormalize(self) -> None:
        for dtype in (torch.float, torch.double):
            X = torch.tensor([0.0, 0.25, 0.5], device=self.device, dtype=dtype).view(
                -1, 1
            )
            expected_X_normalized = torch.tensor(
                [0.0, 0.5, 1.0], device=self.device, dtype=dtype
            ).view(-1, 1)
            bounds = torch.tensor([0.0, 0.5], device=self.device, dtype=dtype).view(
                -1, 1
            )
            X_normalized = normalize(X, bounds=bounds)
            self.assertTrue(torch.equal(expected_X_normalized, X_normalized))
            self.assertTrue(torch.equal(X, unnormalize(X_normalized, bounds=bounds)))
            X2 = torch.tensor(
                [[0.25, 0.125, 0.0], [0.25, 0.0, 0.5]], device=self.device, dtype=dtype
            ).transpose(1, 0)
            expected_X2_normalized = torch.tensor(
                [[1.0, 0.5, 0.0], [0.5, 0.0, 1.0]], device=self.device, dtype=dtype
            ).transpose(1, 0)
            bounds2 = torch.tensor(
                [[0.0, 0.0], [0.25, 0.5]], device=self.device, dtype=dtype
            )
            X2_normalized = normalize(X2, bounds=bounds2)
            self.assertTrue(torch.equal(X2_normalized, expected_X2_normalized))
            self.assertTrue(torch.equal(X2, unnormalize(X2_normalized, bounds=bounds2)))

    def test_with_constant_bounds(self) -> None:
        X = torch.rand(10, 2, dtype=torch.double)
        # First dimension is constant, second has a range of 1.
        # The transform should just add 1 to each dimension.
        bounds = -torch.ones(2, 2, dtype=torch.double)
        bounds[1, 1] = 0.0
        X_normalized = normalize(X, bounds=bounds)
        self.assertAllClose(X_normalized, X + 1)
        X_unnormalized = unnormalize(X_normalized, bounds=bounds)
        self.assertAllClose(X_unnormalized, X)


class BMIMTestClass(BotorchTestCase):
    @t_batch_mode_transform(assert_output_shape=False)
    def q_method(self, X: Tensor) -> None:
        return X

    @t_batch_mode_transform(expected_q=1, assert_output_shape=False)
    def q1_method(self, X: Tensor) -> None:
        return X

    @t_batch_mode_transform(assert_output_shape=False)
    def kw_method(self, X: Tensor, dummy_arg: Any = None):
        self.assertIsNotNone(dummy_arg)
        return X

    @t_batch_mode_transform(assert_output_shape=True)
    def wrong_shape_method(self, X: Tensor):
        return X

    @t_batch_mode_transform(assert_output_shape=True)
    def correct_shape_method(self, X: Tensor):
        return X.mean(dim=(-1, -2)).squeeze(-1)

    @concatenate_pending_points
    def dummy_method(self, X: Tensor) -> Tensor:
        return X

    @t_batch_mode_transform(assert_output_shape=True)
    def broadcast_batch_shape_method(self, X: Tensor):
        return X.mean(dim=(-1, -2)).repeat(2, *[1] * (X.dim() - 2))


class NotSoAbstractBaseModel(Model):
    def posterior(self, X, output_indices, observation_noise, **kwargs):
        pass

    @property
    def batch_shape(self) -> torch.Size():
        if hasattr(self, "_batch_shape"):
            return self._batch_shape
        else:
            return super().batch_shape


class TestBatchModeTransform(BotorchTestCase):
    def test_verify_output_shape(self):
        # output shape matching t-batch shape of X
        self.assertTrue(
            _verify_output_shape(acqf=None, X=torch.ones(3, 2, 1), output=torch.ones(3))
        )
        # output shape is [], t-batch shape of X is [1]
        X = torch.ones(1, 1, 1)
        self.assertTrue(_verify_output_shape(acqf=None, X=X, output=torch.tensor(1)))
        # shape mismatch and cls does not have model attribute
        acqf = BMIMTestClass()
        with self.assertWarns(RuntimeWarning):
            self.assertTrue(_verify_output_shape(acqf=acqf, X=X, output=X))
        # shape mismatch and cls.model does not define batch shape
        acqf.model = NotSoAbstractBaseModel()
        with self.assertWarns(RuntimeWarning):
            self.assertTrue(_verify_output_shape(acqf=acqf, X=X, output=X))
        # Output matches model batch shape.
        acqf.model._batch_shape = torch.Size([3, 5])
        self.assertTrue(_verify_output_shape(acqf=acqf, X=X, output=torch.empty(3, 5)))
        # Output has additional dimensions beyond model batch shape.
        for X_batch in [(2, 3, 5), (2, 1, 5), (2, 1, 1)]:
            self.assertTrue(
                _verify_output_shape(
                    acqf=acqf,
                    X=torch.empty(*X_batch, 1, 1),
                    output=torch.empty(2, 3, 5),
                )
            )

    def test_t_batch_mode_transform(self):
        c = BMIMTestClass()
        # test with q != 1
        # non-batch
        X = torch.rand(3, 2)
        Xout = c.q_method(X)
        self.assertTrue(torch.equal(Xout, X.unsqueeze(0)))
        # test with expected_q = 1
        with self.assertRaises(AssertionError):
            c.q1_method(X)
        # batch
        X = X.unsqueeze(0)
        Xout = c.q_method(X)
        self.assertTrue(torch.equal(Xout, X))
        # test with expected_q = 1
        with self.assertRaises(AssertionError):
            c.q1_method(X)

        # test with q = 1
        X = torch.rand(1, 2)
        Xout = c.q_method(X)
        self.assertTrue(torch.equal(Xout, X.unsqueeze(0)))
        # test with expected_q = 1
        Xout = c.q1_method(X)
        self.assertTrue(torch.equal(Xout, X.unsqueeze(0)))
        # batch
        X = X.unsqueeze(0)
        Xout = c.q_method(X)
        self.assertTrue(torch.equal(Xout, X))
        # test with expected_q = 1
        Xout = c.q1_method(X)
        self.assertTrue(torch.equal(Xout, X))

        # test single-dim
        X = torch.zeros(1)
        with self.assertRaises(ValueError):
            c.q_method(X)

        # test with kwargs
        X = torch.rand(1, 2)
        with self.assertRaises(AssertionError):
            c.kw_method(X)
        Xout = c.kw_method(X, dummy_arg=5)
        self.assertTrue(torch.equal(Xout, X.unsqueeze(0)))

        # test assert_output_shape
        X = torch.rand(5, 1, 2)
        with self.assertWarns(RuntimeWarning):
            c.wrong_shape_method(X)
        Xout = c.correct_shape_method(X)
        self.assertEqual(Xout.shape, X.shape[:-2])
        # test when output shape is torch.Size()
        Xout = c.correct_shape_method(torch.rand(1, 2))
        self.assertEqual(Xout.shape, torch.Size())
        # test with model batch shape
        c.model = MockModel(MockPosterior(mean=X))
        with self.assertRaises(AssertionError):
            c.broadcast_batch_shape_method(X)
        c.model = MockModel(MockPosterior(mean=X.repeat(2, *[1] * X.dim())))
        Xout = c.broadcast_batch_shape_method(X)
        self.assertEqual(Xout.shape, c.model.batch_shape)

        # test with non-tensor argument
        X = ((3, 4), {"foo": True})
        Xout = c.q_method(X)
        self.assertEqual(X, Xout)


class TestConcatenatePendingPoints(BotorchTestCase):
    def test_concatenate_pending_points(self):
        c = BMIMTestClass()
        # test if no pending points
        c.X_pending = None
        X = torch.rand(1, 2)
        self.assertTrue(torch.equal(c.dummy_method(X), X))
        # basic test
        X_pending = torch.rand(2, 2)
        c.X_pending = X_pending
        X_expected = torch.cat([X, X_pending], dim=-2)
        self.assertTrue(torch.equal(c.dummy_method(X), X_expected))
        # batch test
        X = torch.rand(2, 1, 2)
        X_expected = torch.cat([X, X_pending.expand(2, 2, 2)], dim=-2)
        self.assertTrue(torch.equal(c.dummy_method(X), X_expected))


class TestMatchBatchShape(BotorchTestCase):
    def test_match_batch_shape(self):
        X = torch.rand(3, 2)
        Y = torch.rand(1, 3, 2)
        X_tf = match_batch_shape(X, Y)
        self.assertTrue(torch.equal(X_tf, X.unsqueeze(0)))

        X = torch.rand(1, 3, 2)
        Y = torch.rand(2, 3, 2)
        X_tf = match_batch_shape(X, Y)
        self.assertTrue(torch.equal(X_tf, X.repeat(2, 1, 1)))

        X = torch.rand(2, 3, 2)
        Y = torch.rand(1, 3, 2)
        with self.assertRaises(RuntimeError):
            match_batch_shape(X, Y)

    def test_match_batch_shape_multi_dim(self):
        X = torch.rand(1, 3, 2)
        Y = torch.rand(5, 4, 3, 2)
        X_tf = match_batch_shape(X, Y)
        self.assertTrue(torch.equal(X_tf, X.expand(5, 4, 3, 2)))

        X = torch.rand(4, 3, 2)
        Y = torch.rand(5, 4, 3, 2)
        X_tf = match_batch_shape(X, Y)
        self.assertTrue(torch.equal(X_tf, X.repeat(5, 1, 1, 1)))

        X = torch.rand(2, 1, 3, 2)
        Y = torch.rand(2, 4, 3, 2)
        X_tf = match_batch_shape(X, Y)
        self.assertTrue(torch.equal(X_tf, X.repeat(1, 4, 1, 1)))

        X = torch.rand(4, 2, 3, 2)
        Y = torch.rand(4, 3, 3, 2)
        with self.assertRaises(RuntimeError):
            match_batch_shape(X, Y)


class TorchNormalizeIndices(BotorchTestCase):
    def test_normalize_indices(self):
        self.assertIsNone(normalize_indices(None, 3))
        indices = [0, 2]
        nlzd_indices = normalize_indices(indices, 3)
        self.assertEqual(nlzd_indices, indices)
        nlzd_indices = normalize_indices(indices, 4)
        self.assertEqual(nlzd_indices, indices)
        indices = [0, -1]
        nlzd_indices = normalize_indices(indices, 3)
        self.assertEqual(nlzd_indices, [0, 2])
        with self.assertRaises(ValueError):
            nlzd_indices = normalize_indices([3], 3)
        with self.assertRaises(ValueError):
            nlzd_indices = normalize_indices([-4], 3)


class TestIsFullyBayesian(BotorchTestCase):
    def test_is_fully_bayesian(self):
        X, Y = torch.rand(3, 2), torch.randn(3, 1)
        vanilla_gp = SingleTaskGP(train_X=X, train_Y=Y)
        deterministic = GenericDeterministicModel(f=lambda x: x)

        fully_bayesian_models = (
            SaasFullyBayesianSingleTaskGP(train_X=X, train_Y=Y),
            SaasFullyBayesianMultiTaskGP(train_X=X, train_Y=Y, task_feature=-1),
        )
        for m in fully_bayesian_models:
            self.assertTrue(is_fully_bayesian(model=m))
            # ModelList
            self.assertTrue(is_fully_bayesian(model=ModelList(m, m)))
            self.assertTrue(is_fully_bayesian(model=ModelList(m, vanilla_gp)))
            self.assertTrue(is_fully_bayesian(model=ModelList(m, deterministic)))
            # Nested ModelList
            self.assertTrue(is_fully_bayesian(model=ModelList(ModelList(m), m)))
            self.assertTrue(
                is_fully_bayesian(model=ModelList(ModelList(m), deterministic))
            )

        non_fully_bayesian_models = (
            GenericDeterministicModel(f=lambda x: x),
            SingleTaskGP(train_X=X, train_Y=Y),
            MultiTaskGP(train_X=X, train_Y=Y, task_feature=-1),
            HigherOrderGP(train_X=X, train_Y=Y),
            SingleTaskMultiFidelityGP(train_X=X, train_Y=Y, data_fidelities=[3]),
            MixedSingleTaskGP(train_X=X, train_Y=Y, cat_dims=[1]),
            PairwiseGP(datapoints=X, comparisons=None),
        )
        for m in non_fully_bayesian_models:
            self.assertFalse(is_fully_bayesian(model=m))
            # ModelList
            self.assertFalse(is_fully_bayesian(model=ModelList(m, m)))
            self.assertFalse(is_fully_bayesian(model=ModelList(m, vanilla_gp)))
            self.assertFalse(is_fully_bayesian(model=ModelList(m, deterministic)))
            # Nested ModelList
            self.assertFalse(is_fully_bayesian(model=ModelList(ModelList(m), m)))
            self.assertFalse(
                is_fully_bayesian(model=ModelList(ModelList(m), deterministic))
            )


class TestIsEnsemble(BotorchTestCase):
    def test_is_ensemble(self):
        X, Y = torch.rand(3, 2), torch.randn(3, 1)
        vanilla_gp = SingleTaskGP(train_X=X, train_Y=Y)
        deterministic = GenericDeterministicModel(f=lambda x: x)

        ensemble_models = (
            SaasFullyBayesianSingleTaskGP(train_X=X, train_Y=Y),
            SaasFullyBayesianMultiTaskGP(train_X=X, train_Y=Y, task_feature=-1),
        )
        for m in ensemble_models:
            self.assertTrue(is_ensemble(model=m))
            # ModelList
            self.assertTrue(is_ensemble(model=ModelList(m, m)))
            self.assertTrue(is_ensemble(model=ModelList(m, vanilla_gp)))
            self.assertTrue(is_ensemble(model=ModelList(m, deterministic)))
            # Nested ModelList
            self.assertTrue(is_ensemble(model=ModelList(ModelList(m), m)))
            self.assertTrue(is_ensemble(model=ModelList(ModelList(m), deterministic)))

        non_ensemble_models = (
            GenericDeterministicModel(f=lambda x: x),
            SingleTaskGP(train_X=X, train_Y=Y),
            MultiTaskGP(train_X=X, train_Y=Y, task_feature=-1),
            HigherOrderGP(train_X=X, train_Y=Y),
            SingleTaskMultiFidelityGP(train_X=X, train_Y=Y, data_fidelities=[3]),
            MixedSingleTaskGP(train_X=X, train_Y=Y, cat_dims=[1]),
            PairwiseGP(datapoints=X, comparisons=None),
        )
        for m in non_ensemble_models:
            self.assertFalse(is_ensemble(model=m))
            # ModelList
            self.assertFalse(is_ensemble(model=ModelList(m, m)))
            self.assertFalse(is_ensemble(model=ModelList(m, vanilla_gp)))
            self.assertFalse(is_ensemble(model=ModelList(m, deterministic)))
            # Nested ModelList
            self.assertFalse(is_ensemble(model=ModelList(ModelList(m), m)))
            self.assertFalse(is_ensemble(model=ModelList(ModelList(m), deterministic)))
