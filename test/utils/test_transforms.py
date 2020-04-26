#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from botorch import settings
from botorch.utils.testing import BotorchTestCase
from botorch.utils.transforms import (
    concatenate_pending_points,
    gpt_posterior_settings,
    match_batch_shape,
    normalize,
    normalize_indices,
    squeeze_last_dim,
    standardize,
    t_batch_mode_transform,
    unnormalize,
)
from gpytorch import settings as gpt_settings
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
    def test_normalize_unnormalize(self):
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


class BMIMTestClass:
    @t_batch_mode_transform()
    def q_method(self, X: Tensor) -> None:
        return X

    @t_batch_mode_transform(expected_q=1)
    def q1_method(self, X: Tensor) -> None:
        return X

    @concatenate_pending_points
    def dummy_method(self, X: Tensor) -> Tensor:
        return X


class TestBatchModeTransform(BotorchTestCase):
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


class TestSqueezeLastDim(BotorchTestCase):
    def test_squeeze_last_dim(self):
        Y = torch.rand(2, 1, 1)
        Y_squeezed = squeeze_last_dim(Y=Y)
        self.assertTrue(torch.equal(Y_squeezed, Y.squeeze(-1)))


class TestGPTPosteriorSettings(BotorchTestCase):
    def test_gpt_posterior_settings(self):
        for propagate_grads in (False, True):
            with settings.propagate_grads(propagate_grads):
                with gpt_posterior_settings():
                    self.assertTrue(gpt_settings.debug.off())
                    self.assertTrue(gpt_settings.fast_pred_var.on())
                    if settings.propagate_grads.off():
                        self.assertTrue(gpt_settings.detach_test_caches.on())
                    else:
                        self.assertTrue(gpt_settings.detach_test_caches.off())
