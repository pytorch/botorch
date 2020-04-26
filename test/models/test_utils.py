#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings

import torch
from botorch import settings
from botorch.exceptions import InputDataError, InputDataWarning
from botorch.models.utils import (
    add_output_dim,
    check_min_max_scaling,
    check_no_nans,
    check_standardization,
    multioutput_to_batch_mode_transform,
    validate_input_scaling,
)
from botorch.utils.testing import BotorchTestCase


class TestMultiOutputToBatchModeTransform(BotorchTestCase):
    def test_multioutput_to_batch_mode_transform(self):
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            n = 3
            num_outputs = 2
            train_X = torch.rand(n, 1, **tkwargs)
            train_Y = torch.rand(n, num_outputs, **tkwargs)
            train_Yvar = torch.rand(n, num_outputs, **tkwargs)
            X_out, Y_out, Yvar_out = multioutput_to_batch_mode_transform(
                train_X=train_X,
                train_Y=train_Y,
                num_outputs=num_outputs,
                train_Yvar=train_Yvar,
            )
            expected_X_out = train_X.unsqueeze(0).expand(num_outputs, -1, 1)
            self.assertTrue(torch.equal(X_out, expected_X_out))
            self.assertTrue(torch.equal(Y_out, train_Y.transpose(0, 1)))
            self.assertTrue(torch.equal(Yvar_out, train_Yvar.transpose(0, 1)))


class TestAddOutputDim(BotorchTestCase):
    def test_add_output_dim(self):
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            original_batch_shape = torch.Size([2])
            # check exception is raised when trailing batch dims do not line up
            X = torch.rand(2, 3, 2, 1, **tkwargs)
            with self.assertRaises(RuntimeError):
                add_output_dim(X=X, original_batch_shape=original_batch_shape)
            # test no new batch dims
            X = torch.rand(2, 2, 1, **tkwargs)
            X_out, output_dim_idx = add_output_dim(
                X=X, original_batch_shape=original_batch_shape
            )
            self.assertTrue(torch.equal(X_out, X.unsqueeze(1)))
            self.assertEqual(output_dim_idx, 1)
            # test new batch dims
            X = torch.rand(3, 2, 2, 1, **tkwargs)
            X_out, output_dim_idx = add_output_dim(
                X=X, original_batch_shape=original_batch_shape
            )
            self.assertTrue(torch.equal(X_out, X.unsqueeze(2)))
            self.assertEqual(output_dim_idx, 2)


class TestInputDataChecks(BotorchTestCase):
    def test_check_no_nans(self):
        check_no_nans(torch.tensor([1.0, 2.0]))
        with self.assertRaises(InputDataError):
            check_no_nans(torch.tensor([1.0, float("nan")]))

    def test_check_min_max_scaling(self):
        with settings.debug(True):
            # check unscaled input in unit cube
            X = 0.1 + 0.8 * torch.rand(4, 2, 3)
            with warnings.catch_warnings(record=True) as ws:
                check_min_max_scaling(X=X)
                self.assertFalse(
                    any(issubclass(w.category, InputDataWarning) for w in ws)
                )
            check_min_max_scaling(X=X, raise_on_fail=True)
            with warnings.catch_warnings(record=True) as ws:
                check_min_max_scaling(X=X, strict=True)
                self.assertTrue(
                    any(issubclass(w.category, InputDataWarning) for w in ws)
                )
                self.assertTrue(any("not scaled" in str(w.message) for w in ws))
            with self.assertRaises(InputDataError):
                check_min_max_scaling(X=X, strict=True, raise_on_fail=True)
            # check proper input
            Xmin, Xmax = X.min(dim=-1, keepdim=True)[0], X.max(dim=-1, keepdim=True)[0]
            Xstd = (X - Xmin) / (Xmax - Xmin)
            with warnings.catch_warnings(record=True) as ws:
                check_min_max_scaling(X=Xstd)
                self.assertFalse(
                    any(issubclass(w.category, InputDataWarning) for w in ws)
                )
            check_min_max_scaling(X=Xstd, raise_on_fail=True)
            with warnings.catch_warnings(record=True) as ws:
                check_min_max_scaling(X=Xstd, strict=True)
                self.assertFalse(
                    any(issubclass(w.category, InputDataWarning) for w in ws)
                )
            check_min_max_scaling(X=Xstd, strict=True, raise_on_fail=True)
            # check violation
            X[0, 0, 0] = 2
            with warnings.catch_warnings(record=True) as ws:
                check_min_max_scaling(X=X)
                self.assertTrue(
                    any(issubclass(w.category, InputDataWarning) for w in ws)
                )
                self.assertTrue(any("not contained" in str(w.message) for w in ws))
            with self.assertRaises(InputDataError):
                check_min_max_scaling(X=X, raise_on_fail=True)
            with warnings.catch_warnings(record=True) as ws:
                check_min_max_scaling(X=X, strict=True)
                self.assertTrue(
                    any(issubclass(w.category, InputDataWarning) for w in ws)
                )
                self.assertTrue(any("not contained" in str(w.message) for w in ws))
            with self.assertRaises(InputDataError):
                check_min_max_scaling(X=X, strict=True, raise_on_fail=True)

    def test_check_standardization(self):
        Y = torch.randn(3, 4, 2)
        # check standardized input
        Yst = (Y - Y.mean(dim=-2, keepdim=True)) / Y.std(dim=-2, keepdim=True)
        with settings.debug(True):
            with warnings.catch_warnings(record=True) as ws:
                check_standardization(Y=Yst)
                self.assertFalse(
                    any(issubclass(w.category, InputDataWarning) for w in ws)
                )
            check_standardization(Y=Yst, raise_on_fail=True)
            # check nonzero mean
            with warnings.catch_warnings(record=True) as ws:
                check_standardization(Y=Yst + 1)
                self.assertTrue(
                    any(issubclass(w.category, InputDataWarning) for w in ws)
                )
                self.assertTrue(any("not standardized" in str(w.message) for w in ws))
            with self.assertRaises(InputDataError):
                check_standardization(Y=Yst + 1, raise_on_fail=True)
            # check non-unit variance
            with warnings.catch_warnings(record=True) as ws:
                check_standardization(Y=Yst * 2)
                self.assertTrue(
                    any(issubclass(w.category, InputDataWarning) for w in ws)
                )
                self.assertTrue(any("not standardized" in str(w.message) for w in ws))
            with self.assertRaises(InputDataError):
                check_standardization(Y=Yst * 2, raise_on_fail=True)

    def test_validate_input_scaling(self):
        train_X = 2 + torch.rand(3, 4, 3)
        train_Y = torch.randn(3, 4, 2)
        # check that nothing is being checked
        with settings.validate_input_scaling(False), settings.debug(True):
            with warnings.catch_warnings(record=True) as ws:
                validate_input_scaling(train_X=train_X, train_Y=train_Y)
                self.assertFalse(
                    any(issubclass(w.category, InputDataWarning) for w in ws)
                )
        # check that warnings are being issued
        with settings.debug(True), warnings.catch_warnings(record=True) as ws:
            validate_input_scaling(train_X=train_X, train_Y=train_Y)
            self.assertTrue(any(issubclass(w.category, InputDataWarning) for w in ws))
        # check that errors are raised when requested
        with settings.debug(True):
            with self.assertRaises(InputDataError):
                validate_input_scaling(
                    train_X=train_X, train_Y=train_Y, raise_on_fail=True
                )
        # check that no errors are being raised if everything is standardized
        train_X_min = train_X.min(dim=-1, keepdim=True)[0]
        train_X_max = train_X.max(dim=-1, keepdim=True)[0]
        train_X_std = (train_X - train_X_min) / (train_X_max - train_X_min)
        train_Y_std = (train_Y - train_Y.mean(dim=-2, keepdim=True)) / train_Y.std(
            dim=-2, keepdim=True
        )
        with settings.debug(True), warnings.catch_warnings(record=True) as ws:
            validate_input_scaling(train_X=train_X_std, train_Y=train_Y_std)
            self.assertFalse(any(issubclass(w.category, InputDataWarning) for w in ws))
        # test that negative variances raise an error
        train_Yvar = torch.rand_like(train_Y_std)
        train_Yvar[0, 0, 1] = -0.5
        with settings.debug(True):
            with self.assertRaises(InputDataError):
                validate_input_scaling(
                    train_X=train_X_std, train_Y=train_Y_std, train_Yvar=train_Yvar
                )
        # check that NaNs raise errors
        train_X_std[0, 0, 0] = float("nan")
        with settings.debug(True):
            with self.assertRaises(InputDataError):
                validate_input_scaling(train_X=train_X_std, train_Y=train_Y_std)
