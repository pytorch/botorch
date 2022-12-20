#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings

import gpytorch.settings as gp_settings
import linear_operator.settings as linop_settings
from botorch import settings
from botorch.exceptions import BotorchWarning
from botorch.utils.testing import BotorchTestCase


class TestSettings(BotorchTestCase):
    def test_flags(self):
        for flag in (settings.debug, settings.propagate_grads):
            self.assertFalse(flag.on())
            self.assertTrue(flag.off())
            with flag(True):
                self.assertTrue(flag.on())
                self.assertFalse(flag.off())
            self.assertFalse(flag.on())
            self.assertTrue(flag.off())

    def test_debug(self):
        # turn on BotorchWarning
        settings.debug._set_state(True)
        # check that warnings are suppressed
        with settings.debug(False):
            with warnings.catch_warnings(record=True) as ws:
                warnings.warn("test", BotorchWarning)
            self.assertEqual(len(ws), 0)
        # check that warnings are not suppressed outside of context manager
        with warnings.catch_warnings(record=True) as ws:
            warnings.warn("test", BotorchWarning)
        self.assertEqual(len(ws), 1)

        # turn off BotorchWarnings
        settings.debug._set_state(False)
        # check that warnings are not suppressed
        with settings.debug(True):
            with warnings.catch_warnings(record=True) as ws:
                warnings.warn("test", BotorchWarning)
            self.assertEqual(len(ws), 1)
        # check that warnings are suppressed outside of context manager
        with warnings.catch_warnings(record=True) as ws:
            warnings.warn("test", BotorchWarning)
        self.assertEqual(len(ws), 0)


class TestDefaultGPyTorchLinOpSettings(BotorchTestCase):
    def test_default_gpytorch_linop_settings(self):
        self.assertTrue(linop_settings._fast_covar_root_decomposition.off())
        self.assertTrue(linop_settings._fast_log_prob.off())
        self.assertTrue(linop_settings._fast_solves.off())
        self.assertEqual(linop_settings.cholesky_max_tries.value(), 6)
        self.assertEqual(linop_settings.max_cholesky_size.value(), 4096)
        self.assertEqual(gp_settings.max_eager_kernel_size.value(), 4096)
