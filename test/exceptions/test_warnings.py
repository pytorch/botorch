#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings

from botorch.exceptions.warnings import (
    BadInitialCandidatesWarning,
    BotorchTensorDimensionWarning,
    BotorchWarning,
    CostAwareWarning,
    InputDataWarning,
    OptimizationWarning,
    SamplingWarning,
    UserInputWarning,
)
from botorch.utils.testing import BotorchTestCase


class TestBotorchWarnings(BotorchTestCase):
    def test_botorch_warnings_hierarchy(self):
        self.assertIsInstance(BotorchWarning(), Warning)
        self.assertIsInstance(BadInitialCandidatesWarning(), BotorchWarning)
        self.assertIsInstance(CostAwareWarning(), BotorchWarning)
        self.assertIsInstance(InputDataWarning(), BotorchWarning)
        self.assertIsInstance(OptimizationWarning(), BotorchWarning)
        self.assertIsInstance(SamplingWarning(), BotorchWarning)
        self.assertIsInstance(BotorchTensorDimensionWarning(), BotorchWarning)
        self.assertIsInstance(UserInputWarning(), BotorchWarning)

    def test_botorch_warnings(self):
        for WarningClass in (
            BotorchTensorDimensionWarning,
            BotorchWarning,
            BadInitialCandidatesWarning,
            CostAwareWarning,
            InputDataWarning,
            OptimizationWarning,
            SamplingWarning,
            UserInputWarning,
        ):
            with warnings.catch_warnings(record=True) as ws:
                warnings.warn("message", WarningClass, stacklevel=1)
            self.assertEqual(len(ws), 1)
            self.assertTrue(issubclass(ws[-1].category, WarningClass))
            self.assertTrue("message" in str(ws[-1].message))
