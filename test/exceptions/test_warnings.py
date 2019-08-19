#! /usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest
import warnings

from botorch.exceptions.warnings import (
    BadInitialCandidatesWarning,
    BotorchTensorDimensionWarning,
    BotorchWarning,
    InputDataWarning,
    OptimizationWarning,
    SamplingWarning,
)


class TestBotorchWarnings(unittest.TestCase):
    def test_botorch_warnings_hierarchy(self):
        self.assertIsInstance(BotorchWarning(), Warning)
        self.assertIsInstance(BadInitialCandidatesWarning(), BotorchWarning)
        self.assertIsInstance(InputDataWarning(), BotorchWarning)
        self.assertIsInstance(OptimizationWarning(), BotorchWarning)
        self.assertIsInstance(SamplingWarning(), BotorchWarning)
        self.assertIsInstance(BotorchTensorDimensionWarning(), BotorchWarning)

    def test_botorch_warnings(self):
        for WarningClass in (
            BotorchTensorDimensionWarning,
            BotorchWarning,
            BadInitialCandidatesWarning,
            InputDataWarning,
            OptimizationWarning,
            SamplingWarning,
        ):
            with warnings.catch_warnings(record=True) as ws:
                warnings.warn("message", WarningClass)
                self.assertEqual(len(ws), 1)
                self.assertTrue(issubclass(ws[-1].category, WarningClass))
                self.assertTrue("message" in str(ws[-1].message))
