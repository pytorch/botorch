#! /usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest
import warnings

from botorch.exceptions.warnings import (
    BadInitialCandidatesWarning,
    BotorchWarning,
    SamplingWarning,
)


class TestBotorchWarnings(unittest.TestCase):
    def test_botorch_warnings_hierarchy(self):
        self.assertIsInstance(BotorchWarning(), Warning)
        self.assertIsInstance(BadInitialCandidatesWarning(), BotorchWarning)
        self.assertIsInstance(SamplingWarning(), BotorchWarning)

    def test_botorch_warnings(self):
        for WarningClass in (
            BotorchWarning,
            BadInitialCandidatesWarning,
            SamplingWarning,
        ):
            with warnings.catch_warnings(record=True) as w:
                warnings.warn("message", WarningClass)
                self.assertEqual(len(w), 1)
                self.assertTrue(issubclass(w[-1].category, WarningClass))
                self.assertTrue("message" in str(w[-1].message))
