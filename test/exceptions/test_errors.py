#! /usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

from botorch.exceptions.errors import (
    BotorchError,
    CandidateGenerationError,
    UnsupportedError,
)


class TestBotorchExceptions(unittest.TestCase):
    def test_botorch_exception_hierarchy(self):
        self.assertIsInstance(BotorchError(), Exception)
        self.assertIsInstance(CandidateGenerationError(), BotorchError)
        self.assertIsInstance(UnsupportedError(), BotorchError)

    def test_raise_botorch_exceptions(self):
        with self.assertRaises(BotorchError):
            raise BotorchError("message")
        with self.assertRaises(CandidateGenerationError):
            raise CandidateGenerationError("message")
        with self.assertRaises(UnsupportedError):
            raise UnsupportedError("message")
