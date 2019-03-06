#! /usr/bin/env python3

import unittest

from botorch.exceptions.errors import BotorchError, CandidateGenerationError


class TestBotorchExceptions(unittest.TestCase):
    def test_botorch_exception_hierarchy(self):
        self.assertIsInstance(BotorchError(), Exception)
        self.assertIsInstance(CandidateGenerationError(), BotorchError)

    def test_raise_botorch_exceptions(self):
        with self.assertRaises(BotorchError):
            raise BotorchError("message")
        with self.assertRaises(CandidateGenerationError):
            raise CandidateGenerationError("message")
