#! /usr/bin/env python3

import unittest

from botorch.exceptions import (
    BadInitialCandidatesError,
    BotorchError,
    CandidateGenerationError,
)


class TestBotorchExceptions(unittest.TestCase):
    def test_botorch_exception_hierarchy(self):
        self.assertIsInstance(BotorchError(), Exception)
        self.assertIsInstance(CandidateGenerationError(), BotorchError)
        self.assertIsInstance(BadInitialCandidatesError(), CandidateGenerationError)

    def test_raise_botorch_exceptions(self):
        with self.assertRaises(BotorchError):
            raise BotorchError("message")
        with self.assertRaises(CandidateGenerationError):
            raise CandidateGenerationError("message")
        with self.assertRaises(BadInitialCandidatesError):
            raise BadInitialCandidatesError("message")


if __name__ == "__main__":
    unittest.main()
