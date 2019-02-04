#! /usr/bin/env python3

import unittest
import warnings

from botorch.exceptions import (
    BadInitialCandidatesWarning,
    BotorchError,
    CandidateGenerationError,
)


class TestBotorchExceptions(unittest.TestCase):
    def test_botorch_exception_hierarchy(self):
        self.assertIsInstance(BotorchError(), Exception)
        self.assertIsInstance(CandidateGenerationError(), BotorchError)

    def test_raise_botorch_exceptions(self):
        with self.assertRaises(BotorchError):
            raise BotorchError("message")
        with self.assertRaises(CandidateGenerationError):
            raise CandidateGenerationError("message")


class TestBotorchWarnings(unittest.TestCase):
    def test_botorch_warnings(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.warn("message", BadInitialCandidatesWarning)
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[-1].category, BadInitialCandidatesWarning))
            self.assertTrue("message" in str(w[-1].message))


if __name__ == "__main__":
    unittest.main()
