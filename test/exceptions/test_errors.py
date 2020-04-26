#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from botorch.exceptions.errors import (
    BotorchError,
    BotorchTensorDimensionError,
    CandidateGenerationError,
    InputDataError,
    UnsupportedError,
)
from botorch.utils.testing import BotorchTestCase


class TestBotorchExceptions(BotorchTestCase):
    def test_botorch_exception_hierarchy(self):
        self.assertIsInstance(BotorchError(), Exception)
        self.assertIsInstance(CandidateGenerationError(), BotorchError)
        self.assertIsInstance(InputDataError(), BotorchError)
        self.assertIsInstance(UnsupportedError(), BotorchError)
        self.assertIsInstance(BotorchTensorDimensionError(), BotorchError)

    def test_raise_botorch_exceptions(self):
        for ErrorClass in (
            BotorchError,
            BotorchTensorDimensionError,
            CandidateGenerationError,
            InputDataError,
            UnsupportedError,
        ):
            with self.assertRaises(ErrorClass):
                raise ErrorClass("message")
