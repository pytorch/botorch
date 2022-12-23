#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from botorch.utils.profiling import get_memory_usage_preserving_output
from botorch.utils.testing import BotorchTestCase


class TestProfiling(BotorchTestCase):
    def test_get_memory_usage_preserving_output(self) -> None:
        def power(base: int, pow: int = 0) -> int:
            return base**pow

        power_res, memory = get_memory_usage_preserving_output(power, 2, pow=3)
        self.assertEqual(power_res, power(2, 3))
        self.assertIsInstance(memory, list)
        self.assertIsInstance(memory[0], float)
        self.assertTrue(all(mem > 0 for mem in memory))
