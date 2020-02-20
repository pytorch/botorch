#! /usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Monolithic CUDA tests. This implements a single monolithic test for all
CUDA functionality. The main reason for doing this is that if individual tests
are run in separate processes, the overhead of initializing the GPU can vastly
outweight the speedup from parallelization, and, in addition, this can lead
to the GPU running out of memory.
"""

import unittest
from pathlib import Path
from typing import Union

import torch
from botorch.utils.testing import BotorchTestCase


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
class TestBotorchCUDA(unittest.TestCase):
    def test_cuda(self):
        test_dir = Path(__file__).parent.resolve()
        tests = unittest.TestLoader().discover(test_dir)
        run_cuda_tests(tests)


def run_cuda_tests(tests: Union[unittest.TestCase, unittest.TestSuite]) -> None:
    """Function for running all tests on cuda (except TestBotorchCUDA itself)"""
    if isinstance(tests, BotorchTestCase):
        tests.device = torch.device("cuda")
        tests.run()
    elif isinstance(tests, unittest.TestSuite):
        for tests_ in tests:
            run_cuda_tests(tests_)
