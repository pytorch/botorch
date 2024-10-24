#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
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
from itertools import chain
from pathlib import Path

import torch
from botorch.utils.testing import BotorchTestCase


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
class TestBotorchCUDA(unittest.TestCase):
    def test_cuda(self):
        test_dir = Path(__file__).parent.resolve()
        tests = unittest.TestLoader().discover(test_dir)
        self.assertTrue(run_cuda_tests(tests))


def run_cuda_tests(tests: unittest.TestCase | unittest.TestSuite) -> bool:
    """Function for running all tests on cuda (except TestBotorchCUDA itself)"""
    if isinstance(tests, BotorchTestCase):
        tests.device = torch.device("cuda")
        test_result = tests.run()
        if test_result is None:
            # some test runners may return None on skipped tests
            return True
        passed = test_result.wasSuccessful()
        if not passed:
            # print test name
            print(f"test: {tests}")
            for error in chain(test_result.errors, test_result.failures):
                # print traceback
                print(f"error: {error[1]}")
        return passed
    elif isinstance(tests, unittest.TestSuite):
        return all(run_cuda_tests(tests_) for tests_ in tests)
    elif (
        isinstance(tests, unittest.TestCase)
        and tests.id() == "test_cuda.TestBotorchCUDA.test_cuda"
    ):
        # ignore TestBotorchCUDA
        return True
    elif isinstance(tests, unittest.loader._FailedTest):
        # test failed to load, often import error
        print(f"test: {tests}")
        print(f"exception: {tests._exception}")
        return False
    else:
        raise ValueError(f"Unexpected type for test: {tests}")
