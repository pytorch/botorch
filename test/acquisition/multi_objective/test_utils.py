#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings

from botorch.acquisition.multi_objective.utils import get_default_partitioning_alpha
from botorch.utils.testing import BotorchTestCase


class TestUtils(BotorchTestCase):
    def test_get_default_partitioning_alpha(self):
        self.assertEqual(0.0, get_default_partitioning_alpha(num_objectives=2))
        self.assertEqual(1e-5, get_default_partitioning_alpha(num_objectives=3))
        self.assertEqual(1e-4, get_default_partitioning_alpha(num_objectives=4))
        # In `BotorchTestCase.setUp` warnings are filtered, so here we
        # remove the filter to ensure a warning is issued as expected.
        warnings.resetwarnings()
        with warnings.catch_warnings(record=True) as ws:
            self.assertEqual(0.1, get_default_partitioning_alpha(num_objectives=7))
        self.assertEqual(len(ws), 1)
