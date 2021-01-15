#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import warnings

from botorch import settings
from botorch.utils.testing import BotorchTestCase


class TestBoxDecompositionDeprecation(BotorchTestCase):
    def test_deprecation(self):
        with warnings.catch_warnings(record=True) as ws, settings.debug(True):
            from botorch.utils.multi_objective.box_decomposition import (  # noqa: F401
                NondominatedPartitioning,
            )

            self.assertTrue(any(issubclass(w.category, DeprecationWarning) for w in ws))
            self.assertTrue(
                any(
                    "The botorch.utils.multi_objective.box_decomposition module has "
                    in str(w.message)
                    for w in ws
                )
            )
