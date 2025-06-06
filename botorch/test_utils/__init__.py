#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
test_utils has its own directory with 'botorch/' to avoid circular dependencies:
Anything in 'tests/' can depend on anything in 'botorch/test_utils/', and
anything in 'botorch/test_utils/' can depend on anything in the rest of
'botorch/'.
"""

from botorch.test_utils.mock import mock_optimize

__all__ = ["mock_optimize"]
