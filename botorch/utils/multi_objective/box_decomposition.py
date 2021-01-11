#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
DEPRECATED - Box decomposition algorithms.
Use the botorch.utils.multi_objective.box_decompositions instead.
"""

import warnings

from botorch.utils.multi_objective.box_decompositions.non_dominated import (  # noqa F401
    NondominatedPartitioning,
)


warnings.warn(
    "The botorch.utils.multi_objective.box_decomposition module has "
    "been renamed to botorch.utils.multi_objective.box_decompositions. "
    "botorch.utils.multi_objective.box_decomposition will be removed in "
    "the next release.",
    DeprecationWarning,
)
