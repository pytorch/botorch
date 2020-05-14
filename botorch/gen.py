#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
DEPRECATED - Candidate generation utilities.
Use the botorch.generation.gen module instead.
"""

import warnings

from botorch.generation.gen import (  # noqa F401
    gen_candidates_scipy,
    gen_candidates_torch,
    get_best_candidates,
)


warnings.warn(
    "The botorch.gen module has been renamed to botorch.generation.gen",
    DeprecationWarning,
)
