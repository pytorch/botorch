#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
DEPRECATED Probability Distributions.
"""

from __future__ import annotations

import warnings

from torch.distributions.kumaraswamy import Kumaraswamy  # noqa: 401


warnings.warn(
    "The botorch.distributions module has been deprecated in favor of "
    "torch.distributions.",
    DeprecationWarning,
)
