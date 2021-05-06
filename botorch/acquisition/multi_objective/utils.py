#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Utilities for multi-objective acquisition functions.
"""

import warnings

from botorch.exceptions.warnings import BotorchWarning


def get_default_partitioning_alpha(num_objectives: int) -> float:
    """Adaptively selects a reasonable partitioning based on the number of objectives.

    This strategy is derived from the results in [Daulton2020qehvi]_, which suggest
    that this heuristic provides a reasonable trade-off between the closed-loop
    performance and the wall time required for the partitioning.
    """
    if num_objectives == 2:
        return 0.0
    elif num_objectives > 6:
        warnings.warn("EHVI works best for less than 7 objectives.", BotorchWarning)
    return 10 ** (-8 + num_objectives)
