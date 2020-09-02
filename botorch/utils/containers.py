#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Containers to standardize inputs into models and acquisition functions.
"""

from typing import NamedTuple, Optional

from torch import Tensor


class TrainingData(NamedTuple):
    r"""Standardized struct of model training data for a single outcome."""

    X: Tensor
    Y: Tensor
    Yvar: Optional[Tensor] = None
