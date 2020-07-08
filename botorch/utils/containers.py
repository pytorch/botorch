#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Containers to standardize inputs into models and acquisition functions.
"""

from typing import List, NamedTuple, Optional

from torch import Tensor


class TrainingData(NamedTuple):
    r"""Standardized struct of model training data."""

    Xs: List[Tensor]
    Ys: List[Tensor]
    Yvars: Optional[List[Tensor]] = None
