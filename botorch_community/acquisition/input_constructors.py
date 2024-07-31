#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
A registry of helpers for generating inputs to acquisition function
constructors programmatically from a consistent input format.

Contributor: hvarfner (bayesian_active_learning, scorebo)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from botorch.acquisition.input_constructors import acqf_input_constructor
from botorch.acquisition.utils import get_optimal_samples
from botorch.models.model import Model
from botorch_community.acquisition.bayesian_active_learning import (
    qBayesianQueryByComittee,
    qBayesianVarianceReduction,
    qStatisticalDistanceActiveLearning,
)
from botorch_community.acquisition.scorebo import qSelfCorrectingBayesianOptimization
from torch import Tensor


@acqf_input_constructor(
    qBayesianQueryByComittee,
    qBayesianVarianceReduction,
)
def construct_inputs_BAL(
    model: Model,
    X_pending: Optional[Tensor] = None,
):
    inputs = {
        "model": model,
        "X_pending": X_pending,
    }
    return inputs


@acqf_input_constructor(qStatisticalDistanceActiveLearning)
def construct_inputs_SAL(
    model: Model,
    distance_metric: str = "hellinger",
    X_pending: Optional[Tensor] = None,
):
    inputs = {
        "model": model,
        "distance_metric": distance_metric,
        "X_pending": X_pending,
    }
    return inputs


@acqf_input_constructor(qSelfCorrectingBayesianOptimization)
def construct_inputs_SCoreBO(
    model: Model,
    bounds: List[Tuple[float, float]],
    num_optima: int = 8,
    maximize: bool = True,
    distance_metric: str = "hellinger",
    X_pending: Optional[Tensor] = None,
):
    dtype = model.train_targets.dtype
    # the number of optima are per model
    optimal_inputs, optimal_outputs = get_optimal_samples(
        model=model,
        bounds=torch.as_tensor(bounds, dtype=dtype).T,
        num_optima=num_optima,
        maximize=maximize,
    )

    inputs = {
        "model": model,
        "optimal_inputs": optimal_inputs,
        "optimal_outputs": optimal_outputs,
        "distance_metric": distance_metric,
        "maximize": maximize,
        "X_pending": X_pending,
    }
    return inputs
