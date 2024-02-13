#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
A registry of helpers for generating inputs to acquisition function
constructors programmatically from a consistent input format.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.input_constructors import allow_only_specific_variable_kwargs
from botorch.acquisition.utils import get_optimal_samples
from botorch.models.model import Model
from botorch_community.acquisition.bayesian_active_learning import (
    qBayesianActiveLearningByDisagreement,
    qBayesianQueryByComittee,
    qBayesianVarianceReduction,
    qStatisticalDistanceActiveLearning,
)
from botorch_community.acquisition.scorebo import qSelfCorrectingBayesianOptimization
from torch import Tensor


COMMUNITY_ACQF_INPUT_CONSTRUCTOR_REGISTRY = {}


def get_community_acqf_input_constructor(
    acqf_cls: Type[AcquisitionFunction],
) -> Callable[..., Dict[str, Any]]:
    r"""Get acquisition function input constructor from registry.

    Args:
        acqf_cls: The AcquisitionFunction class (not instance) for which
            to retrieve the input constructor.

    Returns:
        The input constructor associated with `acqf_cls`.

    """
    if acqf_cls not in COMMUNITY_ACQF_INPUT_CONSTRUCTOR_REGISTRY:
        raise RuntimeError(
            f"Input constructor for acquisition class `{acqf_cls.__name__}` not "
            "registered. Use the `@acqf_input_constructor` decorator to register "
            "a new method."
        )
    return COMMUNITY_ACQF_INPUT_CONSTRUCTOR_REGISTRY[acqf_cls]


def community_acqf_input_constructor(
    *acqf_cls: Type[AcquisitionFunction],
) -> Callable[..., AcquisitionFunction]:
    r"""Decorator for registering acquisition function input constructors.

    Args:
        acqf_cls: The AcquisitionFunction classes (not instances) for which
            to register the input constructor.
    """
    for acqf_cls_ in acqf_cls:
        if acqf_cls_ in COMMUNITY_ACQF_INPUT_CONSTRUCTOR_REGISTRY:
            raise ValueError(
                "Cannot register duplicate arg constructor for acquisition "
                f"class `{acqf_cls_.__name__}`"
            )

    def decorator(method):
        method_kwargs = allow_only_specific_variable_kwargs(method)
        for acqf_cls_ in acqf_cls:
            COMMUNITY_ACQF_INPUT_CONSTRUCTOR_REGISTRY[acqf_cls_] = method_kwargs
        return method

    return decorator


@community_acqf_input_constructor(
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


@community_acqf_input_constructor(qBayesianActiveLearningByDisagreement)
def construct_inputs_BALD(
    model: Model,
    X_pending: Optional[Tensor] = None,
):
    inputs = {
        "model": model,
        "X_pending": X_pending,
    }
    return inputs


@community_acqf_input_constructor(qStatisticalDistanceActiveLearning)
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


@community_acqf_input_constructor(qSelfCorrectingBayesianOptimization)
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
