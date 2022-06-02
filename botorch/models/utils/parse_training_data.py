#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Parsing rules for BoTorch datasets."""


from __future__ import annotations

from typing import Any, Dict, Hashable, Type, Union

import torch
from botorch.exceptions import UnsupportedError
from botorch.models.model import Model
from botorch.models.multitask import FixedNoiseMultiTaskGP, MultiTaskGP
from botorch.models.pairwise_gp import PairwiseGP
from botorch.utils.datasets import (
    BotorchDataset,
    FixedNoiseDataset,
    RankingDataset,
    SupervisedDataset,
)
from botorch.utils.dispatcher import Dispatcher
from torch import cat, Tensor
from torch.nn.functional import pad


def _encoder(arg: Any) -> Type:
    # Allow type variables to be passed as arguments at runtime
    return arg if isinstance(arg, type) else type(arg)


dispatcher = Dispatcher("parse_training_data", encoder=_encoder)


def parse_training_data(
    consumer: Any,
    training_data: Union[BotorchDataset, Dict[Hashable, BotorchDataset]],
    **kwargs: Any,
) -> Dict[Hashable, Tensor]:
    r"""Prepares a (collection of) datasets for consumption by a given object.

    Args:
        training_datas: A BoTorchDataset or dictionary thereof.
        consumer: The object that will consume the parsed data, or type thereof.

    Returns:
        A dictionary containing the extracted information.
    """
    return dispatcher(consumer, training_data, **kwargs)


@dispatcher.register(Model, SupervisedDataset)
def _parse_model_supervised(
    consumer: Model, dataset: SupervisedDataset, **ignore: Any
) -> Dict[Hashable, Tensor]:
    return {"train_X": dataset.X(), "train_Y": dataset.Y()}


@dispatcher.register(Model, FixedNoiseDataset)
def _parse_model_fixedNoise(
    consumer: Model, dataset: FixedNoiseDataset, **ignore: Any
) -> Dict[Hashable, Tensor]:
    return {
        "train_X": dataset.X(),
        "train_Y": dataset.Y(),
        "train_Yvar": dataset.Yvar(),
    }


@dispatcher.register(PairwiseGP, RankingDataset)
def _parse_pairwiseGP_ranking(
    consumer: PairwiseGP, dataset: RankingDataset, **ignore: Any
) -> Dict[Hashable, Tensor]:
    datapoints = dataset.X.values
    comparisons = dataset.X.indices
    comp_order = dataset.Y()
    comparisons = torch.gather(input=comparisons, dim=-1, index=comp_order)

    return {
        "datapoints": datapoints,
        "comparisons": comparisons,
    }


@dispatcher.register(Model, dict)
def _parse_model_dict(
    consumer: Model,
    training_data: Dict[Hashable, BotorchDataset],
    **kwargs: Any,
) -> Dict[Hashable, Tensor]:
    if len(training_data) != 1:
        raise UnsupportedError(
            "Default training data parsing logic does not support "
            "passing multiple datasets to single task models."
        )
    return dispatcher(consumer, next(iter(training_data.values())))


@dispatcher.register((MultiTaskGP, FixedNoiseMultiTaskGP), dict)
def _parse_multitask_dict(
    consumer: Model,
    training_data: Dict[Hashable, BotorchDataset],
    *,
    task_feature: int = 0,
    task_feature_container: Hashable = "train_X",
    **kwargs: Any,
) -> Dict[Hashable, Tensor]:
    cache = {}
    for task_id, dataset in enumerate(training_data.values()):
        parse = parse_training_data(consumer, dataset, **kwargs)
        if task_feature_container not in parse:
            raise ValueError(f"Missing required term `{task_feature_container}`.")

        if cache and cache.keys() != parse.keys():
            raise UnsupportedError(
                "Cannot combine datasets with heterogeneous parsed formats."
            )

        # Add task indicator features to specified container
        X = parse[task_feature_container]
        d = X.shape[-1]
        i = d + task_feature + 1 if task_feature < 0 else task_feature
        if i < 0 or d < i:
            raise ValueError("Invalid `task_feature`: out-of-bounds.")

        if i == 0:
            X = pad(X, (1, 0), value=task_id)
        elif i == d:
            X = pad(X, (0, 1), value=task_id)
        else:
            A, B = X.split([i, d - i], dim=-1)
            X = cat([pad(A, (0, 1), value=task_id), B], dim=-1)
        parse[task_feature_container] = X

        if cache:
            for key, val in parse.items():
                cache[key].append(val)
        else:
            cache = {key: [val] for key, val in parse.items()}

    return {key: cat(tensors, dim=0) for key, tensors in cache.items()}
