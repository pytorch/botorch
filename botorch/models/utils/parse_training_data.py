#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Parsing rules for BoTorch datasets."""


from __future__ import annotations

from typing import Any, Dict, Type

import torch
from botorch.models.model import Model
from botorch.models.pairwise_gp import PairwiseGP
from botorch.utils.datasets import RankingDataset, SupervisedDataset
from botorch.utils.dispatcher import Dispatcher
from torch import Tensor


def _encoder(arg: Any) -> Type:
    # Allow type variables to be passed as arguments at runtime
    return arg if isinstance(arg, type) else type(arg)


dispatcher = Dispatcher("parse_training_data", encoder=_encoder)


def parse_training_data(
    consumer: Any,
    training_data: SupervisedDataset,
    **kwargs: Any,
) -> Dict[str, Tensor]:
    r"""Prepares a dataset for consumption by a given object.

    Args:
        training_datas: A SupervisedDataset.
        consumer: The object that will consume the parsed data, or type thereof.

    Returns:
        A dictionary containing the extracted information.
    """
    return dispatcher(consumer, training_data, **kwargs)


@dispatcher.register(Model, SupervisedDataset)
def _parse_model_supervised(
    consumer: Model, dataset: SupervisedDataset, **ignore: Any
) -> Dict[str, Tensor]:
    parsed_data = {"train_X": dataset.X, "train_Y": dataset.Y}
    if dataset.Yvar is not None:
        parsed_data["train_Yvar"] = dataset.Yvar
    return parsed_data


@dispatcher.register(PairwiseGP, RankingDataset)
def _parse_pairwiseGP_ranking(
    consumer: PairwiseGP, dataset: RankingDataset, **ignore: Any
) -> Dict[str, Tensor]:
    # TODO: [T163045056] Not sure what the point of the special container is if we have
    # to further process it here. We should move this logic into RankingDataset.
    datapoints = dataset._X.values
    comparisons = dataset._X.indices
    comp_order = dataset.Y
    comparisons = torch.gather(input=comparisons, dim=-1, index=comp_order)

    return {
        "datapoints": datapoints,
        "comparisons": comparisons,
    }
