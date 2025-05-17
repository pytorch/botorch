#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from botorch.models.model import Model
from botorch_community.models.vbll_helper import DenseNormal, Normal
from torch import Tensor


class AbstractBLLModel(Model, ABC):
    def __init__(self):
        """Abstract class for Bayesian Last Layer (BLL) models."""
        super().__init__()
        self.model = None

    @property
    def num_outputs(self) -> int:
        return self.model.num_outputs

    @property
    def num_inputs(self):
        return self.model.num_inputs

    @property
    def device(self):
        return self.model.device

    @abstractmethod
    def __call__(self, X: Tensor) -> Normal | DenseNormal:
        pass  # pragma: no cover

    @abstractmethod
    def fit(self, *args, **kwargs):
        pass  # pragma: no cover

    @abstractmethod
    def sample(self, sample_shape: torch.Size | None = None) -> nn.Module:
        pass  # pragma: no cover
