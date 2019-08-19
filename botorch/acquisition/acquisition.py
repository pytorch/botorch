#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

r"""
Abstract base module for all botorch acquisition functions.
"""

import warnings
from abc import ABC, abstractmethod
from typing import Optional

from torch import Tensor
from torch.nn import Module

from ..exceptions import BotorchWarning
from ..models.model import Model


class AcquisitionFunction(Module, ABC):
    r"""Abstract base class for acquisition functions."""

    def __init__(self, model: Model) -> None:
        r"""Constructor for the AcquisitionFunction base class.

        Args:
            model: A fitted model.
        """
        super().__init__()
        self.add_module("model", model)

    def set_X_pending(self, X_pending: Optional[Tensor] = None) -> None:
        r"""Informs the acquisition function about pending design points.

        Args:
            X_pending: `n x d` Tensor with `n` `d`-dim design points that have
                been submitted for evaluation but have not yet been evaluated.
        """
        if X_pending is not None:
            if X_pending.requires_grad:
                warnings.warn(
                    "Pending points require a gradient but the acquisition function"
                    " will not provide a gradient to these points.",
                    BotorchWarning,
                )
            self.X_pending = X_pending.clone().detach()
        else:
            self.X_pending = X_pending

    @abstractmethod
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the acquisition function on the candidate set X.

        Args:
            X: A `(b) x q x d`-dim Tensor of `(b)` t-batches with `q` `d`-dim
                design points each.

        Returns:
            A `(b)`-dim Tensor of acquisition function values at the given
            design points `X`.
        """
        pass  # pragma: no cover
