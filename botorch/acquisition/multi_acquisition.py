#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Abstract base module for all botorch acquisition functions."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from ax.exceptions.core import UnsupportedError
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.objective import PosteriorTransform
from botorch.models.model import Model
from botorch.utils.transforms import (
    average_over_ensemble_models,
    t_batch_mode_transform,
)
from torch import Tensor


class MultiOutputAcquisitionFunction(AcquisitionFunction, ABC):
    r"""Abstract base class for multi-output acquisition functions.

    These are intended to be optimized with a multi-objective optimizer (e.g.
    NSGA-II).
    """

    @abstractmethod
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the acquisition function on the candidate set X.

        Args:
            X: A `(b) x q x d`-dim Tensor of `(b)` t-batches with `q` `d`-dim
                design points each.

        Returns:
            A `(b) x m`-dim Tensor of acquisition function values at the given
            design points `X`.
        """

    def set_X_pending(self, X_pending: Tensor | None) -> None:
        r"""Set the pending points.

        Args:
            X_pending: A `batch_shape x m x d`-dim Tensor of `m` design points that
                have points that have been submitted for function evaluation
                (but may not yet have been evaluated).
        """
        raise UnsupportedError(
            "X_pending is not supported for multi-output acquisition functions"
        )


class MultiPosteriorMean(MultiOutputAcquisitionFunction):
    def __init__(
        self, model: Model, posterior_transform: PosteriorTransform | None = None
    ) -> None:
        r"""Constructor for the MultiPosteriorMean.

        Args:
            acqfs: A list of `m` acquisition functions.
        """
        super().__init__(model=model)
        if self.model.num_outputs < 2:
            raise NotImplementedError(
                "MultiPosteriorMean only supports multi-output models."
            )
        self.posterior_transform = posterior_transform

    @t_batch_mode_transform(expected_q=1)
    @average_over_ensemble_models
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the acquisition function on the candidate set X.

        Args:
            X: A `(b) x q x d`-dim Tensor of `(b)` t-batches with `q` `d`-dim
                design points each.

        Returns:
            A `(b) x m`-dim Tensor of acquisition function values at the given
            design points `X`.
        """
        return self.model.posterior(
            X, posterior_transform=self.posterior_transform
        ).mean


class MultiOutputAcquisitionFunctionWrapper(MultiOutputAcquisitionFunction):
    r"""Multi-output wrapper around single-output acquisition functions."""

    def __init__(self, acqfs: list[AcquisitionFunction]) -> None:
        r"""Constructor for the AcquisitionFunction base class.

        Args:
            acqfs: A list of `m` acquisition functions.
        """
        # We could set the model to be an ensemble model consistent of the
        # model used in each acqf
        super().__init__(model=acqfs[0].model)
        self.acqfs: list[AcquisitionFunction] = acqfs

    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the acquisition function on the candidate set X.

        Args:
            X: A `(b) x q x d`-dim Tensor of `(b)` t-batches with `q` `d`-dim
                design points each.

        Returns:
            A `(b) x m`-dim Tensor of acquisition function values at the given
            design points `X`.
        """
        return torch.stack([acqf(X) for acqf in self.acqfs], dim=-1)
