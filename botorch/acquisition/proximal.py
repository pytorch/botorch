#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
A wrapper around AcquisitionFunctions to add proximal weighting of the
acquisition function.
"""

from __future__ import annotations

import torch
from botorch.acquisition import AcquisitionFunction
from botorch.exceptions.errors import UnsupportedError
from botorch.utils import t_batch_mode_transform
from torch import Tensor
from torch.nn import Module


class ProximalAcquisitionFunction(AcquisitionFunction):
    """A wrapper around AcquisitionFunctions to add proximal weighting of the
    acquisition function. Acquisition function is weighted via a squared exponential
    centered at the last training point, with varying lengthscales corresponding to
    `proximal_weights`. Can only be used with acquisition functions based on single
    batch models.

    Small values of `proximal_weights` corresponds to strong biasing towards recently
    observed points, which smoothes optimization with a small potential decrese in
    convergence rate.

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> EI = ExpectedImprovement(model, best_f=0.0)
        >>> proximal_weights = torch.ones(d)
        >>> EI_proximal = ProximalAcquisitionFunction(EI, proximal_weights)
        >>> eip = EI_proximal(test_X)
    """

    def __init__(
        self,
        acq_function: AcquisitionFunction,
        proximal_weights: Tensor,
    ) -> None:
        r"""Derived Acquisition Function weighted by proximity to recently
        observed point.

        Args:
            acq_function: The base acquisition function, operating on input tensors
                of feature dimension `d`.
            proximal_weights: A `d` dim tensor used to bias locality
                along each axis.
        """
        Module.__init__(self)

        self.acq_func = acq_function

        if hasattr(acq_function, "X_pending"):
            if acq_function.X_pending is not None:
                raise UnsupportedError(
                    "Proximal acquisition function requires `X_pending` to be None."
                )
            self.X_pending = acq_function.X_pending

        self.register_buffer("proximal_weights", proximal_weights)

        # check model for train_inputs and single batch
        if not hasattr(self.acq_func.model, "train_inputs"):
            raise UnsupportedError(
                "Acquisition function model must have " "`train_inputs`."
            )

        if (
            self.acq_func.model.batch_shape != torch.Size([])
            and self.acq_func.model.train_inputs[0].shape[1] != 1
        ):
            raise UnsupportedError(
                "Proximal acquisition function requires a single batch model"
            )

        # check to make sure that weights match the training data shape
        if (
            len(self.proximal_weights.shape) != 1
            or self.proximal_weights.shape[0]
            != self.acq_func.model.train_inputs[0][-1].shape[-1]
        ):
            raise ValueError(
                "`proximal_weights` must be a one dimensional tensor with "
                "same feature dimension as model."
            )

    @t_batch_mode_transform(expected_q=1, assert_output_shape=False)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate base acquisition function with proximal weighting.

        Args:
            X: Input tensor of feature dimension `d` .

        Returns:
            Base acquisition function evaluated on tensor `X` multiplied by proximal
            weighting.
        """
        last_X = self.acq_func.model.train_inputs[0][-1].reshape(1, 1, -1)
        diff = X - last_X

        M = torch.linalg.norm(diff / self.proximal_weights, dim=-1) ** 2

        proximal_acq_weight = torch.exp(-0.5 * M)
        return self.acq_func(X) * proximal_acq_weight.flatten()
