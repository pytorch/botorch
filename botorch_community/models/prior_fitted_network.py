#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
This module defines the botorch model for PFNs (`PFNModel`) and it
provides handy helpers to download pretrained, public PFNs
with `download_model` and model paths with `ModelPaths`.
For the latter to work `pfns4bo` must be installed.
"""

from __future__ import annotations

from typing import Optional, Union

import torch.nn as nn
from botorch.acquisition.objective import PosteriorTransform
from botorch.exceptions.errors import UnsupportedError
from botorch.models.model import Model
from botorch_community.posteriors.riemann import BoundedRiemannPosterior
from torch import Tensor


class PFNModel(Model):
    """Prior-data Fitted Network"""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        model: nn.Module,
        train_Yvar: Tensor | None = None,
        device: str = "cpu",
        batch_first: bool = False,
        constant_model_kwargs: dict | None = None,
    ) -> None:
        """Initialize a PFNModel.

        Args:
            train_X: A `n x d` tensor of training features.
            train_Y: A `n x m` tensor of training observations.
            model: A pre-trained PFN model with the following
                forward(train_X, train_Y, X) -> logit predictions of shape
                `n x b x c` where c is the number of discrete buckets
                borders: A `c+1`-dim tensor of bucket borders
            train_Yvar: Not yet supported.
            device: The device on which the model will be instantiated.
            batch_first: Whether the batch dimension is the first dimension of
                the input tensors.
        """
        if train_Yvar is not None:
            raise UnsupportedError("train_Yvar is not supported for PFNModel.")
        super().__init__()

        assert 1 <= train_Y.dim() <= 2, "train_Y must be 1- or 2-dimensional."

        if train_Y.dim() == 2:
            assert train_Y.shape[-1] == 1, "train_Y must be 1-dimensional."
            train_Y = train_Y.squeeze(-1)

        self.train_X = train_X
        self.train_Y = train_Y
        self.pfn = model.to(device)
        self.device = device
        self.batch_first = batch_first
        self.constant_model_kwargs = constant_model_kwargs

    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[list[int]] = None,
        observation_noise: Union[bool, Tensor] = False,
        posterior_transform: Optional[PosteriorTransform] = None,
    ) -> BoundedRiemannPosterior:
        r"""Computes the posterior over model outputs at the provided points.

        Note: The input transforms should be applied here using
            `self.transform_inputs(X)` after the `self.eval()` call and before
            any `model.forward` or `model.likelihood` calls.

        Args:
            X: A `b x q x d`-dim Tensor, where `d` is the dimension of the
                feature space, `q` is the number of points considered jointly,
                and `b` is the batch dimension.
                We only allow `q=1` for PFNModel, so q can also be omitted, i.e.
                `b x d`-dim Tensor.
            **Currently not supported for PFNModel**.
            output_indices: **Currenlty not supported for PFNModel.**
            observation_noise: **Currently not supported for PFNModel**.
            posterior_transform: **Currently not supported for PFNModel**.

        Returns:
            A `BoundedRiemannPosterior` object, representing a batch of `b` joint
            distributions over `q` points and `m` outputs each.
        """
        self.pfn.eval()
        if output_indices is not None:
            raise RuntimeError(
                "output_indices is not None. PFNModel should not "
                "be a multi-output model."
            )
        if observation_noise:
            raise UnsupportedError("observation_noise is not supported for PFNModel.")
        if posterior_transform is not None:
            raise UnsupportedError("posterior_transform is not supported for PFNModel.")

        if len(X.shape) > 2:
            if X.shape[1] > 1:
                raise NotImplementedError(
                    "q must be 1 for PFNModel."
                )  # add support later
        else:
            X = X.unsqueeze(1)

        train_X = self.train_X
        train_Y = self.train_Y

        self.pfn.to(self.device)
        train_X = train_X.to(self.device)
        train_Y = train_Y.to(self.device)
        X = X.to(self.device)

        constant_model_kwargs = self.constant_model_kwargs or {}

        if self.batch_first:
            logits = self.pfn(
                train_X.unsqueeze(0).float(),
                train_Y.unsqueeze(0).float(),
                X.float().transpose(0, 1),
                **constant_model_kwargs,
            ).squeeze(0)
        else:
            logits = self.pfn(
                train_X.unsqueeze(1).float(),
                train_Y.unsqueeze(1).float(),
                X.float(),
                **constant_model_kwargs,
            ).squeeze(1)

        probabilities = logits.softmax(dim=-1)

        return BoundedRiemannPosterior(self.pfn.criterion.borders, probabilities)
