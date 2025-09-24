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

from typing import Any, Optional, Union

import torch
from botorch.acquisition.objective import PosteriorTransform
from botorch.exceptions.errors import UnsupportedError

from botorch.logging import logger
from botorch.models.model import Model
from botorch.models.transforms.input import InputTransform
from botorch_community.models.utils.prior_fitted_network import (
    download_model,
    ModelPaths,
)
from botorch_community.posteriors.riemann import BoundedRiemannPosterior
from torch import Tensor
from torch.nn import Module


class PFNModel(Model):
    """Prior-data Fitted Network"""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        model: Module | None = None,
        checkpoint_url: str = ModelPaths.pfns4bo_hebo,
        train_Yvar: Tensor | None = None,
        batch_first: bool = False,
        constant_model_kwargs: dict[str, Any] | None = None,
        input_transform: InputTransform | None = None,
    ) -> None:
        """Initialize a PFNModel.

        Either a pre-trained PFN model can be provided via the model kwarg,
        or a checkpoint_url can be provided from which the model will be
        downloaded. This defaults to the pfns4bo_hebo model.

        Loading the model does an unsafe "weights_only=False" load, so
        it is essential that checkpoint_url be a trusted source.

        Args:
            train_X: A `n x d` tensor of training features.
            train_Y: A `n x m` tensor of training observations.
            model: A pre-trained PFN model with the following
                forward(train_X, train_Y, X) -> logit predictions of shape
                `n x b x c` where c is the number of discrete buckets
                borders: A `c+1`-dim tensor of bucket borders.
            checkpoint_url: The string URL of the PFN model to download and load.
                Will be ignored if model is provided.
            train_Yvar: Observed variance of train_Y. Currently ignored.
            batch_first: Whether the batch dimension is the first dimension of
                the input tensors. This is needed to support different PFN
                models. For batch-first x has shape `batch x seq_len x features`
                and for non-batch-first it has shape `seq_len x batch x features`.
            constant_model_kwargs: A dictionary of model kwargs that
                will be passed to the model in each forward pass.
            input_transform: A Botorch input transform.

        """
        super().__init__()
        if model is None:
            model = download_model(
                model_path=checkpoint_url,
            )

        if train_Yvar is not None:
            logger.debug("train_Yvar provided but ignored for PFNModel.")

        if not (1 <= train_Y.dim() <= 3):
            raise UnsupportedError("train_Y must be 1- to 3-dimensional.")

        if not (2 <= train_X.dim() <= 3):
            raise UnsupportedError("train_X must be 2- to 3-dimensional.")

        if train_Y.dim() == train_X.dim():
            if train_Y.shape[-1] > 1:
                raise UnsupportedError("Only 1 target allowed for PFNModel.")
            train_Y = train_Y.squeeze(-1)

        if (len(train_X.shape) != len(train_Y.shape) + 1) or (
            train_Y.shape != train_X.shape[:-1]
        ):
            raise UnsupportedError(
                "train_X and train_Y must have the same shape except "
                "for the last dimension."
            )

        if len(train_X.shape) == 2:
            # adding batch dimension
            train_X = train_X.unsqueeze(0)
            train_Y = train_Y.unsqueeze(0)

        with torch.no_grad():
            self.transformed_X = self.transform_inputs(
                X=train_X, input_transform=input_transform
            )

        self.train_X = train_X  # shape: `b x n x d`
        self.train_Y = train_Y  # shape: `b x n`
        self.pfn = model
        self.batch_first = batch_first
        self.constant_model_kwargs = constant_model_kwargs
        if input_transform is not None:
            self.input_transform = input_transform

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
            X: A `b'? x b? x q x d`-dim Tensor, where `d` is the dimension of the
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
            logger.warning(
                "observation_noise is not supported for PFNModel and is being ignored."
            )
        if posterior_transform is not None:
            raise UnsupportedError("posterior_transform is not supported for PFNModel.")

        if not (1 <= len(X.shape) <= 4):
            raise UnsupportedError("X must be 1- to 4-dimensional.")

        # X has shape b'? x b? x q? x d

        orig_X_shape = X.shape
        q_in_orig_X_shape = len(X.shape) > 2

        if len(X.shape) == 1:
            X = X.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # shape `b'=1 x b=1 x q=1 x d`
        elif len(X.shape) == 2:
            X = X.unsqueeze(1).unsqueeze(1)  # shape `b' x b=1 x q=1 x d`
        elif len(X.shape) == 3:
            if self.train_X.shape[0] == 1:
                X = X.unsqueeze(1)  # shape `b' x b=1 x q x d`
            else:
                X = X.unsqueeze(0)  # shape `b'=1 x b x q x d`

        # X has shape `b' x b x q x d`

        if X.shape[2] != 1:
            raise UnsupportedError("Only q=1 is supported for PFNModel.")

        # X has shape `b' x b x q=1 x d`
        X = self.transform_inputs(X)
        train_X = self.transformed_X  # shape `b x n x d`
        train_Y = self.train_Y  # shape `b x n`
        folded_X = X.transpose(0, 2).squeeze(0)  # shape `b x b' x d

        constant_model_kwargs = self.constant_model_kwargs or {}

        if self.batch_first:
            logits = self.pfn(
                train_X.float(),
                train_X.float(),
                folded_X.float(),
                **constant_model_kwargs,
            ).transpose(0, 1)
        else:
            logits = self.pfn(
                train_X.float().transpose(0, 1),
                train_Y.float().transpose(0, 1),
                folded_X.float().transpose(0, 1),
                **constant_model_kwargs,
            )

        # logits shape `b' x b x logits_dim`

        logits = logits.view(
            *orig_X_shape[:-1], -1
        )  # orig shape w/o q but logits_dim at end: `b'? x b? x q? x logits_dim`
        if q_in_orig_X_shape:
            logits = logits.squeeze(-2)  # shape `b'? x b? x logits_dim`

        probabilities = logits.softmax(dim=-1)

        return BoundedRiemannPosterior(self.pfn.criterion.borders, probabilities)
