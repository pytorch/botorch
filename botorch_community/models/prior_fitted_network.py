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
from botorch.utils.transforms import match_batch_shape
from botorch_community.models.utils.prior_fitted_network import (
    download_model,
    ModelPaths,
)
from botorch_community.posteriors.riemann import BoundedRiemannPosterior
from gpytorch.likelihoods.gaussian_likelihood import FixedNoiseGaussianLikelihood
from pfns.train import MainConfig  # @manual=//pytorch/PFNs:PFNs
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
        load_training_checkpoint: bool = False,
    ) -> None:
        """Initialize a PFNModel.

        Either a pre-trained PFN model can be provided via the model kwarg,
        or a checkpoint_url can be provided from which the model will be
        downloaded. This defaults to the pfns4bo_hebo model.

        Loading the model does an unsafe "weights_only=False" load, so
        it is essential that checkpoint_url be a trusted source.

        Args:
            train_X: A `n x d` tensor of training features.
            train_Y: A `n x 1` tensor of training observations.
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
            load_training_checkpoint: Whether to load a training checkpoint as
                produced by the PFNs training code, see github.com/automl/PFNs.

        """
        super().__init__()
        if model is None:
            model = download_model(
                model_path=checkpoint_url,
            )

        if load_training_checkpoint:
            # the model is not an actual model, but a training checkpoint
            # make a model out of it
            checkpoint = model
            config = MainConfig.from_dict(checkpoint["config"])
            model = config.model.create_model()
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()

        if train_Yvar is not None:
            logger.debug("train_Yvar provided but ignored for PFNModel.")

        if train_Y.dim() != 2:
            raise UnsupportedError("train_Y must be 2-dimensional.")

        if train_X.dim() != 2:
            raise UnsupportedError("train_X must be 2-dimensional.")

        if train_Y.shape[-1] > 1:
            raise UnsupportedError("Only 1 target allowed for PFNModel.")

        if train_X.shape[0] != train_Y.shape[0]:
            raise UnsupportedError(
                "train_X and train_Y must have the same number of rows."
            )

        with torch.no_grad():
            self.transformed_X = self.transform_inputs(
                X=train_X, input_transform=input_transform
            )

        self.train_X = train_X  # shape: (n, d)
        self.train_Y = train_Y  # shape: (n, 1)
        # Downstream botorch tooling expects a likelihood to be specified,
        # so here we use a FixedNoiseGaussianLikelihood that is unused.
        if train_Yvar is None:
            train_Yvar = torch.zeros_like(train_Y)
        self.likelihood = FixedNoiseGaussianLikelihood(noise=train_Yvar)
        self.pfn = model.to(device=train_X.device)
        self.batch_first = batch_first
        self.constant_model_kwargs = constant_model_kwargs or {}
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
            X: A b? x q? x d`-dim Tensor, where `d` is the dimension of the
                feature space.
            output_indices: **Currenlty not supported for PFNModel.**
            observation_noise: **Currently not supported for PFNModel**.
            posterior_transform: **Currently not supported for PFNModel**.

        Returns:
            A `BoundedRiemannPosterior`, representing a batch of b? x q?`
            distributions.
        """
        self.pfn.eval()
        if output_indices is not None:
            raise UnsupportedError(
                "output_indices is not None. PFNModel should not "
                "be a multi-output model."
            )
        if observation_noise:
            logger.warning(
                "observation_noise is not supported for PFNModel and is being ignored."
            )
        if posterior_transform is not None:
            raise UnsupportedError("posterior_transform is not supported for PFNModel.")

        orig_X_shape = X.shape  # X has shape b? x q? x d
        X = self.prepare_X(X)  # shape (b, q, d)
        train_X = match_batch_shape(self.transformed_X, X)  # shape (b, n, d)
        train_Y = match_batch_shape(self.train_Y, X)  # shape (b, n, 1)

        probabilities = self.pfn_predict(
            X=X, train_X=train_X, train_Y=train_Y
        )  # (b, q, num_buckets)
        probabilities = probabilities.view(
            *orig_X_shape[:-1], -1
        )  # (b?, q?, num_buckets)

        # Get posterior with the right dtype
        borders = self.pfn.criterion.borders.to(X.dtype)
        return BoundedRiemannPosterior(
            borders=borders,
            probabilities=probabilities,
        )

    def prepare_X(self, X: Tensor) -> Tensor:
        if len(X.shape) > 3:
            raise UnsupportedError(f"X must be at most 3-d, got {X.shape}.")
        while len(X.shape) < 3:
            X = X.unsqueeze(0)

        X = self.transform_inputs(X)  # shape (b , q, d)
        return X

    def pfn_predict(self, X: Tensor, train_X: Tensor, train_Y: Tensor) -> Tensor:
        """
        X has shape (b, q, d)
        train_X has shape (b, n, d)
        train_Y has shape (b, n, 1)
        """
        if not self.batch_first:
            X = X.transpose(0, 1)  # shape (q, b, d)
            train_X = train_X.transpose(0, 1)  # shape (n, b, d)
            train_Y = train_Y.transpose(0, 1)  # shape (n, b, 1)

        logits = self.pfn(
            train_X.float(),
            train_Y.float(),
            X.float(),
            **self.constant_model_kwargs,
        )
        if not self.batch_first:
            logits = logits.transpose(0, 1)  # shape (b, q, num_buckets)
        logits = logits.to(X.dtype)

        probabilities = logits.softmax(dim=-1)  # shape (b, q, num_buckets)
        return probabilities
