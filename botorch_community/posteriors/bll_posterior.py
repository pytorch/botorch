#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math

import torch
from botorch.posteriors import GPyTorchPosterior, Posterior

from botorch_community.models.blls import AbstractBLLModel

from torch import Tensor


class BLLPosterior(Posterior):
    def __init__(
        self,
        posterior: GPyTorchPosterior,
        model: AbstractBLLModel,
        X: Tensor,
        output_dim: int,
    ):
        """A posterior for Bayesian last layer models.

        Args:
            posterior: A posterior object.
            model: A BLL model
            X: Input data on which the posterior was computed.
            output_dim: Output dimension of the model.
        """
        super().__init__()
        self.posterior = posterior
        self.model = model
        self.output_dim = output_dim
        self.X = X
        self._is_mt = output_dim > 1

    def rsample(
        self,
        sample_shape: torch.Size | None = None,
    ) -> Tensor:
        """
        For VBLLs, we need to sample from W and then create the
        generalized linear model to get posterior samples.

        Args:
            sample_shape: The shape of the samples to be drawn. If None, a single
                sample is drawn. Otherwise, the shape should be a tuple of integers
                representing the desired dimensions.

        Returns:
            A `(sample_shape) x N x output_dim`-dim Tensor of maximum posterior samples.
        """
        n_samples = 1 if sample_shape is None else math.prod(sample_shape)
        samples_list = [self.model.sample()(self.X) for _ in range(n_samples)]
        samples = torch.stack(samples_list, dim=0)

        # reshape to [sample_shape, n, output_dim]
        sample_shape = torch.Size([1]) if sample_shape is None else sample_shape
        new_shape = sample_shape + samples.shape[-2:]
        return samples.reshape(new_shape)

    @property
    def mean(self) -> Tensor:
        """The posterior mean."""
        # Directly return the mean from the underlying posterior
        return self.posterior.mean

    @property
    def variance(self) -> Tensor:
        """The posterior variance."""
        # Directly return the variance from the underlying posterior
        return self.posterior.variance

    @property
    def device(self) -> torch.device:
        return self.posterior.device

    @property
    def dtype(self) -> torch.dtype:
        """The torch dtype of the distribution."""
        return self.posterior.dtype
