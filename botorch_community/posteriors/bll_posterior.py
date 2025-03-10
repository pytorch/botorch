#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import math

import torch
from torch import Tensor

from botorch.posteriors import Posterior, GPyTorchPosterior

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from botorch_community.models.vblls import AbstractBLLModel


class BLLPosterior(Posterior):
    def __init__(
        self,
        posterior: GPyTorchPosterior,
        model: "AbstractBLLModel",
        X: Tensor,
        output_dim: int,
    ):
        super().__init__()
        self.posterior = posterior
        self.model = model
        self.old_model = model
        self.output_dim = output_dim
        self.X = X

    def rsample(
        self,
        sample_shape: torch.Size | None = None,
    ) -> Tensor:
        """
        For VBLLs, we need to sample from W and then create the
        generalized linear model to get posterior samples.
        """
        n_samples = 1 if sample_shape is None else math.prod(sample_shape)
        samples_list = [self.model.sample()(self.X) for _ in range(n_samples)]
        samples = torch.stack(samples_list, dim=0)
        new_shape = samples.shape[:-1]
        return samples.reshape(*new_shape, -1, self.output_dim)

    @property
    def mean(self) -> Tensor:
        """The posterior mean."""
        post_mean = self.posterior.mean.squeeze(-1)
        shape = post_mean.shape
        return post_mean.reshape(*shape[:-1], -1, self.output_dim)

    @property
    def variance(self) -> Tensor:
        """The posterior variance."""
        post_var = self.posterior.variance.squeeze(-1)
        shape = post_var.shape
        return post_var.reshape(*shape[:-1], -1, self.output_dim)

    @property
    def device(self) -> torch.device:
        return self.posterior.device

    @property
    def dtype(self) -> torch.dtype:
        """The torch dtype of the distribution."""
        return self.posterior.dtype
