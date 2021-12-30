#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Callable, Optional

import torch
from botorch.posteriors.posterior import Posterior
from torch import Tensor


class TransformedPosterior(Posterior):
    r"""An generic transformation of a posterior (implicitly represented)"""

    def __init__(
        self,
        posterior: Posterior,
        sample_transform: Callable[[Tensor], Tensor],
        mean_transform: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
        variance_transform: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
    ) -> None:
        r"""An implicitly represented transformed posterior

        Args:
            posterior: The posterior object to be transformed.
            sample_transform: A callable applying a sample-level transform to a
                `sample_shape x batch_shape x q x m`-dim tensor of samples from
                the original posterior, returning a tensor of samples of the
                same shape.
            mean_transform: A callable transforming a 2-tuple of mean and
                variance (both of shape `batch_shape x m x o`) of the original
                posterior to the mean of the transformed posterior.
            variance_transform: A callable transforming a 2-tuple of mean and
                variance (both of shape `batch_shape x m x o`) of the original
                posterior to a variance of the transformed posterior.
        """
        self._posterior = posterior
        self._sample_transform = sample_transform
        self._mean_transform = mean_transform
        self._variance_transform = variance_transform

    @property
    def base_sample_shape(self) -> torch.Size:
        r"""The shape of a base sample used for constructing posterior samples."""
        return self._posterior.base_sample_shape

    @property
    def device(self) -> torch.device:
        r"""The torch device of the posterior."""
        return self._posterior.device

    @property
    def dtype(self) -> torch.dtype:
        r"""The torch dtype of the posterior."""
        return self._posterior.dtype

    @property
    def event_shape(self) -> torch.Size:
        r"""The event shape (i.e. the shape of a single sample)."""
        return self._posterior.event_shape

    @property
    def mean(self) -> Tensor:
        r"""The mean of the posterior as a `batch_shape x n x m`-dim Tensor."""
        if self._mean_transform is None:
            raise NotImplementedError("No mean transform provided.")
        try:
            variance = self._posterior.variance
        except (NotImplementedError, AttributeError):
            variance = None
        return self._mean_transform(self._posterior.mean, variance)

    @property
    def variance(self) -> Tensor:
        r"""The variance of the posterior as a `batch_shape x n x m`-dim Tensor."""
        if self._variance_transform is None:
            raise NotImplementedError("No variance transform provided.")
        return self._variance_transform(self._posterior.mean, self._posterior.variance)

    def rsample(
        self,
        sample_shape: Optional[torch.Size] = None,
        base_samples: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Sample from the posterior (with gradients).

        Args:
            sample_shape: A `torch.Size` object specifying the sample shape. To
                draw `n` samples, set to `torch.Size([n])`. To draw `b` batches
                of `n` samples each, set to `torch.Size([b, n])`.
            base_samples: An (optional) Tensor of `N(0, I)` base samples of
                appropriate dimension, typically obtained from a `Sampler`.
                This is used for deterministic optimization.

        Returns:
            A `sample_shape x event`-dim Tensor of samples from the posterior.
        """
        samples = self._posterior.rsample(
            sample_shape=sample_shape, base_samples=base_samples
        )
        return self._sample_transform(samples)
