#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Deterministic (degenerate) posteriors. Used in conjunction with deterministic
models.
"""

from __future__ import annotations

from typing import Optional

import torch
from botorch.posteriors.posterior import Posterior
from torch import Tensor


class DeterministicPosterior(Posterior):
    r"""Deterministic posterior."""

    def __init__(self, values: Tensor) -> None:
        r"""
        Args:
            values: Values of the samples produced by this posterior.
        """
        self.values = values

    @property
    def base_sample_shape(self) -> torch.Size:
        r"""The shape of a base sample used for constructing posterior samples.

        This function may be overwritten by subclasses in case `base_sample_shape`
        and `event_shape` do not agree (e.g. if the posterior is a Multivariate
        Gaussian that is not full rank).
        """
        return torch.Size()

    @property
    def device(self) -> torch.device:
        r"""The torch device of the posterior."""
        return self.values.device

    @property
    def dtype(self) -> torch.dtype:
        r"""The torch dtype of the posterior."""
        return self.values.dtype

    @property
    def event_shape(self) -> torch.Size:
        r"""The event shape (i.e. the shape of a single sample)."""
        return self.values.shape

    @property
    def mean(self) -> Tensor:
        r"""The mean of the posterior as a `(b) x n x m`-dim Tensor."""
        return self.values

    @property
    def variance(self) -> Tensor:
        r"""The variance of the posterior as a `(b) x n x m`-dim Tensor.

        As this is a deterministic posterior, this is a tensor of zeros.
        """
        return torch.zeros_like(self.values)

    def rsample(
        self,
        sample_shape: Optional[torch.Size] = None,
        base_samples: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Sample from the posterior (with gradients).

        For the deterministic posterior, this just returns the values expanded
        to the requested shape.

        Args:
            sample_shape: A `torch.Size` object specifying the sample shape. To
                draw `n` samples, set to `torch.Size([n])`. To draw `b` batches
                of `n` samples each, set to `torch.Size([b, n])`.
            base_samples: An (optional) Tensor of `N(0, I)` base samples of
                appropriate dimension, typically obtained from a `Sampler`.
                Ignored in construction of the samples (used only for shape
                validation).

        Returns:
            A `sample_shape x event`-dim Tensor of samples from the posterior.
        """
        if sample_shape is None:
            sample_shape = torch.Size([1])
        if base_samples is not None:
            if base_samples.shape[: len(sample_shape)] != sample_shape:
                raise RuntimeError("sample_shape disagrees with shape of base_samples.")
        return self.values.expand(sample_shape + self.values.shape)
