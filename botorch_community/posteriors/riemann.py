#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Abstract base module for all botorch posteriors.
"""

from __future__ import annotations

from typing import Callable, Optional, Union

import torch
from botorch.posteriors.posterior import Posterior
from torch import Tensor


class BoundedRiemannPosterior(Posterior):
    """
    Notes: Bounded posterior for now, will work on unbounded posteriors.
    This is also only over 1 test point, not batches.
    """

    def __init__(self, borders, probabilities):
        r"""Bounded Riemann Posterior.

        A posterior distribution represented by a piecewise constant probability density
        function over a bounded domain. The domain is divided into buckets defined by
        borders, with each bucket having an associated probability.

        Args:
            borders: A tensor of shape `(n_buckets + 1,)` defining the boundaries of
                the buckets. Must be monotonically increasing.
            probabilities: A tensor of shape `(..., n_buckets,)` defining the
                probability mass in each bucket. Must sum to 1 in the last dim.
        """

        assert torch.allclose(
            probabilities.sum(-1),
            torch.tensor(1.0, device=probabilities.device, dtype=probabilities.dtype),
            atol=1e-3,
        ), f"Probabilities must sum to 1, but sum to {probabilities.sum()}."

        self.borders = borders
        self.probabilities = probabilities
        self.cumprobs = torch.cumsum(self.probabilities, -1)

    def integrate(self, ag_integrate_fn: Callable[[Tensor, Tensor], Tensor]) -> Tensor:
        r"""Integrate over the posterior, by calculating
            $$\int_{min}^{max} ag(y) posterior(y) dy$$.

        Args:
            ag_integral_fn: A vectorized function that integrates ag from lower_bound
                to upper_bound, that is, $$\int_{lower_bound}^{upper_bound} ag(y) dy$$.

        Returns:
            Tensor: The integral of the posterior.
        """
        all_lower = self.borders[:-1]
        all_upper = self.borders[1:]
        bucket_results = ag_integrate_fn(all_lower, all_upper)
        return (bucket_results * (self.probabilities / (all_upper - all_lower))).sum(-1)

    def rsample(
        self,
        sample_shape: Optional[torch.Size] = None,
    ) -> Tensor:
        r"""Sample from the posterior (with gradients).

        Args:
            sample_shape: A `torch.Size` object specifying the sample shape. To
                draw `n` samples, set to `torch.Size([n])`. To draw `b` batches
                of `n` samples each, set to `torch.Size([b, n])`.

        Returns:
            Samples from the posterior, a tensor of shape
            `self._extended_shape(sample_shape=sample_shape)`.
        """
        sample_shape = sample_shape if sample_shape is not None else torch.Size([1])
        z = torch.rand(sample_shape)
        return self.rsample_from_base_samples(sample_shape, z)

    def rsample_from_base_samples(
        self, sample_shape: torch.Size, base_samples: Tensor
    ) -> Tensor:
        if base_samples.shape[: len(sample_shape)] != sample_shape:
            raise RuntimeError(
                "`sample_shape` disagrees with shape of `base_samples`. "
                f"Got {sample_shape=} and {base_samples.shape=}."
            )
        return self.icdf(base_samples)

    @property
    def device(self) -> torch.device:
        r"""The torch device of the distribution."""
        return self.borders.device

    @property
    def dtype(self) -> torch.dtype:
        r"""The torch dtype of the distribution."""
        return self.borders.dtype

    @property
    def mean(self):
        r"""The mean of the posterior distribution."""
        bucket_widths = self.borders[1:] - self.borders[:-1]
        bucket_means = self.borders[:-1] + bucket_widths / 2
        return (self.probabilities @ bucket_means).unsqueeze(-1)

    @property
    def mean_of_square(self) -> torch.Tensor:
        """Computes E[x^2]."""
        left_borders = self.borders[:-1]
        right_borders = self.borders[1:]
        bucket_mean_of_square = (
            left_borders.square()
            + right_borders.square()
            + left_borders * right_borders
        ) / 3.0
        return (self.probabilities @ bucket_mean_of_square).unsqueeze(-1)

    @property
    def variance(self) -> torch.Tensor:
        """Computes the variance via Var[x] = E[x^2] - E[x]^2."""
        return self.mean_of_square - self.mean.square()

    def confidence_region(
        self, confidence_level: float = 0.95
    ) -> tuple[Tensor, Tensor]:
        r"""
        Compute the lower and upper bounds of the confidence region.
        Args:
            confidence_level: The probability between the bounds
                of the confidence interval/region.
                Use .954 for 2 sigma of a normal distribution.
        """
        side_probs = (1.0 - confidence_level) / 2
        return self.icdf(side_probs), self.icdf(1.0 - side_probs)

    def icdf(self, value: Union[Tensor, float]) -> Tensor:
        r"""Inverse cdf (with gradients).
        Use value to get the index of the bucket that contains the value
        and then interpolate between the left and right borders of the bucket

        Args:
            value: The value at which to evaluate the inverse CDF.

        Returns:
            The inverse CDF of the posterior at the given value(s).
            The shape of the return is the shape of value, with the batch
            shape of the probs (all dims up to the final dim) appended
            with a final trailing dimension of 1, for the dim of the dist.
        """

        # final shape is (batch_shape, -1)
        value = torch.as_tensor(
            value, device=self.borders.device, dtype=self.borders.dtype
        )
        value_shape = value.shape
        # shape of cumprobs is (batch_shape, n_buckets)
        value = value.broadcast_to(size=(*self.cumprobs.shape[:-1], *value_shape))
        value = value.reshape(*self.cumprobs.shape[:-1], -1)

        # get first index where cumprobs > value
        index = torch.searchsorted(self.cumprobs, value)

        left_border = self.borders[index]
        right_border = self.borders[index + 1]

        bucket_width = right_border - left_border
        right_cum_probs = torch.gather(self.cumprobs, -1, index)
        prob_width = torch.gather(self.probabilities, -1, index)

        bucket_proportion_remaining = (right_cum_probs - value) / prob_width
        result = left_border + (1 - bucket_proportion_remaining) * bucket_width

        # reshape to (value_shape, batch_shape, 1)
        result = result.transpose(0, -1)
        return result.reshape(*value_shape, *self.cumprobs.shape[:-1], 1)
