#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Posterior Module to be used with GPyTorch models.
"""

from __future__ import annotations

from contextlib import ExitStack
from typing import Optional

import torch
from botorch.exceptions.errors import BotorchTensorDimensionError
from botorch.posteriors.base_samples import _reshape_base_samples_non_interleaved
from botorch.posteriors.posterior import Posterior
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from linear_operator import settings as linop_settings
from linear_operator.operators import (
    BlockDiagLinearOperator,
    LinearOperator,
    SumLinearOperator,
)
from torch import Tensor


class GPyTorchPosterior(Posterior):
    r"""A posterior based on GPyTorch's multi-variate Normal distributions."""

    def __init__(self, mvn: MultivariateNormal) -> None:
        r"""A posterior based on GPyTorch's multi-variate Normal distributions.

        Args:
            mvn: A GPyTorch MultivariateNormal (single-output case) or
                MultitaskMultivariateNormal (multi-output case).
        """
        self.mvn = mvn
        self._is_mt = isinstance(mvn, MultitaskMultivariateNormal)

    @property
    def base_sample_shape(self) -> torch.Size:
        r"""The shape of a base sample used for constructing posterior samples."""
        shape = self.mvn.batch_shape + self.mvn.base_sample_shape
        if not self._is_mt:
            shape += torch.Size([1])
        return shape

    @property
    def device(self) -> torch.device:
        r"""The torch device of the posterior."""
        return self.mvn.loc.device

    @property
    def dtype(self) -> torch.dtype:
        r"""The torch dtype of the posterior."""
        return self.mvn.loc.dtype

    @property
    def event_shape(self) -> torch.Size:
        r"""The event shape (i.e. the shape of a single sample) of the posterior."""
        shape = self.mvn.batch_shape + self.mvn.event_shape
        if not self._is_mt:
            shape += torch.Size([1])
        return shape

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
            A `sample_shape x event_shape`-dim Tensor of samples from the posterior.
        """
        if sample_shape is None:
            sample_shape = torch.Size([1])
        if base_samples is not None:
            if base_samples.shape[: len(sample_shape)] != sample_shape:
                raise RuntimeError("sample_shape disagrees with shape of base_samples.")
            # get base_samples to the correct shape
            base_samples = base_samples.expand(sample_shape + self.event_shape)
            if self._is_mt:
                base_samples = _reshape_base_samples_non_interleaved(
                    mvn=self.mvn, base_samples=base_samples, sample_shape=sample_shape
                )
            # remove output dimension in single output case
            else:
                base_samples = base_samples.squeeze(-1)
        with ExitStack() as es:
            if linop_settings._fast_covar_root_decomposition.is_default():
                es.enter_context(linop_settings._fast_covar_root_decomposition(False))
            samples = self.mvn.rsample(
                sample_shape=sample_shape, base_samples=base_samples
            )
        # make sure there always is an output dimension
        if not self._is_mt:
            samples = samples.unsqueeze(-1)
        return samples

    @property
    def mean(self) -> Tensor:
        r"""The posterior mean."""
        mean = self.mvn.mean
        if not self._is_mt:
            mean = mean.unsqueeze(-1)
        return mean

    @property
    def variance(self) -> Tensor:
        r"""The posterior variance."""
        variance = self.mvn.variance
        if not self._is_mt:
            variance = variance.unsqueeze(-1)
        return variance


def scalarize_posterior(
    posterior: GPyTorchPosterior, weights: Tensor, offset: float = 0.0
) -> GPyTorchPosterior:
    r"""Affine transformation of a multi-output posterior.

    Args:
        posterior: The posterior over `m` outcomes to be scalarized.
            Supports `t`-batching.
        weights: A tensor of weights of size `m`.
        offset: The offset of the affine transformation.

    Returns:
        The transformed (single-output) posterior. If the input posterior has
            mean `mu` and covariance matrix `Sigma`, this posterior has mean
            `weights^T * mu` and variance `weights^T Sigma w`.

    Example:
        Example for a model with two outcomes:

        >>> X = torch.rand(1, 2)
        >>> posterior = model.posterior(X)
        >>> weights = torch.tensor([0.5, 0.25])
        >>> new_posterior = scalarize_posterior(posterior, weights=weights)
    """
    if weights.ndim > 1:
        raise BotorchTensorDimensionError("`weights` must be one-dimensional")
    mean = posterior.mean
    q, m = mean.shape[-2:]
    batch_shape = mean.shape[:-2]
    if m != weights.size(0):
        raise RuntimeError("Output shape not equal to that of weights")
    mvn = posterior.mvn
    cov = mvn.lazy_covariance_matrix if mvn.islazy else mvn.covariance_matrix

    if m == 1:  # just scaling, no scalarization necessary
        new_mean = offset + (weights[0] * mean).view(*batch_shape, q)
        new_cov = weights[0] ** 2 * cov
        new_mvn = MultivariateNormal(new_mean, new_cov)
        return GPyTorchPosterior(new_mvn)

    new_mean = offset + (mean @ weights).view(*batch_shape, q)

    if q == 1:
        new_cov = weights.unsqueeze(-2) @ (cov @ weights.unsqueeze(-1))
    else:
        # we need to handle potentially different representations of the multi-task mvn
        if mvn._interleaved:
            w_cov = weights.repeat(q).unsqueeze(0)
            sum_shape = batch_shape + torch.Size([q, m, q, m])
            sum_dims = (-1, -2)
        else:
            # special-case the independent setting
            if isinstance(cov, BlockDiagLinearOperator):
                new_cov = SumLinearOperator(
                    *[
                        cov.base_linear_op[..., i, :, :] * weights[i].pow(2)
                        for i in range(cov.base_linear_op.size(-3))
                    ]
                )
                new_mvn = MultivariateNormal(new_mean, new_cov)
                return GPyTorchPosterior(new_mvn)

            w_cov = torch.repeat_interleave(weights, q).unsqueeze(0)
            sum_shape = batch_shape + torch.Size([m, q, m, q])
            sum_dims = (-2, -3)

        cov_scaled = w_cov * cov * w_cov.transpose(-1, -2)
        # TODO: Do not instantiate full covariance for LinearOperators
        # (ideally we simplify this in GPyTorch:
        # https://github.com/cornellius-gp/gpytorch/issues/1055)
        if isinstance(cov_scaled, LinearOperator):
            cov_scaled = cov_scaled.to_dense()
        new_cov = cov_scaled.view(sum_shape).sum(dim=sum_dims[0]).sum(dim=sum_dims[1])

    new_mvn = MultivariateNormal(new_mean, new_cov)
    return GPyTorchPosterior(new_mvn)
