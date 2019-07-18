#! /usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

r"""
Posterior Module to be used with GPyTorch models.
"""

from typing import Optional

import gpytorch
import torch
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from torch import Tensor

from ..exceptions.errors import UnsupportedError
from .posterior import Posterior


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
            # remove output dimension in single output case
            if not self._is_mt:
                base_samples = base_samples.squeeze(-1)
        with gpytorch.settings.fast_computations(covar_root_decomposition=False):
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
        posterior: The posterior to be transformed. Must be single-point (`q=1`).
            Supports `t`-batching.
        weights: An single-dimensional tensor of weights. Number of elements
            must be the numbe of outputs of the posterior.
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
    mean = posterior.mean
    if mean.shape[-1] != len(weights):
        raise RuntimeError("Output shape not equal to that of weights")
    if mean.shape[-2] != 1:
        raise UnsupportedError("scalarize_posterior currently not supported for q>1")
    # no need to use lazy here since q=1
    cov = posterior.mvn.covariance_matrix
    batch_shape = cov.shape[:-2]
    new_cov = ((cov @ weights) @ weights).view(*batch_shape, 1, 1)
    new_mean = offset + (mean @ weights).view(*batch_shape, 1)
    new_mvn = MultivariateNormal(new_mean, new_cov)
    return GPyTorchPosterior(new_mvn)
