#! /usr/bin/env python3

r"""
Posterior Module to be used with GPyTorch models.
"""

from typing import Optional

import gpytorch
import torch
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from torch import Tensor

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
