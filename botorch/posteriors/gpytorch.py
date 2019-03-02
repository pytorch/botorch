#! /usr/bin/env python3

from typing import Optional

import gpytorch
import torch
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from torch import Tensor

from .posterior import Posterior


class GPyTorchPosterior(Posterior):
    """A posterior based on GPyTorch's multi-variate Normal distributions."""

    def __init__(self, mvn: MultivariateNormal) -> None:
        """A posterior based on GPyTorch's multi-variate Normal distributions.

        Args:
            mvn: A GPyTorch MultivariateNormal (single-output case) or
                MultitaskMultivariateNormal (multi-output case).
        """
        self.mvn = mvn
        self._is_mt = isinstance(mvn, MultitaskMultivariateNormal)

    @property
    def device(self) -> torch.device:
        return self.mvn.loc.device

    @property
    def dtype(self) -> torch.dtype:
        return self.mvn.loc.dtype

    @property
    def event_shape(self) -> torch.Size:
        shape = self.mvn.batch_shape + self.mvn.event_shape
        if not self._is_mt:
            shape += torch.Size([1])
        return shape

    def rsample(
        self,
        sample_shape: Optional[torch.Size] = None,
        base_samples: Optional[Tensor] = None,
    ) -> Tensor:
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
        # make sure there alwayas is an output dimension
        if not self._is_mt:
            samples = samples.unsqueeze(-1)
        return samples

    @property
    def mean(self):
        mean = self.mvn.mean
        if not self._is_mt:
            mean = mean.unsqueeze(-1)
        return mean

    @property
    def variance(self):
        variance = self.mvn.variance
        if not self._is_mt:
            variance = variance.unsqueeze(-1)
        return variance

    def get_base_samples(
        self,
        sample_shape: Optional[torch.Size] = None,
        collapse_batch_dims: bool = False,
    ) -> Tensor:
        if sample_shape is None:
            sample_shape = torch.Size()
        base_samples = self.mvn.get_base_samples(sample_shape=sample_shape)
        if not self._is_mt:
            base_samples = base_samples.unsqueeze(-1)
        # TODO: Push collapse_batch functionality upstream to gpytorch
        if collapse_batch_dims and len(self.batch_shape) > 0:
            # TODO: Figure out a cleaner way to do programmatic multi-indexing
            for _ in self.batch_shape:
                base_samples = base_samples.select(-3, 0).unsqueeze(-3)
        return base_samples
