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

    @property
    def device(self) -> torch.device:
        return self.mvn.loc.device

    @property
    def dtype(self) -> torch.dtype:
        return self.mvn.loc.dtype

    @property
    def event_shape(self) -> torch.Size:
        event_shape = self.mvn.event_shape
        if not isinstance(self.mvn, MultitaskMultivariateNormal):
            event_shape += torch.Size([1])
        return event_shape

    def rsample(
        self,
        sample_shape: Optional[torch.Size] = None,
        base_samples: Optional[Tensor] = None,
    ) -> Tensor:
        if sample_shape is not None and base_samples is not None:
            if tuple(base_samples.shape[: len(sample_shape)]) != tuple(sample_shape):
                raise RuntimeError("Sample shape disagrees with base_samples.")
        if sample_shape is None and base_samples is None:
            kwargs = {}
        elif base_samples is not None:
            if (
                not isinstance(self.mvn, MultitaskMultivariateNormal)
                and base_samples.shape[-1] == 1
            ):
                base_samples = base_samples.squeeze(-1)
            kwargs = {"base_samples": base_samples}
        elif sample_shape is not None:
            kwargs = {"sample_shape": sample_shape}
        samples = None
        with gpytorch.settings.fast_computations(covar_root_decomposition=False):
            samples = self.mvn.rsample(**kwargs)
        # make sure there alwayas is an output dimension
        if not isinstance(self.mvn, MultitaskMultivariateNormal):
            samples = samples.unsqueeze(-1)
        return samples

    @property
    def mean(self):
        mean = self.mvn.mean
        if not isinstance(self.mvn, MultitaskMultivariateNormal):
            mean = mean.unsqueeze(-1)
        return mean

    @property
    def variance(self):
        variance = self.mvn.variance
        if not isinstance(self.mvn, MultitaskMultivariateNormal):
            variance = variance.unsqueeze(-1)
        return variance

    def get_base_samples(self, sample_shape: Optional[torch.Size] = None) -> Tensor:
        base_samples = self.mvn.get_base_samples(sample_shape=sample_shape)
        if not isinstance(self.mvn, MultitaskMultivariateNormal):
            base_samples = base_samples.unsqueeze(-1)
        return base_samples
