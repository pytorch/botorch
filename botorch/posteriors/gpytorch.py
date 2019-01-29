#! /usr/bin/env python3

from typing import Optional

import gpytorch
import torch
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from gpytorch.lazy.diag_lazy_tensor import DiagLazyTensor
from torch import Tensor

from .posterior import Posterior


class GPyTorchPosterior(Posterior):
    def __init__(self, mvn: MultivariateNormal) -> None:
        self.mvn = mvn

    @property
    def device(self) -> torch.device:
        return self.mvn.loc.device

    @property
    def dtype(self) -> torch.dtype:
        return self.mvn.loc.dtype

    @property
    def event_shape(self) -> torch.Size:
        """Return the event shape (i.e. the shape of a single sample)"""
        return self.mvn.event_shape

    def rsample(
        self,
        sample_shape: Optional[torch.Size] = None,
        base_samples: Optional[Tensor] = None,
        eps: float = 1e-6,
    ) -> Tensor:
        """Sample from the posterior (with gradients)

        Args:
            sample_shape: shape of the samples. To draw `n` samples, set to
                `torch.Size([n])`. To draw `b` batches of `n` samples each, set
                to `torch.Size([b, n])`
            base_samples: An (optional) tensor of iid normal base samples of
                appropriate dimension, typically obtained using `get_base_samples`.
                If provided, takes priority over `sample_shape` (though it must
                comply with `sample_shape`)
            eps: A small value to add to the diagonal of the covariance matrix if
                the covariance matrix is degenerate.
        Returns:
            A tensor of shape `sample_shape + event_shape`, where `event_shape`
                is the shape of a single sample from the posterior.

        """
        if sample_shape is not None and base_samples is not None:
            if tuple(base_samples.shape[: len(sample_shape)]) != tuple(sample_shape):
                raise RuntimeError("Sample shape disagrees with base_samples.")
        if sample_shape is None and base_samples is None:
            kwargs = {}
        elif base_samples is not None:
            kwargs = {"base_samples": base_samples}
        elif sample_shape is not None:
            kwargs = {"sample_shape": sample_shape}
        failed = False
        samples = None
        with gpytorch.settings.fast_computations(covar_root_decomposition=False):
            try:
                samples = self.mvn.rsample(**kwargs)
            except RuntimeError:
                failed = True
        # if the covariance matrix is degenerate, add eps to the diagonal
        if failed or torch.any(torch.isnan(samples)):
            covar = self.mvn.lazy_covariance_matrix
            eps_diag = DiagLazyTensor(
                torch.ones(
                    (1, covar.shape[-1]),
                    dtype=self.mvn.mean.dtype,
                    device=self.mvn.mean.device,
                )
                * eps
            )
            if self.mvn.mean.dim() > 2:
                mvn_class = MultitaskMultivariateNormal
            else:
                mvn_class = MultivariateNormal
            mvn = mvn_class(self.mvn.mean, covar + eps_diag)
            with gpytorch.settings.fast_computations(covar_root_decomposition=False):
                samples = mvn.rsample(**kwargs)
        return samples

    @property
    def mean(self):
        """Posterior mean"""
        return self.mvn.mean

    @property
    def variance(self):
        """Posterior variance"""
        return self.mvn.variance

    def zero_mean_mvn_samples(self, num_samples: int) -> Tensor:
        return self.mvn.lazy_covariance_matrix.zero_mean_mvn_samples(num_samples)

    def get_base_samples(self, sample_shape: Optional[torch.Size] = None) -> Tensor:
        return self.mvn.get_base_samples(sample_shape=sample_shape)
