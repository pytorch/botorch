# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from functools import lru_cache
from typing import Callable, Tuple

import torch
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from torch import Tensor


MCMC_DIM = -3  # Location of the MCMC batch dimension
TOL = 1e-6  # Bisection tolerance


def batched_bisect(
    f: Callable, target: float, bounds: Tensor, tol: float = TOL, max_steps: int = 32
):
    r"""Batched bisection with a fixed number of steps.

    Args:
        f: Target function that takes a `(b1 x ... x bk)`-dim tensor and returns a
            `(b1 x ... x bk)`-dim tensor.
        target: Scalar target value of type float.
        bounds: Lower and upper bounds, of size `2 x b1 x ... x bk`.
        tol: We termniate if all elements satisfy are within `tol` of the `target`.
        max_steps: Maximum number of bisection steps.

    Returns:
        Tensor X of size `b1 x ... x bk` such that `f(X) = target`.
    """
    # Make sure target is actually contained in the interval
    f1, f2 = f(bounds[0]), f(bounds[1])
    if not ((f1 <= target) & (target <= f2)).all():
        raise ValueError(
            "The target is not contained in the interval specified by the bounds"
        )
    bounds = bounds.clone()  # Will be modified in-place
    center = bounds.mean(dim=0)
    f_center = f(center)
    for _ in range(max_steps):
        go_left = f_center > target
        bounds[1, go_left] = center[go_left]
        bounds[0, ~go_left] = center[~go_left]
        center = bounds.mean(dim=0)
        f_center = f(center)
        # Check convergence
        if (f_center - target).abs().max() <= tol:
            return center
    return center


class FullyBayesianPosterior(GPyTorchPosterior):
    r"""A posterior for a fully Bayesian model.

    The MCMC batch dimension that corresponds to the models in the mixture is located
    at `MCMC_DIM` (defined at the top of this file). Note that while each MCMC sample
    corresponds to a Gaussian posterior, the fully Bayesian posterior is rather a
    mixture of Gaussian distributions.
    """

    def __init__(self, distribution: MultivariateNormal) -> None:
        r"""A posterior for a fully Bayesian model.

        Args:
            distribution: A GPyTorch MultivariateNormal (single-output case)
        """
        super().__init__(distribution=distribution)
        self._mean = (
            distribution.mean if self._is_mt else distribution.mean.unsqueeze(-1)
        )
        self._variance = (
            distribution.variance
            if self._is_mt
            else distribution.variance.unsqueeze(-1)
        )

    @property
    @lru_cache(maxsize=None)
    def mixture_mean(self) -> Tensor:
        r"""The posterior mean for the mixture of models."""
        return self._mean.mean(dim=MCMC_DIM)

    @property
    @lru_cache(maxsize=None)
    def mixture_variance(self) -> Tensor:
        r"""The posterior variance for the mixture of models."""
        num_mcmc_samples = self.mean.shape[MCMC_DIM]
        t1 = self._variance.sum(dim=MCMC_DIM) / num_mcmc_samples
        t2 = self._mean.pow(2).sum(dim=MCMC_DIM) / num_mcmc_samples
        t3 = -(self._mean.sum(dim=MCMC_DIM) / num_mcmc_samples).pow(2)
        return t1 + t2 + t3

    @lru_cache(maxsize=None)
    def quantile(self, value: Tensor) -> Tensor:
        r"""Compute the posterior quantiles for the mixture of models."""
        if value.numel() > 1:
            return torch.stack([self.quantile(v) for v in value], dim=0)
        if value <= 0 or value >= 1:
            raise ValueError("value is expected to be in the range (0, 1).")
        dist = torch.distributions.Normal(loc=self.mean, scale=self.variance.sqrt())
        if self.mean.shape[MCMC_DIM] == 1:  # Analytical solution
            return dist.icdf(value).squeeze(MCMC_DIM)
        icdf_val = dist.icdf(value)
        low = icdf_val.min(dim=MCMC_DIM).values - TOL
        high = icdf_val.max(dim=MCMC_DIM).values + TOL
        bounds = torch.cat((low.unsqueeze(0), high.unsqueeze(0)), dim=0)
        return batched_bisect(
            f=lambda x: dist.cdf(x.unsqueeze(MCMC_DIM)).mean(dim=MCMC_DIM),
            target=value.item(),
            bounds=bounds,
        )

    @property
    def batch_range(self) -> Tuple[int, int]:
        r"""The t-batch range.

        This is used in samplers to identify the t-batch component of the
        `base_sample_shape`. The base samples are expanded over the t-batches to
        provide consistency in the acquisition values, i.e., to ensure that a
        candidate produces same value regardless of its position on the t-batch.
        """
        if self._is_mt:
            return (0, -3)
        else:
            return (0, -2)
