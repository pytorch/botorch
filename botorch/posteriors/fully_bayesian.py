# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from functools import lru_cache
from typing import Callable, List, Optional

import torch
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.posteriors.posterior import PosteriorList
from gpytorch.distributions.multivariate_normal import MultivariateNormal
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
    mixture of Gaussian distributions. We provide convenience properties/methods for
    computing the mean, variance, median, and quantiles of this mixture.
    """

    def __init__(self, mvn: MultivariateNormal) -> None:
        r"""A posterior for a fully Bayesian model.

        Args:
            mvn: A GPyTorch MultivariateNormal (single-output case)
        """
        super().__init__(mvn=mvn)
        self._mean = mvn.mean if self._is_mt else mvn.mean.unsqueeze(-1)
        self._variance = mvn.variance if self._is_mt else mvn.variance.unsqueeze(-1)

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

    @property
    @lru_cache(maxsize=None)
    def mixture_median(self) -> Tensor:
        r"""The posterior median for the mixture of models."""
        return self.mixture_quantile(q=0.5)

    @lru_cache(maxsize=None)
    def mixture_quantile(self, q: float) -> Tensor:
        r"""The posterior quantiles for the mixture of models."""
        if not isinstance(q, float):
            raise ValueError("q is expected to be a float.")
        if q <= 0 or q >= 1:
            raise ValueError("q is expected to be in the range (0, 1).")
        q_tensor = torch.tensor(q).to(self.mean)
        dist = torch.distributions.Normal(loc=self.mean, scale=self.variance.sqrt())
        if self.mean.shape[MCMC_DIM] == 1:  # Analytical solution
            return dist.icdf(q_tensor).squeeze(MCMC_DIM)
        low = dist.icdf(q_tensor).min(dim=MCMC_DIM).values - TOL
        high = dist.icdf(q_tensor).max(dim=MCMC_DIM).values + TOL
        bounds = torch.cat((low.unsqueeze(0), high.unsqueeze(0)), dim=0)
        return batched_bisect(
            f=lambda x: dist.cdf(x.unsqueeze(MCMC_DIM)).mean(dim=MCMC_DIM),
            target=q,
            bounds=bounds,
        )


class FullyBayesianPosteriorList(PosteriorList):
    r"""A Posterior represented by a list of independent Posteriors.

    This posterior should only be used when at least one posterior is a
    `FullyBayesianPosterior`. Posteriors that aren't of type `FullyBayesianPosterior`
    are automatically reshaped to match the size of the fully Bayesian posteriors
    to allow mixing, e.g., deterministic and fully Bayesian models.

    Args:
        *posteriors: A variable number of single-outcome posteriors.

    Example:
        >>> p_1 = model_1.posterior(test_X)
        >>> p_2 = model_2.posterior(test_X)
        >>> p_12 = FullyBayesianPosteriorList(p_1, p_2)
    """

    def _get_mcmc_batch_dimension(self) -> int:
        """Return the number of MCMC samples in the corresponding batch dimension."""
        mcmc_samples = [
            p.mean.shape[MCMC_DIM]
            for p in self.posteriors
            if isinstance(p, FullyBayesianPosterior)
        ]
        if len(set(mcmc_samples)) > 1:
            raise NotImplementedError(
                "All MCMC batch dimensions must have the same size, got shapes: "
                f"{mcmc_samples}."
            )
        return mcmc_samples[0]

    @staticmethod
    def _reshape_tensor(X: Tensor, mcmc_samples: int) -> Tensor:
        """Reshape a tensor without an MCMC batch dimension to match the shape."""
        X = X.unsqueeze(MCMC_DIM)
        return X.expand(*X.shape[:MCMC_DIM], mcmc_samples, *X.shape[MCMC_DIM + 1 :])

    def _reshape_and_cat(self, Xs: List[Tensor]):
        r"""Reshape and cat a list of tensors."""
        mcmc_samples = self._get_mcmc_batch_dimension()
        return torch.cat(
            [
                x
                if isinstance(p, FullyBayesianPosterior)
                else self._reshape_tensor(x, mcmc_samples=mcmc_samples)
                for x, p in zip(Xs, self.posteriors)
            ],
            dim=-1,
        )

    @property
    def event_shape(self) -> torch.Size:
        r"""The event shape (i.e. the shape of a single sample)."""
        fully_bayesian_posteriors = [
            p for p in self.posteriors if isinstance(p, FullyBayesianPosterior)
        ]
        event_shape = fully_bayesian_posteriors[0].event_shape
        if not all(event_shape == p.event_shape for p in fully_bayesian_posteriors):
            # Make sure all fully Bayesian posteriors have the same event shape
            raise NotImplementedError(
                f"`{self.__class__.__name__}.event_shape` is only supported if all "
                "constituent posteriors have the same `event_shape`."
            )
        event_shapes = [event_shape for _ in self.posteriors]
        batch_shapes = [es[:-1] for es in event_shapes]
        return batch_shapes[0] + torch.Size([es[-1] for es in event_shapes])

    @property
    def mean(self) -> Tensor:
        r"""The mean of the posterior as a `(b) x n x m`-dim Tensor."""
        return self._reshape_and_cat(Xs=[p.mean for p in self.posteriors])

    @property
    def variance(self) -> Tensor:
        r"""The variance of the posterior as a `(b) x n x m`-dim Tensor."""
        return self._reshape_and_cat(Xs=[p.variance for p in self.posteriors])

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
        samples = super()._rsample(sample_shape=sample_shape, base_samples=base_samples)
        return self._reshape_and_cat(Xs=samples)
