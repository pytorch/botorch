#!/usr/bin/env python3

"""
Batch acquisition functions using the reparameterization trick in combination
with (quasi) Monte-Carlo sampling.

.. [Wilson2017reparam]
    Wilson, J. T., Moriconi, R., Hutter, F., & Deisenroth, M. P. (2017). The
    reparameterization trick for acquisition functions. arXiv preprint
    arXiv:1712.00424.
"""

import math
from abc import ABC, abstractmethod
from typing import Optional, Union

import torch
from torch import Tensor

from ..models.model import Model
from ..utils.transforms import batch_mode_transform, match_batch_shape
from .acquisition import AcquisitionFunction
from .objective import IdentityMCObjective, MCAcquisitionObjective
from .sampler import MCSampler, SobolQMCNormalSampler


class MCAcquisitionFunction(AcquisitionFunction, ABC):
    """Abstract base class for Monte-Carlo based batch acquisition functions."""

    def __init__(
        self,
        model: Model,
        sampler: Optional[MCSampler] = None,
        objective: Optional[MCAcquisitionObjective] = None,
    ) -> None:
        """Constructor for the MCAcquisitionFunction base class.

        Args:
            model: A fitted model.
            sampler: The sampler used to draw base samples. Defaults to
                `SobolQMCNormalSampler(num_samples=500, collapse_batch_dims=True)`
            objective: THe MCAcquisitionObjective under which the samples are
                evaluated. Defaults to `IdentityMCObjective()`.
        """
        super().__init__(model=model)
        if sampler is None:
            sampler = SobolQMCNormalSampler(num_samples=500, collapse_batch_dims=True)
        self.add_module("sampler", sampler)
        if objective is None:
            objective = IdentityMCObjective()
        self.add_module("objective", objective)

    @abstractmethod
    def forward(self, X: Tensor) -> Tensor:
        """Takes in a `(b) x q x d` X Tensor of `b` t-batches with `q` `d`-dim
        design points each, expands and concatenates `self.X_pending` and
        returns a one-dimensional Tensor with `b` elements."""
        pass


class qExpectedImprovement(MCAcquisitionFunction):
    """MC-based batch Expected Improvement

    TODO: description + math
    """

    def __init__(
        self,
        model: Model,
        best_f: Union[float, Tensor],
        sampler: Optional[MCSampler] = None,
        objective: Optional[MCAcquisitionObjective] = None,
    ) -> None:
        """q-Expected Improvement.

        Args:
            model: A fitted model.
            best_f: The best (feasible) function value observed so far (assumed
                noiseless).
            sampler: The sampler used to draw base samples. Defaults to
                `SobolQMCNormalSampler(num_samples=500, collapse_batch_dims=True)`
            objective: The MCAcquisitionObjective under which the samples are
                evaluated. Defaults to `IdentityMCObjective()`.
        """
        super().__init__(model=model, sampler=sampler, objective=objective)
        if not torch.is_tensor(best_f):
            best_f = torch.tensor(float(best_f))
        self.register_buffer("best_f", best_f)

    @batch_mode_transform
    def forward(self, X: Tensor) -> Tensor:
        """Evaluate qExpectedImprovement on the candidate set `X`.

        Args:
            X: A `(b) x q x d`-dim Tensor of `(b)` t-batches with `q` `d`-dim
                design points each.

        Returns:
            Tensor: A `(b)`-dim Tensor of Expected Improvement values at the
                given design points `X`.
        """
        posterior = self.model.posterior(X)
        samples = self.sampler(posterior)
        obj = self.objective(samples)
        obj = (obj - self.best_f).clamp_min(0)
        q_ei = obj.max(dim=2)[0].mean(dim=0)
        return q_ei


class qNoisyExpectedImprovement(MCAcquisitionFunction):
    """q-Noisy Expected Improvement with constraints.

    TODO: description + math
    """

    def __init__(
        self,
        model: Model,
        X_baseline: Tensor,
        sampler: Optional[MCSampler] = None,
        objective: Optional[MCAcquisitionObjective] = None,
    ) -> None:
        """q-Noisy Expected Improvement.

        Args:
            model: A fitted model.
            X_baseline: A `m x d`-dim Tensor of `m` design points that have
                either already been observed or whose evaluation is pending.
                These points are considered as the potential best design point.
            sampler: The sampler used to draw base samples. Defaults to
                `SobolQMCNormalSampler(num_samples=500, collapse_batch_dims=True)`
            objective: The MCAcquisitionObjective under which the samples are
                evaluated. Defaults to `IdentityMCObjective()`.
        """
        super().__init__(model=model, sampler=sampler, objective=objective)
        self.register_buffer("X_baseline", X_baseline)

    @batch_mode_transform
    def forward(self, X: Tensor) -> Tensor:
        """Evaluate qNoisyExpectedImprovement on the candidate set `X`.

        Args:
            X: A `(b) x q x d`-dim Tensor of `(b)` t-batches with `q` `d`-dim
                design points each.

        Returns:
            Tensor: A `(b)`-dim Tensor of Noisy Expected Improvement values at
                the given design points `X`.
        """
        q = X.shape[-2]
        X_full = torch.cat([X, match_batch_shape(self.X_baseline, X)], dim=-2)
        # TODO (T41248036): Implement more efficient way to compute posterior
        # over both training and test points in GPyTorch
        posterior = self.model.posterior(X_full)
        samples = self.sampler(posterior)
        obj = self.objective(samples)
        diffs = obj[:, :, :q].max(dim=2)[0] - obj[:, :, q:].max(dim=2)[0]
        return diffs.clamp_min(0).mean(dim=0)


class qProbabilityOfImprovement(MCAcquisitionFunction):
    """q-Probability of Improvement.

    TODO: description + math
    """

    def __init__(
        self,
        model: Model,
        best_f: Union[float, Tensor],
        sampler: Optional[MCSampler] = None,
        objective: Optional[MCAcquisitionObjective] = None,
        tau: float = 1e-3,
    ) -> None:
        """q-Expected Improvement.

        Args:
            model: A fitted model.
            best_f: The best (feasible) function value observed so far (assumed
                noiseless).
            sampler: The sampler used to draw base samples. Defaults to
                `SobolQMCNormalSampler(num_samples=500, collapse_batch_dims=True)`
            objective: The MCAcquisitionObjective under which the samples are
                evaluated. Defaults to `IdentityMCObjective()`.
            tau: The temperature parameter used in the sigmoid approximation
                of the step function. Smaller values yield more accurate
                approximations of the function, but result in gradients
                estimates with higher variance.
        """
        super().__init__(model=model, sampler=sampler, objective=objective)
        if not torch.is_tensor(best_f):
            best_f = torch.tensor(float(best_f))
        self.register_buffer("best_f", best_f)
        if not torch.is_tensor(tau):
            tau = torch.tensor(float(tau))
        self.register_buffer("tau", tau)

    @batch_mode_transform
    def forward(self, X: Tensor) -> Tensor:
        """Evaluate qProbabilityOfImprovement on the candidate set `X`.

        Args:
            X: A `(b) x q x d`-dim Tensor of `(b)` t-batches with `q` `d`-dim
                design points each.

        Returns:
            Tensor: A `(b)`-dim Tensor of Probability of Improvement values at
                the given design points `X`.
        """
        posterior = self.model.posterior(X)
        samples = self.sampler(posterior)
        max_obj = self.objective(samples).max(dim=-2)[0]
        val = torch.sigmoid((max_obj - self.best_f) / self.tau).mean(dim=0)
        return val


class qSimpleRegret(MCAcquisitionFunction):
    """q-Simple Regret.

    TODO: description + math
    """

    @batch_mode_transform
    def forward(self, X: Tensor) -> Tensor:
        """Evaluate qSimpleRegret on the candidate set `X`.

        Args:
            X: A `(b) x q x d`-dim Tensor of `(b)` t-batches with `q` `d`-dim
                design points each.

        Returns:
            Tensor: A `(b)`-dim Tensor of Simple Regret values at the given
                design points `X`.
        """
        posterior = self.model.posterior(X)
        samples = self.sampler(posterior)
        val = samples.max(dim=2)[0].mean(dim=0)
        return val


class qUpperConfidenceBound(MCAcquisitionFunction):
    """q-Upper Confidence Bound.

    TODO: description + math
    """

    def __init__(
        self,
        model: Model,
        beta: float,
        sampler: Optional[MCSampler] = None,
        objective: Optional[MCAcquisitionObjective] = None,
    ) -> None:
        """q-Upper Confidence Bound.

        Args:
            model: A fitted model.
            beta: Controls tradeoff between mean and standard deviation in UCB.
            sampler: The sampler used to draw base samples. Defaults to
                `SobolQMCNormalSampler(num_samples=500, collapse_batch_dims=True)`
            objective: The MCAcquisitionObjective under which the samples are
                evaluated. Defaults to `IdentityMCObjective()`.
        """
        super().__init__(model=model, sampler=sampler, objective=objective)
        self.register_buffer("beta", torch.tensor(float(beta)))

    @batch_mode_transform
    def forward(self, X: Tensor) -> Tensor:
        """Evaluate qUpperConfidenceBound on the candidate set `X`.

        Args:
            X: A `(b) x q x d`-dim Tensor of `(b)` t-batches with `q` `d`-dim
                design points each.

        Returns:
            Tensor: A `(b)`-dim Tensor of Upper Confidence Bound values at the
                given design points `X`.
        """
        posterior = self.model.posterior(X)
        samples = self.sampler(posterior)
        mean = posterior.mean
        ucb_samples = mean + math.sqrt(self.beta * math.pi / 2) * (samples - mean).abs()
        return ucb_samples.max(dim=-2)[0].mean(dim=0)
