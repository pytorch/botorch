#!/usr/bin/env python3

r"""
Batch acquisition functions using the reparameterization trick in combination
with (quasi) Monte-Carlo sampling. See [Rezende2014reparam]_ and
[Wilson2017reparam]_

.. [Rezende2014reparam]
    D. J. Rezende, S. Mohamed, and D. Wierstra. Stochastic backpropagation and
    approximate inference in deep generative models. ICML 2014.

.. [Wilson2017reparam]
    J. T. Wilson, R. Moriconi, F. Hutter, and M. P. Deisenroth.
    The reparameterization trick for acquisition functions. ArXiv 2017.
"""

import math
from abc import ABC, abstractmethod
from typing import Optional, Union

import torch
from torch import Tensor

from ..models.model import Model
from ..utils.transforms import match_batch_shape, t_batch_mode_transform
from .acquisition import AcquisitionFunction
from .objective import IdentityMCObjective, MCAcquisitionObjective
from .sampler import MCSampler, SobolQMCNormalSampler


class MCAcquisitionFunction(AcquisitionFunction, ABC):
    r"""Abstract base class for Monte-Carlo based batch acquisition functions."""

    def __init__(
        self,
        model: Model,
        sampler: Optional[MCSampler] = None,
        objective: Optional[MCAcquisitionObjective] = None,
    ) -> None:
        r"""Constructor for the MCAcquisitionFunction base class.

        Args:
            model: A fitted model.
            sampler: The sampler used to draw base samples. Defaults to
                `SobolQMCNormalSampler(num_samples=500, collapse_batch_dims=True)`.
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
        r"""Takes in a `(b) x q x d` X Tensor of `b` t-batches with `q` `d`-dim
        design points each, expands and concatenates `self.X_pending` and
        returns a one-dimensional Tensor with `b` elements."""
        pass  # pragma: no cover


class qExpectedImprovement(MCAcquisitionFunction):
    r"""MC-based batch Expected Improvement.

    This computes qEI by
    (1) sampling the joint posterior over q points
    (2) evaluating the improvement over the current best for each sample
    (3) maximizing over q
    (4) averaging over the samples

    `qEI(X) = E(max(max Y - best_f, 0)), Y ~ f(X), where X = (x_1,...,x_q)`

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> best_f = train_Y.max()[0]
        >>> sampler = SobolQMCNormalSampler(1000)
        >>> qEI = qExpectedImprovement(model, best_f, sampler)
        >>> qei = qEI(test_X)
    """

    def __init__(
        self,
        model: Model,
        best_f: Union[float, Tensor],
        sampler: Optional[MCSampler] = None,
        objective: Optional[MCAcquisitionObjective] = None,
    ) -> None:
        r"""q-Expected Improvement.

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

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate qExpectedImprovement on the candidate set `X`.

        Args:
            X: A `(b) x q x d`-dim Tensor of `(b)` t-batches with `q` `d`-dim
                design points each.

        Returns:
            A `(b)`-dim Tensor of Expected Improvement values at the given
            design points `X`.
        """
        posterior = self.model.posterior(X)
        samples = self.sampler(posterior)
        obj = self.objective(samples)
        obj = (obj - self.best_f).clamp_min(0)
        q_ei = obj.max(dim=-1)[0].mean(dim=0)
        return q_ei


class qNoisyExpectedImprovement(MCAcquisitionFunction):
    r"""MC-based batch Noisy Expected Improvement.

    This function does not assume a `best_f` is known (which would require
    noiseless observations). Instead, it uses samples from the joint posterior
    over the `q` test points and previously observed points. The improvement
    over previously observed points is computed for each sample and averaged.

    `qNEI(X) = E(max(max Y - max Y_baseline, 0))`, where
    `(Y, Y_baseline) ~ f((X, X_baseline)), X = (x_1,...,x_q)`

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> sampler = SobolQMCNormalSampler(1000)
        >>> qNEI = qNoisyExpectedImprovement(model, train_X, sampler)
        >>> qnei = qNEI(test_X)
    """

    def __init__(
        self,
        model: Model,
        X_baseline: Tensor,
        sampler: Optional[MCSampler] = None,
        objective: Optional[MCAcquisitionObjective] = None,
    ) -> None:
        r"""q-Noisy Expected Improvement.

        Args:
            model: A fitted model.
            X_baseline: A `m x d`-dim Tensor of `m` design points that have
                either already been observed or whose evaluation is pending.
                These points are considered as the potential best design point.
            sampler: The sampler used to draw base samples. Defaults to
                `SobolQMCNormalSampler(num_samples=500, collapse_batch_dims=True)`.
            objective: The MCAcquisitionObjective under which the samples are
                evaluated. Defaults to `IdentityMCObjective()`.
        """
        super().__init__(model=model, sampler=sampler, objective=objective)
        self.register_buffer("X_baseline", X_baseline)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate qNoisyExpectedImprovement on the candidate set `X`.

        Args:
            X: A `(b) x q x d`-dim Tensor of `(b)` t-batches with `q` `d`-dim
                design points each.

        Returns:
            A `(b)`-dim Tensor of Noisy Expected Improvement values at the given
            design points `X`.
        """
        q = X.shape[-2]
        X_full = torch.cat([X, match_batch_shape(self.X_baseline, X)], dim=-2)
        # TODO (T41248036): Implement more efficient way to compute posterior
        # over both training and test points in GPyTorch
        posterior = self.model.posterior(X_full)
        samples = self.sampler(posterior)
        obj = self.objective(samples)
        diffs = obj[:, :, :q].max(dim=-1)[0] - obj[:, :, q:].max(dim=-1)[0]
        return diffs.clamp_min(0).mean(dim=0)


class qProbabilityOfImprovement(MCAcquisitionFunction):
    r"""MC-based batch Probability of Improvement.

    Estimates the probability of improvement over the current best observed
    value by sampling from the joint posterior distribution of the q-batch.
    MC-based estimates of a probability involves taking expectation of an
    indicator function; to support auto-differntiation, the indicator is
    replaced with a sigmoid function with temperature parameter `tau`.

    `qPI(X) = P(max Y >= best_f), Y ~ f(X), X = (x_1,...,x_q)`

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> best_f = train_Y.max()[0]
        >>> sampler = SobolQMCNormalSampler(1000)
        >>> qPI = qProbabilityOfImprovement(model, best_f, sampler)
        >>> qpi = qPI(test_X)
    """

    def __init__(
        self,
        model: Model,
        best_f: Union[float, Tensor],
        sampler: Optional[MCSampler] = None,
        objective: Optional[MCAcquisitionObjective] = None,
        tau: float = 1e-3,
    ) -> None:
        r"""q-Probability of Improvement.

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

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate qProbabilityOfImprovement on the candidate set `X`.

        Args:
            X: A `(b) x q x d`-dim Tensor of `(b)` t-batches with `q` `d`-dim
                design points each.

        Returns:
            A `(b)`-dim Tensor of Probability of Improvement values at the given
            design points `X`.
        """
        posterior = self.model.posterior(X)
        samples = self.sampler(posterior)
        obj = self.objective(samples)
        max_obj = obj.max(dim=-1)[0]
        val = torch.sigmoid((max_obj - self.best_f) / self.tau).mean(dim=0)
        return val


class qSimpleRegret(MCAcquisitionFunction):
    r"""MC-based batch Simple Regret.

    Samples from the joint posterior over the q-batch and computes the simple
    regret.

    `qSR(X) = E(max Y), Y ~ f(X), X = (x_1,...,x_q)`

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> sampler = SobolQMCNormalSampler(1000)
        >>> qSR = qSimpleRegret(model, sampler)
        >>> qsr = qSR(test_X)
    """

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate qSimpleRegret on the candidate set `X`.

        Args:
            X: A `(b) x q x d`-dim Tensor of `(b)` t-batches with `q` `d`-dim
                design points each.

        Returns:
            A `(b)`-dim Tensor of Simple Regret values at the given design
            points `X`.
        """
        posterior = self.model.posterior(X)
        samples = self.sampler(posterior)
        obj = self.objective(samples)
        val = obj.max(dim=-1)[0].mean(dim=0)
        return val


class qUpperConfidenceBound(MCAcquisitionFunction):
    r"""MC-based batch Upper Confidence Bound.

    Uses a reparameterization to extend UCB to qUCB for q > 1 (See Appendix A
    of [Wilson2017reparam].)

    `qUCB = E(max(mu + |Y_tilde - mu|))`, where `Y_tilde ~ N(mu, beta pi/2 Sigma)`
    and `f(X)` has distribution `N(mu, Sigma)`.

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> sampler = SobolQMCNormalSampler(1000)
        >>> qUCB = qUpperConfidenceBound(model, 0.1, sampler)
        >>> qucb = qUCB(test_X)
    """

    def __init__(
        self,
        model: Model,
        beta: float,
        sampler: Optional[MCSampler] = None,
        objective: Optional[MCAcquisitionObjective] = None,
    ) -> None:
        r"""q-Upper Confidence Bound.

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

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate qUpperConfidenceBound on the candidate set `X`.

        Args:
            X: A `(b) x q x d`-dim Tensor of `(b)` t-batches with `q` `d`-dim
                design points each.

        Returns:
            A `(b)`-dim Tensor of Upper Confidence Bound values at the given
            design points `X`.
        """
        posterior = self.model.posterior(X)
        samples = self.sampler(posterior)
        obj = self.objective(samples)
        mean = obj.mean(dim=0)
        ucb_samples = mean + math.sqrt(self.beta * math.pi / 2) * (obj - mean).abs()
        return ucb_samples.max(dim=-1)[0].mean(dim=0)
