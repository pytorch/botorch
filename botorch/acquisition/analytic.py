#!/usr/bin/env python3

"""
Analytic Acquisition Functions that evalute the posterior without performing
Monte-Carlo sampling.
"""

from abc import ABC
from typing import Union

import torch
from torch import Tensor
from torch.distributions import Normal

from ..models.gpytorch import GPyTorchModel
from ..models.model import Model
from .acquisition import AcquisitionFunction


class AnalyticAcquisitionFunction(AcquisitionFunction, ABC):
    """Base class for analytic acquisition functions."""

    pass


class SingleOutcomeAcquisitionFunction(AnalyticAcquisitionFunction, ABC):
    """Base class for single-outcome Acquistion functions."""

    def __init__(self, model: Model) -> None:
        if model.num_outputs > 1:
            raise RuntimeError(
                "SingleOutcomeAcquisitionFunction can only be used with "
                "single-outcome models"
            )
        super().__init__(model=model)


class ExpectedImprovement(SingleOutcomeAcquisitionFunction):
    """Single-outcome Expected Improvement.

    TODO: Add description + math
    """

    def __init__(
        self, model: Model, best_f: Union[float, Tensor], maximize: bool = True
    ) -> None:
        """Single-outcome analytic Expected Improvement.

        Args:
            model: A fitted single-outcome model.
            best_f: Either a scalar or a `b`-dim Tensor (batch mode) representing
                the best function value observed so far (assumed noiseless).
            maximize: If True, consider the problem a maximization problem.
        """
        super().__init__(model=model)
        self.maximize = maximize
        if not torch.is_tensor(best_f):
            best_f = torch.tensor(best_f)
        self.register_buffer("best_f", best_f)

    def forward(self, X: Tensor) -> Tensor:
        """Evaluate Expected Improvement on the candidate set X.

        Args:
            X: A `(b) x d`-dim Tensor of `(b)` t-batches of `d`-dim design
                points each.

        Returns:
            A `(b)`-dim Tensor of Expected Improvement values at the given
                design points `X`.
        """
        self.best_f = self.best_f.to(device=X.device, dtype=X.dtype)
        posterior = self.model.posterior(X.unsqueeze(-2))
        mean = posterior.mean.view(X.shape[:-1])
        sigma = posterior.variance.sqrt().clamp_min(1e-9).view(X.shape[:-1])
        u = (mean - self.best_f.expand_as(mean)) / sigma
        if not self.maximize:
            u = -u
        normal = Normal(torch.zeros_like(u), torch.ones_like(u))
        ucdf = normal.cdf(u)
        updf = torch.exp(normal.log_prob(u))
        ei = sigma * (updf + u * ucdf)
        return ei


class PosteriorMean(SingleOutcomeAcquisitionFunction):
    """Single-outcome posterior mean.

    TODO: Add description
    """

    def forward(self, X: Tensor) -> Tensor:
        """Evaluate the posterior mean on the candidate set X.

        Args:
            X: A `(b) x d`-dim Tensor of `(b)` t-batches of `d`-dim design
                points each.

        Returns:
            A `(b)`-dim Tensor of Posterior Mean values at the given
                design points `X`.
        """
        return self.model.posterior(X.unsqueeze(-2)).mean.view(X.shape[:-1])


class ProbabilityOfImprovement(SingleOutcomeAcquisitionFunction):
    """Single-outcome Probability of Improvement.

    TODO: Add description + math
    """

    def __init__(
        self, model: Model, best_f: Union[float, Tensor], maximize: bool = True
    ) -> None:
        """Single-outcome analytic Probability of Improvement.

        Args:
            model: A fitted single-outcome model.
            best_f: Either a scalar or a `b`-dim Tensor (batch mode) representing
                the best function value observed so far (assumed noiseless).
            maximize: If True, consider the problem a maximization problem.
        """
        super().__init__(model=model)
        self.maximize = maximize
        if not torch.is_tensor(best_f):
            best_f = torch.tensor(best_f)
        self.register_buffer("best_f", best_f)

    def forward(self, X: Tensor) -> Tensor:
        """Evaluate the Probability of Improvement on the candidate set X.

        Args:
            X: A `(b) x d`-dim Tensor of `(b)` t-batches of `d`-dim design
                points each.

        Returns:
            A `(b)`-dim Tensor of Probability of Improvement values at the given
                design points `X`.
        """
        self.best_f = self.best_f.to(device=X.device, dtype=X.dtype)
        posterior = self.model.posterior(X.unsqueeze(-2))
        mean, sigma = posterior.mean, posterior.variance.sqrt()
        mean = posterior.mean.view(X.shape[:-1])
        sigma = posterior.variance.sqrt().clamp_min(1e-9).view(X.shape[:-1])
        u = (mean - self.best_f.expand_as(mean)) / sigma
        if not self.maximize:
            u = -u
        normal = Normal(torch.zeros_like(u), torch.ones_like(u))
        return normal.cdf(u)


class UpperConfidenceBound(SingleOutcomeAcquisitionFunction):
    """Single-outcome Upper Confidence Bound.

    TODO: Add description + math
    """

    def __init__(
        self, model: Model, beta: Union[float, Tensor], maximize: bool = True
    ) -> None:
        """Single-outcome Upper Confidence Bound.

        Args:
            model: A fitted single-outcome GP model (must be in batch mode if
                candidate sets X will be)
            beta: Either a scalar or a one-dim tensor with `b` elements (batch mode)
                representing the trade-off parameter between mean and covariance
            maximize: If True, consider the problem a maximization problem.
        """
        super().__init__(model=model)
        self.maximize = maximize
        if not torch.is_tensor(beta):
            beta = torch.tensor(beta)
        self.register_buffer("beta", beta)

    def forward(self, X: Tensor) -> Tensor:
        """Evaluate the Upper Confidence Bound on the candidate set X.

        Args:
            X: A `(b) x d`-dim Tensor of `(b)` t-batches of `d`-dim design
                points each.

        Returns:
            A `(b)`-dim Tensor of Upper Confidence Bound values at the given
                design points `X`.
        """
        self.beta = self.beta.to(device=X.device, dtype=X.dtype)
        posterior = self.model.posterior(X.unsqueeze(-2))
        mean = posterior.mean.view(X.shape[:-1])
        variance = posterior.variance.view(X.shape[:-1])
        delta = (self.beta.expand_as(mean) * variance).sqrt()
        if self.maximize:
            return mean + delta
        else:
            return mean - delta


class NoisyExpectedImprovement(ExpectedImprovement):
    """Single-outcome Noisy Expected Improvement.

    THIS FUNCTION CURRENTLY RELIES ON USING A GPYTORCH ExactGP

    TODO: Add description + math
    """

    def __init__(
        self,
        model: GPyTorchModel,
        X_observed: Tensor,
        num_fantasies: int = 20,
        maximize: bool = True,
    ) -> None:
        """Single-outcome analytic Noisy Expected Improvement.

        Args:
            model: A fitted single-outcome model.
            X_observed: A `m x d` Tensor of observed points that are likely to
                be the best observed points so far.
            num_fantasies: The number of fantasies to generate. The higher this
                number the more accurate the model (at the expense of model
                complexity and performance).
            maximize: If True, consider the problem a maximization problem.
        """
        self.num_fantasies = num_fantasies
        self.register_buffer("X_observed", X_observed)
        # construct fantasy model
        posterior = model.posterior(X_observed)
        # TODO: Use qMC sampling here
        Y_fantasized = posterior.sample(torch.Size([num_fantasies]))
        # TODO (see T40723547): these observations should be noiseless
        fantasy_model = model.get_fantasy_model(X_observed, Y_fantasized)
        # TODO: get Tensor of best_f values from fantasies
        # best_f = _get_best_f(...)
        best_f = 0.0  # temp hack
        super().__init__(model=fantasy_model, best_f=best_f, maximize=maximize)

    def forward(self, X: Tensor) -> Tensor:
        """Evaluate Expected Improvement on the candidate set X.

        Args:
            X: A `(b) x d`-dim Tensor of `(b)` t-batches of `d`-dim design
                points each.

        Returns:
            A `(b)`-dim Tensor of Noisy Expected Improvement values at the given
                design points `X`.
        """
        # evaluate all fantasy models (i.e. batches of the model) at the same points
        X = X.expand(X.shape[:-2] + torch.Size([self.num_fantasies]) + X.shape[-2:])
        return super().forward(X).mean(dim=-1)
