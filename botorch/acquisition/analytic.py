#!/usr/bin/env python3

"""
Analytic Acquisition Functions that evaluate the posterior without performing
Monte-Carlo sampling.
"""

from abc import ABC
from typing import Dict, Optional, Tuple, Union

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


def convert_pre_hook(module, *args):
    module.to(args[0][0])


class ConstrainedExpectedImprovement(AnalyticAcquisitionFunction):
    """Single-outcome Expected Improvement.

    TODO: Add description + math
    """

    def __init__(
        self,
        model: Model,
        best_f: Union[float, Tensor],
        objective_index: int,
        constraints: Dict[int, Tuple[Optional[float], Optional[float]]],
        maximize: bool = True,
    ) -> None:
        """Single-outcome analytic Expected Improvement.

        Args:
            model: A fitted single-outcome model.
            best_f: Either a scalar or a `b`-dim Tensor (batch mode) representing
                the best function value observed so far (assumed noiseless).
            objective_index: The index of the objective.
            constraints: A dictionary of the form `{i: [lower, upper]}`, where
                `i` is the output index, and `lower` and `upper` are lower and upper
                bounds on that output (resp. interpreted as -Inf / Inf if None)
            maximize: If True, consider the problem a maximization problem.
        """
        super().__init__(model=model)
        self.maximize = maximize
        self.objective_index = objective_index
        self.constraints = constraints
        if not torch.is_tensor(best_f):
            best_f = torch.tensor(best_f)
        self.register_buffer("best_f", best_f)
        self._preprocess_bounds(constraints)
        self.register_forward_pre_hook(convert_pre_hook)

    def forward(self, X: Tensor) -> Tensor:
        """Evaluate Expected Improvement on the candidate set X.

        Args:
            X: A `(b) x d`-dim Tensor of `(b)` t-batches of `d`-dim design
                points each.

        Returns:
            A `(b)`-dim Tensor of Expected Improvement values at the given
                design points `X`.
        """
        posterior = self.model.posterior(X.unsqueeze(dim=-2))
        means = posterior.mean.squeeze(dim=-2)  # (b) x t
        sigmas = posterior.variance.squeeze(dim=-2).sqrt().clamp_min(1e-9)  # (b) x t

        # (b) x 1
        mean_obj = means[..., [self.objective_index]]
        sigma_obj = sigmas[..., [self.objective_index]]
        u = (mean_obj - self.best_f.expand_as(mean_obj)) / sigma_obj
        if not self.maximize:
            u = -u
        normal = Normal(
            torch.zeros_like(u, device=u.device, dtype=u.dtype),
            torch.ones_like(u, device=u.device, dtype=u.dtype),
        )
        ei_pdf = torch.exp(normal.log_prob(u))  # (b) x 1
        ei_cdf = normal.cdf(u)
        ei = sigma_obj * (ei_pdf + u * ei_cdf)
        prob_feas = self._compute_prob_feas(X, means, sigmas)
        ei.mul_(prob_feas)
        return ei.squeeze(dim=-1)

    def _preprocess_bounds(
        self, constraints: Dict[int, Tuple[Optional[float], Optional[float]]]
    ):
        constraint_lower, self.constraint_lower_inds = [], []
        constraint_upper, self.constraint_upper_inds = [], []
        constraint_both, self.constraint_both_inds = [], []
        constraint_indices = list(constraints.keys())
        if len(constraint_indices) == 0:
            raise ValueError("There must be at least one constraint.")
        if self.objective_index in constraint_indices:
            raise ValueError(
                "Output corresponding to objective should not be a constraint."
            )
        for k in constraint_indices:
            if constraints[k][0] is not None and constraints[k][1] is not None:
                if constraints[k][1] <= constraints[k][0]:
                    raise ValueError("Upper bound is less than the lower bound.")
                self.constraint_both_inds.append(k)
                constraint_both.append([constraints[k][0], constraints[k][1]])
            elif constraints[k][0] is not None:
                self.constraint_lower_inds.append(k)
                constraint_lower.append(constraints[k][0])
            elif constraints[k][1] is not None:
                self.constraint_upper_inds.append(k)
                constraint_upper.append(constraints[k][1])
        self.register_buffer(
            "constraint_both", torch.tensor(constraint_both, dtype=torch.float)
        )
        self.register_buffer(
            "constraint_lower", torch.tensor(constraint_lower, dtype=torch.float)
        )
        self.register_buffer(
            "constraint_upper", torch.tensor(constraint_upper, dtype=torch.float)
        )

    def _compute_prob_feas(self, X: Tensor, means: Tensor, sigmas: Tensor) -> Tensor:
        # This function does casework for upper bound, lower bound, and both-sided
        # bounds. Another way to do it would be to use 'inf' and -'inf' for the
        # one-sided bounds and use the logic for the both-sided case. But this
        # causes issue with autograd since we get 0 * inf. Investigate later.
        output_shape = list(X.shape)
        output_shape[-1] = 1
        prob_feas = torch.ones(output_shape, device=X.device, dtype=X.dtype)
        if len(self.constraint_lower_inds) > 0:
            normal_lower = _construct_dist(means, sigmas, self.constraint_lower_inds)
            prob_feas.mul_(
                torch.prod(
                    1 - normal_lower.cdf(self.constraint_lower), dim=-1, keepdim=True
                )
            )
        if len(self.constraint_upper_inds) > 0:
            normal_upper = _construct_dist(means, sigmas, self.constraint_upper_inds)
            prob_feas.mul_(
                torch.prod(
                    normal_upper.cdf(self.constraint_upper), dim=-1, keepdim=True
                )
            )
        if len(self.constraint_both_inds) > 0:
            normal_both = _construct_dist(means, sigmas, self.constraint_both_inds)
            prob_feas.mul_(
                torch.prod(
                    normal_both.cdf(self.constraint_both[:, 1])
                    - normal_both.cdf(self.constraint_both[:, 0]),
                    dim=-1,
                    keepdim=True,
                )
            )
        return prob_feas


def _construct_dist(means: Tensor, sigmas: Tensor, inds: Tensor):
    mean = means[..., inds]
    sigma = sigmas[..., inds]
    return Normal(loc=mean, scale=sigma)


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
