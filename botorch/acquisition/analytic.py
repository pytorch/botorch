#!/usr/bin/env python3

r"""
Analytic Acquisition Functions that evaluate the posterior without performing
Monte-Carlo sampling.
"""

from abc import ABC
from typing import Dict, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.distributions import Normal

from ..exceptions import UnsupportedError
from ..models.gpytorch import GPyTorchModel
from ..models.model import Model
from ..posteriors.posterior import Posterior
from ..utils.transforms import convert_to_target_pre_hook, q_batch_mode_transform
from .acquisition import AcquisitionFunction
from .sampler import SobolQMCNormalSampler


class AnalyticAcquisitionFunction(AcquisitionFunction, ABC):
    r"""Base class for analytic acquisition functions."""

    def _validate_single_output_posterior(self, posterior: Posterior) -> None:
        # Validates that the computed posterior is single-output and raises
        # an UnsupportedError if not.
        if posterior.event_shape[-1] != 1:
            raise UnsupportedError(
                "Multi-Output posteriors are not supported for acquisition "
                f" function of type {self.__class__}"
            )


class ExpectedImprovement(AnalyticAcquisitionFunction):
    r"""Single-outcome Expected Improvement over the current best observed value,
    computed using the analytic formula for a Normal posterior distribution. Only
    supports the case of q=1. The model must be single-outcome.

    EI(x) = E(max(y - best_f, 0)), y ~ f(x)
    """

    def __init__(
        self, model: Model, best_f: Union[float, Tensor], maximize: bool = True
    ) -> None:
        r"""Single-outcome analytic Expected Improvement.

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

    @q_batch_mode_transform
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate Expected Improvement on the candidate set X.

        Args:
            X: A `b1 x ... bk x d`-dim batched tensor of `d`-dim design points.

        Returns:
            A `b1 x ... bk`-dim tensor of Expected Improvement values at the
                given design points `X`.
        """
        self.best_f = self.best_f.to(X)
        posterior = self.model.posterior(X)
        self._validate_single_output_posterior(posterior)
        mean = posterior.mean
        # deal with batch evaluation and broadcasting
        view_shape = mean.shape[:-2] if mean.dim() >= X.dim() else X.shape[:-2]
        mean = mean.view(view_shape)
        sigma = posterior.variance.clamp_min(1e-9).sqrt().view(view_shape)
        u = (mean - self.best_f.expand_as(mean)) / sigma
        if not self.maximize:
            u = -u
        normal = Normal(torch.zeros_like(u), torch.ones_like(u))
        ucdf = normal.cdf(u)
        updf = torch.exp(normal.log_prob(u))
        ei = sigma * (updf + u * ucdf)
        return ei


class PosteriorMean(AnalyticAcquisitionFunction):
    r"""Single-outcome posterior mean. Only supports the case of q=1.
    The model must be single-outcome.
    """

    @q_batch_mode_transform
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the posterior mean on the candidate set X.

        Args:
            X: A `(b) x d`-dim Tensor of `(b)` t-batches of `d`-dim design
                points each.

        Returns:
            A `(b)`-dim Tensor of Posterior Mean values at the given
                design points `X`.
        """
        posterior = self.model.posterior(X)
        self._validate_single_output_posterior(posterior)
        return posterior.mean.view(X.shape[:-2])


class ProbabilityOfImprovement(AnalyticAcquisitionFunction):
    r"""Single-outcome Probability of Improvement over the current best
    observed value, computed using the analytic formula under
    a Normal posterior distribution. Only supports the case of q=1. The model
    must be single-outcome.

    PI(x) = P(y >= best_f), y ~ f(x)
    """

    def __init__(
        self, model: Model, best_f: Union[float, Tensor], maximize: bool = True
    ) -> None:
        r"""Single-outcome analytic Probability of Improvement.

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

    @q_batch_mode_transform
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the Probability of Improvement on the candidate set X.

        Args:
            X: A `(b) x d`-dim Tensor of `(b)` t-batches of `d`-dim design
                points each.

        Returns:
            A `(b)`-dim Tensor of Probability of Improvement values at the given
                design points `X`.
        """
        self.best_f = self.best_f.to(X)
        batch_shape = X.shape[:-2]
        posterior = self.model.posterior(X)
        self._validate_single_output_posterior(posterior)
        mean, sigma = posterior.mean, posterior.variance.sqrt()
        mean = posterior.mean.view(batch_shape)
        sigma = posterior.variance.sqrt().clamp_min(1e-9).view(batch_shape)
        u = (mean - self.best_f.expand_as(mean)) / sigma
        if not self.maximize:
            u = -u
        normal = Normal(torch.zeros_like(u), torch.ones_like(u))
        return normal.cdf(u)


class UpperConfidenceBound(AnalyticAcquisitionFunction):
    r"""Single-outcome Upper Confidence Bound, which comprises of the posterior
    mean plus a bonus term: the posterior standard deviation weighted by
    a trade-off parameter, beta. Only supports the case of q=1. The model must be
    single-outcome.

    UCB(x) = mu(x) + sqrt(beta) * sigma(x), where mu and sigma are the posterior
    mean and standard deviation, respectively.
    """

    def __init__(
        self, model: Model, beta: Union[float, Tensor], maximize: bool = True
    ) -> None:
        r"""Single-outcome Upper Confidence Bound.

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

    @q_batch_mode_transform
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the Upper Confidence Bound on the candidate set X.

        Args:
            X: A `(b) x d`-dim Tensor of `(b)` t-batches of `d`-dim design
                points each.

        Returns:
            A `(b)`-dim Tensor of Upper Confidence Bound values at the given
                design points `X`.
        """
        self.beta = self.beta.to(X)
        batch_shape = X.shape[:-2]
        posterior = self.model.posterior(X)
        self._validate_single_output_posterior(posterior)
        mean = posterior.mean.view(batch_shape)
        variance = posterior.variance.view(batch_shape)
        delta = (self.beta.expand_as(mean) * variance).sqrt()
        if self.maximize:
            return mean + delta
        else:
            return mean - delta


class ConstrainedExpectedImprovement(AnalyticAcquisitionFunction):
    r"""Constrained Expected Improvement. Computes the analytic expected improvement
    for a Normal posterior distribution, weighted by a probability of feasibility.
    The objective and constraints are assumed to be independent. Only supports the
    case of q=1. The model should be multi-outcome, with the index of the objective
    and constraints passed to the constructor.

    Constrained_EI(x) = EI(x) * Product_i P(y_i \in [lower_i, upper_i]),
    where y_i ~ constraint_i(x) and lower_i, upper_i are the lower and upper
    bounds for the i-th constraint.
    """

    def __init__(
        self,
        model: Model,
        best_f: Union[float, Tensor],
        objective_index: int,
        constraints: Dict[int, Tuple[Optional[float], Optional[float]]],
        maximize: bool = True,
    ) -> None:
        r"""Analytic Constrained Expected Improvement.

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
        self._preprocess_constraint_bounds(constraints=constraints)
        self.register_forward_pre_hook(convert_to_target_pre_hook)

    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate Constrained Expected Improvement on the candidate set X.

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
        prob_feas = self._compute_prob_feas(X=X, means=means, sigmas=sigmas)
        ei = ei.mul(prob_feas)
        return ei.squeeze(dim=-1)

    def _preprocess_constraint_bounds(
        self, constraints: Dict[int, Tuple[Optional[float], Optional[float]]]
    ) -> None:
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
        # causes an issue with autograd since we get 0 * inf. Investigate later.

        output_shape = list(X.shape)
        output_shape[-1] = 1
        prob_feas = torch.ones(output_shape, device=X.device, dtype=X.dtype)

        if len(self.constraint_lower_inds) > 0:
            normal_lower = _construct_dist(means, sigmas, self.constraint_lower_inds)
            prob_feas = prob_feas.mul(
                torch.prod(
                    1 - normal_lower.cdf(self.constraint_lower), dim=-1, keepdim=True
                )
            )
        if len(self.constraint_upper_inds) > 0:
            normal_upper = _construct_dist(means, sigmas, self.constraint_upper_inds)
            prob_feas = prob_feas.mul(
                torch.prod(
                    normal_upper.cdf(self.constraint_upper), dim=-1, keepdim=True
                )
            )
        if len(self.constraint_both_inds) > 0:
            normal_both = _construct_dist(means, sigmas, self.constraint_both_inds)
            prob_feas = prob_feas.mul(
                torch.prod(
                    normal_both.cdf(self.constraint_both[:, 1])
                    - normal_both.cdf(self.constraint_both[:, 0]),
                    dim=-1,
                    keepdim=True,
                )
            )
        return prob_feas


class NoisyExpectedImprovement(ExpectedImprovement):
    r"""Single-outcome Noisy Expected Improvement, computed by averaging over
    the ExpectedImprovement over a number of fantasy models. Only supports the
    case of q=1. The model must be single-outcome.

    `NEI(x) = E(max(y - max Y_baseline), 0)), (y, Y_baseline) ~ f((x, X_baseline))`,
    where X_baseline are previously observed points.

    Note: This acquisition function currently relies on using a GPyTorch ExactGP.
    """

    def __init__(
        self,
        model: GPyTorchModel,
        X_observed: Tensor,
        num_fantasies: int = 20,
        maximize: bool = True,
    ) -> None:
        r"""Single-outcome analytic Noisy Expected Improvement.

        Args:
            model: A fitted single-outcome model.
            X_observed: A `m x d` Tensor of observed points that are likely to
                be the best observed points so far.
            num_fantasies: The number of fantasies to generate. The higher this
                number the more accurate the model (at the expense of model
                complexity and performance).
            maximize: If True, consider the problem a maximization problem.
        """
        # construct fantasy model (batch mode)
        posterior = model.posterior(X_observed)
        self._validate_single_output_posterior(posterior=posterior)
        sampler = SobolQMCNormalSampler(num_fantasies)
        Y_fantasized = sampler(posterior).squeeze(-1)
        noise = torch.full_like(Y_fantasized, 1e-7)  # "noiseless" fantasies
        batch_X_observed = X_observed.expand(num_fantasies, *X_observed.shape)
        # The fantasy model will operate in batch mode
        fantasy_model = model.get_fantasy_model(
            batch_X_observed, Y_fantasized, noise=noise
        )
        best_f = Y_fantasized.max(dim=-1)[0]
        super().__init__(model=fantasy_model, best_f=best_f, maximize=maximize)

    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate Expected Improvement on the candidate set X.

        Args:
            X: A `b1 x ... bk x d`-dim batched tensor of `d`-dim design points.

        Returns:
            A `b1 x ... bk`-dim tensor of Noisy Expected Improvement values at
                the given design points `X`.
        """
        # add a single-element batch dimension to be broadcasted against the
        # batch dimension of the fantasy model. This will be in addition to the
        # single-element q-batch dimension added by the forward method of
        # ExpectedImprovement
        return super().forward(X.unsqueeze(-2)).mean(dim=-1)


def _construct_dist(means: Tensor, sigmas: Tensor, inds: Tensor):
    mean = means[..., inds]
    sigma = sigmas[..., inds]
    return Normal(loc=mean, scale=sigma)
