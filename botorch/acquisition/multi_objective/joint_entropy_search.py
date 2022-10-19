#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Acquisition functions for joint entropy search for multi-objective Bayesian
optimization (JES).

References:

.. [Tu2022]
    B. Tu, A. Gandy, N. Kantas and B.Shafei. Joint Entropy Search for Multi-Objective
    Bayesian Optimization. Advances in Neural Information Processing Systems, 35.
    2022.

"""
from __future__ import annotations

from typing import Any, Optional
import torch
from torch import Tensor

from botorch.models.model import Model
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.sampling.samplers import MCSampler, SobolQMCNormalSampler
from botorch.utils.transforms import (
    concatenate_pending_points,
    t_batch_mode_transform,
)

from botorch.models.utils import fantasize as fantasize_flag
from botorch import settings
from torch.distributions import Normal

from math import pi

CLAMP_LB = 1.0e-8


class qLowerBoundJointEntropySearch(AcquisitionFunction):
    r"""The acquisition function for the Joint Entropy Search, where the batches
    `q > 1` are supported through the lower bound formulation.

    This acquisition function computes the mutual information between the observation
    at a candidate point X and the Pareto optimal input-output pairs.
    """

    def __init__(
        self,
        model: Model,
        pareto_sets: Tensor,
        pareto_fronts: Tensor,
        hypercell_bounds: Tensor,
        maximize: bool = True,
        X_pending: Optional[Tensor] = None,
        estimation_type: str = "Lower bound",
        sampling_noise: Optional[bool] = True,
        only_diagonal: Optional[bool] = False,
        sampler: Optional[MCSampler] = None,
        num_samples: Optional[int] = 64,
        num_constraints: Optional[int] = 0,
        **kwargs: Any,
    ) -> None:
        r"""Joint entropy search acquisition function.

        Args:
            model: A fitted batch model with 'M + K' number of outputs. The
                first `M` corresponds to the objectives and the rest corresponds
                to the constraints. An input is feasible if f_k(x) <= 0 for the
                constraint functions f_k. The number `K` is specified the variable
                `num_constraints`.
            pareto_sets: A `num_pareto_samples x num_pareto_points x d`-dim Tensor.
            pareto_fronts: A `num_pareto_samples x num_pareto_points x M`-dim Tensor.
            hypercell_bounds:  A `num_pareto_samples x 2 x J x (M + K)`-dim
                Tensor containing the hyper-rectangle bounds for integration. In the
                unconstrained case, this gives the partition of the dominated space.
                In the constrained case, this gives the partition of the feasible
                dominated space union the infeasible space.
            maximize: If True, consider the problem a maximization problem.
            X_pending: A `m x d`-dim Tensor of `m` design points that have been
                submitted for function evaluation but have not yet been evaluated.
            estimation_type: A string to determine which entropy estimate is
                computed: "Noiseless", "Lower bound" or "Monte Carlo".
            only_diagonal: If true we only compute the diagonal elements of the
                variance for the `Lower bound` estimation strategy i.e. the LB2
                strategy.
            sampler: The sampler used if Monte Carlo is used to estimate the entropy.
                Defaults to 'SobolQMCNormalSampler(num_samples=64,
                collapse_batch_dims=True)'.
            num_samples: The number of Monte Carlo samples if using the default Monte
                Carlo sampler.
            num_constraints: The number of constraints.
        """
        super().__init__(model=model)
        self.prior_model = model
        self.pareto_sets = pareto_sets
        self.pareto_fronts = pareto_fronts

        self.num_pareto_samples = pareto_fronts.shape[0]
        self.num_pareto_points = pareto_fronts.shape[-2]

        self.sampling_noise = sampling_noise

        # Condition the model on the sampled pareto optimal points.
        # TODO: Apparently, we need to make a call to the posterior otherwise
        #  we run into a gpytorch runtime error:
        #  "Fantasy observations can only be added after making predictions with a
        #  model so that all test independent caches exist."
        with fantasize_flag():
            with settings.propagate_grads(False):
                post_ps = self.prior_model.posterior(
                    self.pareto_sets, observation_noise=False
                )
            # Condition with observation noise.
            self.posterior_model = self.prior_model.condition_on_observations(
                X=self.prior_model.transform_inputs(self.pareto_sets),
                Y=self.pareto_fronts,
            )

        self.hypercell_bounds = hypercell_bounds
        self.maximize = maximize
        self.weight = 1.0 if maximize else -1.0

        self.estimation_type = estimation_type
        estimation_types = [
            "Noiseless",
            "Lower bound",
            "Monte Carlo"
        ]

        if estimation_type not in estimation_types:
            raise NotImplementedError(
                "Currently the only supported estimation type are: "
                + " ".joint(estimation_types) + "."
            )

        self.only_diagonal = only_diagonal
        if sampler is None:
            self.sampler = SobolQMCNormalSampler(
                num_samples=num_samples, collapse_batch_dims=True
            )
        else:
            self.sampler = sampler

        self.num_constraints = num_constraints
        self.set_X_pending(X_pending)

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Compute joint entropy search at the design points `X`.

        Args:
            X: A `batch_shape x q x d`-dim Tensor of `batch_shape` t-batches with `1`
                `d`-dim design points each.

        Returns:
            A `batch_shape`-dim Tensor of JES values at the given design points `X`.
        """
        K = self.num_constraints
        M = self.prior_model.num_outputs - K

        # Compute the prior entropy term depending on `X`.
        prior_posterior_plus_noise = self.prior_model.posterior(
            X, observation_noise=True
        )

        # Additional constant term.
        add_term = .5 * (M + K) * (1 + torch.log(torch.ones(1) * 2 * pi))
        # The variance initially has shape `batch_shape x (q*(M+K)) x (q*(M+K))`
        # prior_entropy has shape `batch_shape`.
        prior_entropy = add_term + .5 * torch.logdet(
            prior_posterior_plus_noise.mvn.covariance_matrix
        )

        # Compute the posterior entropy term.
        post_posterior = self.posterior_model.posterior(
            X.unsqueeze(-2).unsqueeze(-3), observation_noise=False
        )
        post_posterior_plus_noise = self.posterior_model.posterior(
            X.unsqueeze(-2).unsqueeze(-3), observation_noise=True
        )
        # `batch_shape x num_pareto_samples x q x 1 x (M+K)`
        post_mean = post_posterior.mean.swapaxes(-4, -3)
        post_var = post_posterior.variance.clamp_min(CLAMP_LB).swapaxes(-4, -3)

        post_var_plus_noise = post_posterior_plus_noise.variance.clamp_min(
            CLAMP_LB
        ).swapaxes(-4, -3)

        # `batch_shape x q` dim Tensor of entropy estimates
        if self.estimation_type == "Noiseless":
            post_entropy = _compute_entropy_noiseless(
                hypercell_bounds=self.hypercell_bounds.unsqueeze(1),
                mean=post_mean,
                variance=post_var,
                variance_plus_noise=post_var_plus_noise,
            )

        if self.estimation_type == "Lower bound":
            post_entropy = _compute_entropy_upper_bound(
                hypercell_bounds=self.hypercell_bounds.unsqueeze(1),
                mean=post_mean,
                variance=post_var,
                variance_plus_noise=post_var_plus_noise,
                only_diagonal=self.only_diagonal
            )
        if self.estimation_type == "Monte Carlo":
            # `num_mc_samples x batch_shape x q x num_pareto_samples x 1 x (M+K)`
            samples = self.sampler(post_posterior_plus_noise)

            # `num_mc_samples x batch_shape x q x num_pareto_samples`
            if (M + K) == 1:
                samples_log_prob = post_posterior_plus_noise.mvn.log_prob(
                    samples.squeeze(-1)
                )
            else:
                samples_log_prob = post_posterior_plus_noise.mvn.log_prob(
                    samples
                )

            # Swap axes to get:
            # samples shape `num_mc_samples x batch_shape x num_pareto_samples x q
            # x 1 x (M+K)`
            # log prob shape `num_mc_samples x batch_shape x num_pareto_samples x q`
            post_entropy = _compute_entropy_monte_carlo(
                hypercell_bounds=self.hypercell_bounds.unsqueeze(1),
                mean=post_mean,
                variance=post_var,
                variance_plus_noise=post_var_plus_noise,
                samples=samples.swapaxes(-4, -3),
                samples_log_prob=samples_log_prob.swapaxes(-2, -1)
            )

        # Sum over the batch.
        return prior_entropy - post_entropy.sum(dim=-1)


def _compute_entropy_noiseless(
        hypercell_bounds: Tensor,
        mean: Tensor,
        variance: Tensor,
        variance_plus_noise: Tensor,
) -> Tensor:
    r"""Computes the entropy estimate at the design points `X` assuming noiseless
    observations. This is used for the JES-0 and MES-0 estimate.

    `num_fantasies = 1` for non-fantasized models.

    Args:
        hypercell_bounds: A `num_pareto_samples x num_fantasies x 2 x J x (M + K)`
            -dim Tensor containing the box decomposition bounds, where
            `J = max(num_boxes)`.
        mean: A `batch_shape x num_pareto_samples x num_fantasies x 1 x (M + K)`-dim
            Tensor containing the posterior mean at X.
        variance: A `batch_shape x num_pareto_samples x num_fantasies x 1 x (M + K)`
            -dim Tensor containing the posterior variance at X excluding observation
            noise.
        variance_plus_noise: A `batch_shape x num_pareto_samples x num_fantasies x 1
            x (M + K)`-dim Tensor containing the posterior variance at X including
            observation noise.

    Returns:
        A `batch_shape x num_fantasies`-dim Tensor of entropy estimate at the given
        design points `X` (`num_fantasies=1` for non-fantasized models).
    """
    # Standardize the box decomposition bounds and compute normal quantities.
    # `batch_shape x num_pareto_samples x num_fantasies x 2 x J x (M + K)`
    g = (hypercell_bounds - mean.unsqueeze(-2)) / torch.sqrt(variance.unsqueeze(-2))
    normal = Normal(torch.zeros_like(g), torch.ones_like(g))
    gcdf = normal.cdf(g)
    gpdf = torch.exp(normal.log_prob(g))
    g_times_gpdf = g * gpdf

    # Compute the differences between the upper and lower terms.
    Wjm = (gcdf[..., 1, :, :] - gcdf[..., 0, :, :]).clamp_min(CLAMP_LB)
    Vjm = g_times_gpdf[..., 1, :, :] - g_times_gpdf[..., 0, :, :]

    # Compute W.
    Wj = torch.exp(torch.sum(torch.log(Wjm), dim=-1, keepdims=True))
    W = torch.sum(Wj, dim=-2, keepdims=True).clamp_max(1.0)

    # Compute the sum of ratios.
    ratios = .5 * (Wj * (Vjm / Wjm)) / W
    # `batch_shape x num_pareto_samples x num_fantasies x 1 x 1`
    ratio_term = torch.sum(ratios, dim=(-2, -1), keepdims=True)

    # Compute the logarithm of the variance.
    log_term = .5 * torch.log(variance_plus_noise).sum(-1, keepdims=True)

    # `batch_shape x num_pareto_samples x num_fantasies x 1 x 1`
    log_term = log_term + torch.log(W)

    # Additional constant term.
    M_plus_K = mean.shape[-1]
    add_term = .5 * M_plus_K * (1 + torch.log(torch.ones(1) * 2 * pi))

    # `batch_shape x num_pareto_samples x num_fantasies`
    entropy = add_term + (log_term - ratio_term).squeeze(-1).squeeze(-1)

    return entropy.mean(-2)


def _compute_entropy_upper_bound(
        hypercell_bounds: Tensor,
        mean: Tensor,
        variance: Tensor,
        variance_plus_noise: Tensor,
        only_diagonal: Optional[bool] = False,
) -> Tensor:
    r"""Computes the entropy upper bound at the design points `X`. This is used for
    the JES-LB and MES-LB estimate. If `only_diagonal` is True, then this computes
    the entropy estimate for the JES-LB2 and MES-LB2.

    `num_fantasies = 1` for non-fantasized models.

    Args:
        hypercell_bounds: A `num_pareto_samples x num_fantasies x 2 x J x (M + K)`
            -dim Tensor containing the box decomposition bounds, where
            `J` = max(num_boxes).
        mean: A `batch_shape x num_pareto_samples x num_fantasies x 1 x (M + K)`-dim
            Tensor containing the posterior mean at X.
        variance: A `batch_shape x num_pareto_samples x num_fantasies x 1 x (M + K)`
            -dim Tensor containing the posterior variance at X excluding observation
            noise.
        variance_plus_noise: A `batch_shape x num_pareto_samples x num_fantasies
            x 1 x (M + K)`-dim Tensor containing the posterior variance at X
            including observation noise.
        only_diagonal: If true we only compute the diagonal elements of the variance.

    Returns:
        A `batch_shape x num_fantasies`-dim Tensor of entropy estimate at the
        given design points `X` (`num_fantasies=1` for non-fantasized models).
    """
    # Standardize the box decomposition bounds and compute normal quantities.
    # `batch_shape x num_pareto_samples x num_fantasies x 2 x J x (M + K)`
    g = (hypercell_bounds - mean.unsqueeze(-2)) / torch.sqrt(variance.unsqueeze(-2))
    normal = Normal(torch.zeros_like(g), torch.ones_like(g))
    gcdf = normal.cdf(g)
    gpdf = torch.exp(normal.log_prob(g))
    g_times_gpdf = g * gpdf

    # Compute the differences between the upper and lower terms.
    Wjm = (gcdf[..., 1, :, :] - gcdf[..., 0, :, :]).clamp_min(CLAMP_LB)
    Vjm = g_times_gpdf[..., 1, :, :] - g_times_gpdf[..., 0, :, :]
    Gjm = gpdf[..., 1, :, :] - gpdf[..., 0, :, :]

    # Compute W.
    Wj = torch.exp(torch.sum(torch.log(Wjm), dim=-1, keepdims=True))
    W = torch.sum(Wj, dim=-2, keepdims=True).clamp_max(1.0)

    Cjm = Gjm / Wjm

    # First moment:
    # `batch_shape x num_pareto_samples x num_fantasies x 1 x (M+K)
    mom1 = mean - torch.sqrt(variance) * (Cjm * Wj / W).sum(-2, keepdims=True)
    # diagonal weighted sum
    # `batch_shape x num_pareto_samples x num_fantasies x 1 x (M+K)
    diag_weighted_sum = (Wj * variance * Vjm / Wjm / W).sum(-2, keepdims=True)

    if only_diagonal:
        # `batch_shape x num_pareto_samples x num_fantasies x 1 x (M + K)`
        mean_squared = mean * mean
        cross_sum = - 2 * (
                mean * torch.sqrt(variance) * Cjm * Wj / W
        ).sum(-2, keepdims=True)
        # `batch_shape x num_pareto_samples x num_fantasies x 1 x (M + K)`
        mom2 = variance_plus_noise - diag_weighted_sum + cross_sum + mean_squared
        var = (mom2 - mom1 * mom1).clamp_min(CLAMP_LB)

        # `batch_shape x num_pareto_samples x num_fantasies
        log_det_term = .5 * torch.log(var).sum(dim=-1).squeeze(-1)
    else:
        # First moment x First moment
        # `batch_shape x num_pareto_samples x num_fantasies x 1 x (M+K) x (M+K)
        cross_mom1 = torch.einsum('...i,...j->...ij', mom1, mom1)

        # Second moment:
        # `batch_shape x num_pareto_samples x num_fantasies x 1 x (M+K) x (M+K)
        # firstly compute the general terms
        mom2_cross1 = - torch.einsum(
            '...i,...j->...ij', mean, torch.sqrt(variance) * Cjm
        )
        mom2_cross2 = - torch.einsum(
            '...i,...j->...ji', mean, torch.sqrt(variance) * Cjm
        )
        mom2_mean_squared = torch.einsum('...i,...j->...ij', mean, mean)

        mom2_weighted_sum = (
            (mom2_cross1 + mom2_cross2) * Wj.unsqueeze(-1) / W.unsqueeze(-1)
        ).sum(-3, keepdims=True)
        mom2_weighted_sum = mom2_weighted_sum + mom2_mean_squared

        # Compute the additional off-diagonal terms.
        mom2_off_diag = torch.einsum(
            '...i,...j->...ij', torch.sqrt(variance) * Cjm, torch.sqrt(variance) * Cjm
        )
        mom2_off_diag_sum = (
                mom2_off_diag * Wj.unsqueeze(-1) / W.unsqueeze(-1)
        ).sum(-3, keepdims=True)

        # Compute the diagonal terms and subtract the diagonal computed before.
        init_diag = torch.diagonal(mom2_off_diag_sum, dim1=-2, dim2=-1)
        diag_weighted_sum = torch.diag_embed(
            variance_plus_noise - diag_weighted_sum - init_diag
        )
        mom2 = mom2_weighted_sum + mom2_off_diag_sum + diag_weighted_sum
        # Compute the variance
        var = (mom2 - cross_mom1).squeeze(-3)

        # Jitter the diagonal.
        # The jitter is probably not needed here at all.
        jitter_diag = 1e-6 * torch.diag_embed(torch.ones(var.shape[:-1]))
        log_det_term = .5 * torch.logdet(var + jitter_diag)

    # Additional terms.
    M_plus_K = mean.shape[-1]
    add_term = .5 * M_plus_K * (1 + torch.log(torch.ones(1) * 2 * pi))

    # `batch_shape x num_pareto_samples x num_fantasies
    entropy = add_term + log_det_term
    return entropy.mean(-2)


def _compute_entropy_monte_carlo(
        hypercell_bounds: Tensor,
        mean: Tensor,
        variance: Tensor,
        variance_plus_noise: Tensor,
        samples: Tensor,
        samples_log_prob: Tensor,
) -> Tensor:
    r"""Computes the Monte Carlo entropy at the design points `X`.  This is used for
    the JES-MC and MES-MC estimate.

    `num_fantasies = 1` for non-fantasized models.

    Args:
        hypercell_bounds: A `num_pareto_samples x num_fantasies x 2 x J x (M+K)`-dim
            Tensor containing the box decomposition bounds, where
            `J` = max(num_boxes).
        mean: A `batch_shape x num_pareto_samples x num_fantasies x 1 x (M+K)`-dim
            Tensor containing the posterior mean at X.
        variance: A `batch_shape x num_pareto_samples x num_fantasies x 1 x (M+K)`
            -dim Tensor containing the posterior variance at X excluding observation
            noise.
        variance_plus_noise: A `batch_shape x num_pareto_samples x num_fantasies x 1
            x (M+K)`-dim Tensor containing the posterior variance at X including
            observation noise.
        samples: A `num_mc_samples x batch_shape x num_pareto_samples x num_fantasies
            x 1 x (M+K)`-dim Tensor containing the noisy samples at `X` from the
            posterior conditioned on the Pareto optimal points.
        samples_log_prob:  A `num_mc_samples x batch_shape x num_pareto_samples
            num_fantasies`-dim Tensor containing the log probability densities
            of the samples.

    Returns:
        A `batch_shape x num_fantasies`-dim Tensor of entropy estimate at the given
        design points `X` (`num_fantasies=1` for non-fantasized models).
    """
    ####################################################################
    # Standardize the box decomposition bounds and compute normal quantities.
    # `batch_shape x num_pareto_samples x num_fantasies x 2 x J x (M+K)`
    g = (hypercell_bounds - mean.unsqueeze(-2)) / torch.sqrt(variance.unsqueeze(-2))
    # `batch_shape x num_pareto_samples x num_fantasies x 1 x (M+K)`
    rho = torch.sqrt(variance / variance_plus_noise)

    # Compute the initial normal quantities.
    normal = Normal(torch.zeros_like(g), torch.ones_like(g))
    gcdf = normal.cdf(g)

    # Compute the differences between the upper and lower terms.
    Wjm = (gcdf[..., 1, :, :] - gcdf[..., 0, :, :]).clamp_min(CLAMP_LB)

    # Compute W.
    Wj = torch.exp(torch.sum(torch.log(Wjm), dim=-1, keepdims=True))
    # `batch_shape x num_pareto_samples x num_fantasies x 1 x 1`
    W = torch.sum(Wj, dim=-2, keepdims=True).clamp_max(1.0)

    ####################################################################
    g = g.unsqueeze(0)
    rho = rho.unsqueeze(0).unsqueeze(-2)
    # `num_mc_samples x batch_shape x num_pareto_samples x num_fantasies x 1 x 1 x
    # (M+K)`
    z = ((samples - mean) / torch.sqrt(variance_plus_noise)).unsqueeze(-2)
    # `num_mc_samples x batch_shape x num_pareto_samples x num_fantasies x 2 x J x
    # (M+K)`
    # Clamping here is important because `1 - rho^2 = 0` at an input where
    # observation noise is zero.
    g_new = (g - rho * z) / torch.sqrt((1 - rho * rho).clamp_min(CLAMP_LB))

    # Compute the initial normal quantities.
    normal_new = Normal(torch.zeros_like(g_new), torch.ones_like(g_new))
    gcdf_new = normal_new.cdf(g_new)

    # Compute the differences between the upper and lower terms.
    Wjm_new = (gcdf_new[..., 1, :, :] - gcdf_new[..., 0, :, :]).clamp_min(CLAMP_LB)

    # Compute W+.
    Wj_new = torch.exp(torch.sum(torch.log(Wjm_new), dim=-1, keepdims=True))
    # `num_mc_samples x batch_shape x num_pareto_samples x num_fantasies x 1 x 1`
    W_new = torch.sum(Wj_new, dim=-2, keepdims=True).clamp_max(1.0)

    ####################################################################
    # W_ratio = W+ / W
    W_ratio = torch.exp(torch.log(W_new) - torch.log(W).unsqueeze(0))
    samples_log_prob = samples_log_prob.unsqueeze(-1).unsqueeze(-1)

    # Compute the Monte Carlo average: - E[W_ratio * log(W+ p(y))] + log(W)
    log_term = torch.log(W_new) + samples_log_prob
    mc_estimate = - (W_ratio * log_term).mean(0)
    # `batch_shape x num_pareto_samples x num_fantasies
    entropy = (mc_estimate + torch.log(W)).squeeze(-1).squeeze(-1)

    # An alternative Monte Carlo estimate: - E[W_ratio * log(W_ratio p(y))]
    # log_term = torch.log(W_ratio) + samples_log_prob
    # mc_estimate = - (W_ratio * log_term).mean(0)
    # # `batch_shape x num_pareto_samples x num_fantasies
    # entropy = mc_estimate.squeeze(-1).squeeze(-1)

    return entropy.mean(-2)
