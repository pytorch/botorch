#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Acquisition functions for max-value entropy search for multi-objective
Bayesian optimization.

References

.. [Belakaria2019]
    S. Belakaria, A. Deshwal, J. R. Doppa. Max-value Entropy Search
    for Multi-Objective Bayesian Optimization. Advances in Neural
    Information Processing Systems, 32. 2019.

.. [Tu2022]
    B. Tu, A. Gandy, N. Kantas and B.Shafei. Joint Entropy Search for
    Multi-Objective Bayesian Optimization. Advances in Neural
    Information Processing Systems, 35. 2022.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch
from botorch.acquisition.max_value_entropy_search import qMaxValueEntropy
from botorch.acquisition.multi_objective.monte_carlo import (
    MultiObjectiveMCAcquisitionFunction,
)
from botorch.models.converter import (
    batched_multi_output_to_single_output,
    model_list_to_batched,
)
from botorch.models.model import Model
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.sampling.samplers import MCSampler, SobolQMCNormalSampler
from botorch.utils.transforms import (
    t_batch_mode_transform,
    concatenate_pending_points,
)
from torch import Tensor
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.multi_objective.joint_entropy_search import (
    _compute_entropy_noiseless,
    _compute_entropy_upper_bound,
    _compute_entropy_monte_carlo
)
from math import pi
CLAMP_LB = 1.0e-8

class qMultiObjectiveMaxValueEntropy(
    qMaxValueEntropy, MultiObjectiveMCAcquisitionFunction
):
    r"""The acquisition function for MESMO.

    This acquisition function computes the mutual information of
    Pareto frontier and a candidate point. See [Belakaria2019]_ for
    a detailed discussion.

    q > 1 is supported through cyclic optimization and fantasies.

    Noisy observations are support by computing information gain with
    observation noise as in Appendix C in [Takeno2020mfmves]_.

    Note: this only supports maximization.

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> MESMO = qMultiObjectiveMaxValueEntropy(model, sample_pfs)
        >>> mesmo = MESMO(test_X)
    """

    def __init__(
        self,
        model: Model,
        sample_pareto_frontiers: Callable[[Model], Tensor],
        num_fantasies: int = 16,
        X_pending: Optional[Tensor] = None,
        sampler: Optional[MCSampler] = None,
        **kwargs: Any,
    ) -> None:
        r"""Multi-objective max-value entropy search acquisition function.

        Args:
            model: A fitted multi-output model.
            sample_pareto_frontiers: A callable that takes a model and returns a
                `num_samples x n' x m`-dim tensor of outcomes to use for constructing
                `num_samples` sampled Pareto frontiers.
            num_fantasies: Number of fantasies to generate. The higher this
                number the more accurate the model (at the expense of model
                complexity, wall time and memory). Ignored if `X_pending` is `None`.
            X_pending: A `m x d`-dim Tensor of `m` design points that have been
                submitted for function evaluation but have not yet been evaluated.
        """
        MultiObjectiveMCAcquisitionFunction.__init__(self, model=model, sampler=sampler)

        # Batch GP models (e.g. fantasized models) are not currently supported
        if isinstance(model, ModelListGP):
            train_X = model.models[0].train_inputs[0]
        else:
            train_X = model.train_inputs[0]
        if train_X.ndim > 3:
            raise NotImplementedError(
                "Batch GP models (e.g. fantasized models) "
                "are not yet supported by qMultiObjectiveMaxValueEntropy"
            )
        # convert to batched MO model
        batched_mo_model = (
            model_list_to_batched(model) if isinstance(model, ModelListGP) else model
        )
        self._init_model = batched_mo_model
        self.mo_model = batched_mo_model
        self.model = batched_multi_output_to_single_output(
            batch_mo_model=batched_mo_model
        )
        self.fantasies_sampler = SobolQMCNormalSampler(num_fantasies)
        self.num_fantasies = num_fantasies
        # weight is used in _compute_information_gain
        self.maximize = True
        self.weight = 1.0
        self.sample_pareto_frontiers = sample_pareto_frontiers

        # this avoids unnecessary model conversion if X_pending is None
        if X_pending is None:
            self._sample_max_values()
        else:
            self.set_X_pending(X_pending)
        # This avoids attribute errors in qMaxValueEntropy code.
        self.posterior_transform = None

    def set_X_pending(self, X_pending: Optional[Tensor] = None) -> None:
        r"""Set pending points.

        Informs the acquisition function about pending design points,
        fantasizes the model on the pending points and draws max-value samples
        from the fantasized model posterior.

        Args:
            X_pending: `m x d` Tensor with `m` `d`-dim design points that have
                been submitted for evaluation but have not yet been evaluated.
        """
        MultiObjectiveMCAcquisitionFunction.set_X_pending(self, X_pending=X_pending)
        if X_pending is not None:
            # fantasize the model
            fantasy_model = self._init_model.fantasize(
                X=X_pending, sampler=self.fantasies_sampler, observation_noise=True
            )
            self.mo_model = fantasy_model
            # convert model to batched single outcome model.
            self.model = batched_multi_output_to_single_output(
                batch_mo_model=self.mo_model
            )
            self._sample_max_values()
        else:
            # This is mainly for setting the model to the original model
            # after the sequential optimization at q > 1
            self.mo_model = self._init_model
            self.model = batched_multi_output_to_single_output(
                batch_mo_model=self.mo_model
            )
            self._sample_max_values()

    def _sample_max_values(self) -> None:
        r"""Sample max values for MC approximation of the expectation in MES"""
        with torch.no_grad():
            # num_samples x (num_fantasies) x n_pareto_points x m
            sampled_pfs = self.sample_pareto_frontiers(self.mo_model)
            if sampled_pfs.ndim == 3:
                # add fantasy dim
                sampled_pfs = sampled_pfs.unsqueeze(-3)
            # take component-wise max value
            self.posterior_max_values = sampled_pfs.max(dim=-2).values

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        r"""Compute max-value entropy at the design points `X`.

        Args:
            X: A `batch_shape x 1 x d`-dim Tensor of `batch_shape` t-batches
                with `1` `d`-dim design points each.

        Returns:
            A `batch_shape`-dim Tensor of MVE values at the given design points `X`.
        """
        # `m` dim tensor of information gains
        # unsqueeze X to add a batch-dim for the batched model
        igs = qMaxValueEntropy.forward(self, X=X.unsqueeze(-3))
        # sum over objectives
        return igs.sum(dim=-1)


class qLowerBoundMaxValueEntropySearch(AcquisitionFunction):
    r"""The acquisition function for the Max-value Entropy Search, where the batches
    `q > 1` are supported through the lower bound formulation.

    This acquisition function computes the mutual information between the observation
    at a candidate point X and the Pareto optimal outputs.
    """

    def __init__(
        self,
        model: Model,
        pareto_fronts: Tensor,
        hypercell_bounds: Tensor,
        maximize: bool = True,
        X_pending: Optional[Tensor] = None,
        estimation_type: str = "Lower bound",
        only_diagonal: Optional[bool] = False,
        sampler: Optional[MCSampler] = None,
        num_samples: Optional[int] = 64,
        num_constraints: Optional[int] = 0,
        **kwargs: Any,
    ) -> None:
        r"""Max-value entropy search acquisition function.

        Args:
            model: A fitted batch model with 'M + K' number of outputs. The
                first `M` corresponds to the objectives and the rest corresponds
                to the constraints. The number `K` is specified the variable
                `num_constraints`.
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
                computed: "Noiseless", "Noiseless lower bound", "Lower bound" or
                "Monte Carlo".
            only_diagonal: If true we only compute the diagonal elements of the
                variance for the `Lower bound` estimation strategy.
            sampler: The sampler used if Monte Carlo is used to estimate the entropy.
                Defaults to 'SobolQMCNormalSampler(num_samples=64,
                collapse_batch_dims=True)'.
            num_samples: The number of Monte Carlo samples if using the default Monte
                Carlo sampler.
            num_constraints: The number of constraints.
        """
        super().__init__(model=model)
        self.model = model

        self.pareto_fronts = pareto_fronts
        self.num_pareto_samples = pareto_fronts.shape[0]
        self.num_pareto_points = pareto_fronts.shape[-2]

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
        r"""Compute maximum entropy search at the design points `X`.

        Args:
            X: A `batch_shape x q x d`-dim Tensor of `batch_shape` t-batches with `1`
            `d`-dim design points each.

        Returns:
            A `batch_shape`-dim Tensor of MES values at the given design points `X`.
        """
        K = self.num_constraints
        M = self.model.num_outputs - K

        # Compute the initial entropy term depending on `X`.
        posterior_plus_noise = self.model.posterior(X, observation_noise=True)

        # Additional constant term.
        add_term = .5 * (M + K) * (1 + torch.log(torch.ones(1) * 2 * pi))
        # The variance initially has shape `batch_shape x (q*(M+K)) x (q*(M+K))`
        # prior_entropy has shape `batch_shape x num_fantasies`
        prior_entropy = add_term + .5 * torch.logdet(
            posterior_plus_noise.mvn.covariance_matrix
        )

        # Compute the posterior entropy term.
        posterior = self.model.posterior(X.unsqueeze(-2), observation_noise=False)
        posterior_plus_noise = self.model.posterior(
            X.unsqueeze(-2), observation_noise=True
        )

        # `batch_shape x q x 1 x (M+K)`
        mean = posterior.mean
        var = posterior.variance.clamp_min(CLAMP_LB)
        var_plus_noise = posterior_plus_noise.variance.clamp_min(CLAMP_LB)

        # Expand shapes to `batch_shape x num_pareto_samples x q x 1 x (M + K)`
        new_shape = mean.shape[:-3] + torch.Size([self.num_pareto_samples]) + \
            mean.shape[-3:]
        mean = mean.unsqueeze(-4).expand(new_shape)
        var = var.unsqueeze(-4).expand(new_shape)
        var_plus_noise = var_plus_noise.unsqueeze(-4).expand(new_shape)

        # `batch_shape x q` dim Tensor of entropy estimates
        if self.estimation_type == "Noiseless":
            post_entropy = _compute_entropy_noiseless(
                hypercell_bounds=self.hypercell_bounds.unsqueeze(1),
                mean=mean,
                variance=var,
                variance_plus_noise=var_plus_noise,
            )

        if self.estimation_type == "Lower bound":
            post_entropy = _compute_entropy_upper_bound(
                hypercell_bounds=self.hypercell_bounds.unsqueeze(1),
                mean=mean,
                variance=var,
                variance_plus_noise=var_plus_noise,
                only_diagonal=self.only_diagonal
            )

        if self.estimation_type == "Monte Carlo":
            # `num_mc_samples x batch_shape x q x 1 x (M+K)`
            samples = self.sampler(posterior_plus_noise)

            # `num_mc_samples x batch_shape x q`
            if (M + K) == 1:
                samples_log_prob = posterior_plus_noise.mvn.log_prob(
                    samples.squeeze(-1)
                )
            else:
                samples_log_prob = posterior_plus_noise.mvn.log_prob(
                    samples
                )

            # Expand shape to `num_mc_samples x batch_shape x num_pareto_samples x
            # q x 1 x (M+K)`
            new_shape = samples.shape[:-3] \
                + torch.Size([self.num_pareto_samples]) + samples.shape[-3:]
            samples = samples.unsqueeze(-4).expand(new_shape)

            # Expand shape to `num_mc_samples x batch_shape x num_pareto_samples x q`
            new_shape = samples_log_prob.shape[:-1] \
                + torch.Size([self.num_pareto_samples]) + samples_log_prob.shape[-1:]
            samples_log_prob = samples_log_prob.unsqueeze(-2).expand(new_shape)

            post_entropy = _compute_entropy_monte_carlo(
                hypercell_bounds=self.hypercell_bounds.unsqueeze(1),
                mean=mean,
                variance=var,
                variance_plus_noise=var_plus_noise,
                samples=samples,
                samples_log_prob=samples_log_prob
            )

        # Sum over the batch.
        return prior_entropy - post_entropy.sum(dim=-1)
