#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Acquisition functions for max-value entropy search for multi-objective
Bayesian optimization (MESMO).

References

.. [Belakaria2019]
    S. Belakaria, A. Deshwal, J. R. Doppa. Max-value Entropy Search
    for Multi-Objective Bayesian Optimization. Advances in Neural
    Information Processing Systems, 32. 2019.

"""

from __future__ import annotations

from collections.abc import Callable

from math import pi

import torch
from botorch.acquisition.max_value_entropy_search import qMaxValueEntropy
from botorch.acquisition.multi_objective.base import MultiObjectiveMCAcquisitionFunction
from botorch.acquisition.multi_objective.joint_entropy_search import (
    LowerBoundMultiObjectiveEntropySearch,
)
from botorch.models.converter import (
    batched_multi_output_to_single_output,
    model_list_to_batched,
)
from botorch.models.model import Model
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.sampling.base import MCSampler
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.transforms import concatenate_pending_points, t_batch_mode_transform
from torch import Tensor


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

    Attributes:
        _default_sample_shape: The `sample_shape` for the default sampler.

    Example:
        >>> model = SingleTaskGP(train_X, train_Y, outcome_transform=None)
        >>> MESMO = qMultiObjectiveMaxValueEntropy(model, sample_pfs)
        >>> mesmo = MESMO(test_X)
    """

    _default_sample_shape = torch.Size([128])

    def __init__(
        self,
        model: Model,
        sample_pareto_frontiers: Callable[[Model], Tensor],
        num_fantasies: int = 16,
        X_pending: Tensor | None = None,
        sampler: MCSampler | None = None,
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
                submitted for function evaluation, but have not yet been evaluated.
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
        self.fantasies_sampler = SobolQMCNormalSampler(
            sample_shape=torch.Size([num_fantasies])
        )
        self.num_fantasies = num_fantasies
        # weight is used in _compute_information_gain
        self.maximize = True
        self.weight = 1.0
        self.sample_pareto_frontiers = sample_pareto_frontiers
        # Set X_pending, register converted model and sample max values.
        self.set_X_pending(X_pending)
        # This avoids attribute errors in qMaxValueEntropy code.
        self.posterior_transform = None

    def set_X_pending(self, X_pending: Tensor | None = None) -> None:
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
                X=X_pending,
                sampler=self.fantasies_sampler,
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
        """Sample max values for MC approximation of the expectation in MES.

        Sets self.posterior_max_values."""
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


class qLowerBoundMultiObjectiveMaxValueEntropySearch(
    LowerBoundMultiObjectiveEntropySearch
):
    r"""The acquisition function for the multi-objective Max-value Entropy Search,
    where the batches `q > 1` are supported through the lower bound formulation.

    This acquisition function computes the mutual information between the observation
    at a candidate point `X` and the Pareto optimal outputs.

    See [Tu2022]_ for a discussion on the estimation procedure.

    NOTES:
    (i) The estimated acquisition value could be negative.

    (ii) The lower bound batch acquisition function might not be monotone in the
    sense that adding more elements to the batch does not necessarily increase the
    acquisition value. Specifically, the acquisition value can become smaller when
    more inputs are added.
    """

    def __init__(
        self,
        model: Model,
        hypercell_bounds: Tensor,
        X_pending: Tensor | None = None,
        estimation_type: str = "LB",
        num_samples: int = 64,
    ) -> None:
        r"""Lower bound multi-objective max-value entropy search acquisition function.

        Args:
            model: A fitted batch model with 'M' number of outputs.
            hypercell_bounds:  A `num_pareto_samples x 2 x J x M`-dim Tensor
                containing the hyper-rectangle bounds for integration, where `J` is
                the number of hyper-rectangles. In the unconstrained case, this gives
                the partition of the dominated space. In the constrained case, this
                gives the partition of the feasible dominated space union the
                infeasible space.
            X_pending: A `m x d`-dim Tensor of `m` design points that have been
                submitted for function evaluation, but have not yet been evaluated.
            estimation_type: A string to determine which entropy estimate is
                computed: "0", "LB", "LB2", or "MC".
            num_samples: The number of Monte Carlo samples for the Monte Carlo
                estimate.
        """
        super().__init__(
            model=model,
            pareto_sets=None,
            pareto_fronts=None,
            hypercell_bounds=hypercell_bounds,
            X_pending=X_pending,
            estimation_type=estimation_type,
            num_samples=num_samples,
        )

    def _compute_posterior_statistics(
        self, X: Tensor
    ) -> dict[str, GPyTorchPosterior | Tensor]:
        r"""Compute the posterior statistics.

        Args:
            X: A `batch_shape x q x d`-dim Tensor of inputs.

        Returns:
            A dictionary containing the posterior variables used to estimate the
            entropy.

            - "initial_entropy": A `batch_shape`-dim Tensor containing the entropy of
                the Gaussian random variable `p(Y| X, D_n)`.
            - "posterior_mean": A `batch_shape x num_pareto_samples x q x 1 x M`-dim
                Tensor containing the posterior mean at the input `X`.
            - "posterior_variance": A `batch_shape x num_pareto_samples x q x 1 x
                M`-dim Tensor containing the posterior variance at the input `X`
                excluding the observation noise.
            - "observation_noise": A `batch_shape x num_pareto_samples x q x 1 x M`
                -dim Tensor containing the observation noise at the input `X`.
            - "posterior_with_noise": The posterior distribution at `X` which
                includes the observation noise. This is used to compute the marginal
                log-probabilities with respect to `p(y| x, D_n)` for `x` in `X`.
        """
        tkwargs = {"dtype": X.dtype, "device": X.device}
        CLAMP_LB = torch.finfo(tkwargs["dtype"]).eps

        # Compute the initial entropy term depending on `X`.
        # TODO: Below we compute posterior_plus_noise twice:
        #  (1) Firstly, we compute p(Y| X, D_n) when computing the initial entropy
        #  (2) Secondly, we compute p(y| x, D_n) for x in X in order to compute
        #  log(p(y|x, D_n)) for x in X in the Monte Carlo estimate..
        #  This could be simplified if we could evaluate log(p(y|x, D_n)) using the
        #  the posterior p(Y| X, D_n)
        posterior_plus_noise = self.initial_model.posterior(X, observation_noise=True)

        # Additional constant term.
        add_term = (
            0.5
            * self.model.num_outputs
            * (1 + torch.log(2 * pi * torch.ones(1, **tkwargs)))
        )
        # The variance initially has shape `batch_shape x (q*M) x (q*M)`
        # prior_entropy has shape `batch_shape x num_fantasies`
        initial_entropy = add_term + 0.5 * torch.logdet(
            posterior_plus_noise.mvn.covariance_matrix
        )
        posterior_statistics = {"initial_entropy": initial_entropy}

        # Compute the posterior entropy term.
        posterior_plus_noise = self.model.posterior(
            X.unsqueeze(-2), observation_noise=True
        )

        # `batch_shape x q x 1 x M`
        mean = posterior_plus_noise.mean
        var_plus_noise = posterior_plus_noise.variance.clamp_min(CLAMP_LB)
        # Expand shapes to `batch_shape x num_pareto_samples x q x 1 x M`
        new_shape = (
            mean.shape[:-3] + torch.Size([self.num_pareto_samples]) + mean.shape[-3:]
        )
        mean = mean.unsqueeze(-4).expand(new_shape)
        var_plus_noise = var_plus_noise.unsqueeze(-4).expand(new_shape)

        # TODO: This computes the observation noise via a second evaluation of the
        #   posterior. This step could be done better.
        posterior = self.model.posterior(X.unsqueeze(-2), observation_noise=False)
        var = posterior.variance.clamp_min(CLAMP_LB)
        var = var.unsqueeze(-4).expand(new_shape)
        obs_noise = var_plus_noise - var

        posterior_statistics["posterior_mean"] = mean
        posterior_statistics["posterior_variance"] = var
        posterior_statistics["observation_noise"] = obs_noise
        posterior_statistics["posterior_with_noise"] = posterior_plus_noise

        return posterior_statistics

    def _compute_monte_carlo_variables(
        self, posterior: GPyTorchPosterior
    ) -> tuple[Tensor, Tensor]:
        r"""Compute the samples and log-probability associated with a posterior
        distribution.

        Args:
            posterior: The posterior distribution, which includes the observation
                noise.

        Returns:
            A two-element tuple containing

            - samples: A `num_mc_samples x batch_shape x num_pareto_samples x q x 1
                x M`-dim Tensor containing the Monte Carlo samples.
            - samples_log_prob: A `num_mc_samples x batch_shape x num_pareto_samples
                x q`-dim Tensor containing the log-probabilities of the Monte Carlo
                samples.
        """

        # `num_mc_samples x batch_shape x q x 1 x M`
        samples = self.get_posterior_samples(posterior)

        # `num_mc_samples x batch_shape x q`
        if self.model.num_outputs == 1:
            samples_log_prob = posterior.mvn.log_prob(samples.squeeze(-1))
        else:
            samples_log_prob = posterior.mvn.log_prob(samples)

        # Expand shape to `num_mc_samples x batch_shape x num_pareto_samples x
        # q x 1 x M`
        new_shape = (
            samples.shape[:-3]
            + torch.Size([self.num_pareto_samples])
            + samples.shape[-3:]
        )
        samples = samples.unsqueeze(-4).expand(new_shape)

        # Expand shape to `num_mc_samples x batch_shape x num_pareto_samples x q`
        new_shape = (
            samples_log_prob.shape[:-1]
            + torch.Size([self.num_pareto_samples])
            + samples_log_prob.shape[-1:]
        )
        samples_log_prob = samples_log_prob.unsqueeze(-2).expand(new_shape)

        return samples, samples_log_prob

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluates qLowerBoundMultiObjectiveMaxValueEntropySearch at the design
        points `X`.

        Args:
            X: A `batch_shape x q x d`-dim Tensor of `batch_shape` t-batches with `q`
            `d`-dim design points each.

        Returns:
            A `batch_shape`-dim Tensor of acquisition values at the given design
            points `X`.
        """
        return self._compute_lower_bound_information_gain(X)
