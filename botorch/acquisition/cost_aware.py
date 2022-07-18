#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Cost functions for cost-aware acquisition functions, e.g. multi-fidelity KG.
To be used in a context where there is an objective/cost tradeoff.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

import torch
from botorch import settings
from botorch.acquisition.objective import IdentityMCObjective, MCAcquisitionObjective
from botorch.exceptions.warnings import CostAwareWarning
from botorch.models.model import Model
from botorch.sampling.samplers import MCSampler
from torch import Tensor
from torch.nn import Module


class CostAwareUtility(Module, ABC):
    r"""
    Abstract base class for cost-aware utilities.

    :meta private:
    """

    @abstractmethod
    def forward(self, X: Tensor, deltas: Tensor, **kwargs: Any) -> Tensor:
        r"""Evaluate the cost-aware utility on the candidates and improvements.

        Args:
            X: A `batch_shape x q x d`-dim Tensor of with `q` `d`-dim design
                points each for each t-batch.
            deltas: A `num_fantasies x batch_shape`-dim Tensor of `num_fantasy`
                samples from the marginal improvement in utility over the
                current state at `X` for each t-batch.

        Returns:
            A `num_fantasies x batch_shape`-dim Tensor of cost-transformed utilities.
        """
        pass  # pragma: no cover


class GenericCostAwareUtility(CostAwareUtility):
    r"""Generic cost-aware utility wrapping a callable."""

    def __init__(self, cost: Callable[[Tensor, Tensor], Tensor]) -> None:
        r"""Generic cost-aware utility wrapping a callable.

        Args:
            cost: A callable mapping a `batch_shape x q x d'`-dim candidate set
                to a `batch_shape`-dim tensor of costs
        """
        super().__init__()
        self._cost_callable: Callable[[Tensor, Tensor], Tensor] = cost

    def forward(self, X: Tensor, deltas: Tensor, **kwargs: Any) -> Tensor:
        r"""Evaluate the cost function on the candidates and improvements.

        Args:
            X: A `batch_shape x q x d'`-dim Tensor of with `q` `d`-dim design
                points for each t-batch.
            deltas: A `num_fantasies x batch_shape`-dim Tensor of `num_fantasy`
                samples from the marginal improvement in utility over the
                current state at `X` for each t-batch.

        Returns:
            A `num_fantasies x batch_shape`-dim Tensor of cost-weighted utilities.
        """
        return self._cost_callable(X, deltas)


class InverseCostWeightedUtility(CostAwareUtility):
    r"""A cost-aware utility using inverse cost weighting based on a model.

    Computes the cost-aware utility by inverse-weighting samples
    `U = (u_1, ..., u_N)` of the increase in utility. If `use_mean=True`, this
    uses the posterior mean `mean_cost` of the cost model, i.e.
    `weighted utility = mean(U) / mean_cost`. If `use_mean=False`, it uses
    samples `C = (c_1, ..., c_N)` from the posterior of the cost model and
    performs the inverse weighting on the sample level:
    `weighted utility = mean(u_1 / c_1, ..., u_N / c_N)`.

    The cost is additive across multiple elements of a q-batch.
    """

    def __init__(
        self,
        cost_model: Model,
        use_mean: bool = True,
        cost_objective: Optional[MCAcquisitionObjective] = None,
        min_cost: float = 1e-2,
    ) -> None:
        r"""Cost-aware utility that weights increase in utiltiy by inverse cost.

        Args:
            cost_model: A Model modeling the cost of evaluating a candidate
                set `X`, where `X` are the same features as in the model for the
                acquisition function this is to be used with. If no cost_objective
                is specified, the outputs are required to be non-negative.
            use_mean: If True, use the posterior mean, otherwise use posterior
                samples from the cost model.
            cost_objective: If specified, transform the posterior mean / the
                posterior samples from the cost model. This can be used e.g. to
                un-transform predictions/samples of a cost model fit on the
                log-transformed cost (often done to ensure non-negativity).
            min_cost: A value used to clamp the cost samples so that they are not
                too close to zero, which may cause numerical issues.
        Returns:
            The inverse-cost-weighted utiltiy.
        """
        super().__init__()
        if cost_objective is None:
            cost_objective = IdentityMCObjective()
        self.cost_model = cost_model
        self.cost_objective = cost_objective
        self._use_mean = use_mean
        self._min_cost = min_cost

    def forward(
        self,
        X: Tensor,
        deltas: Tensor,
        sampler: Optional[MCSampler] = None,
        **kwargs: Any,
    ) -> Tensor:
        r"""Evaluate the cost function on the candidates and improvements.

        Args:
            X: A `batch_shape x q x d`-dim Tensor of with `q` `d`-dim design
                points each for each t-batch.
            deltas: A `num_fantasies x batch_shape`-dim Tensor of `num_fantasy`
                samples from the marginal improvement in utility over the
                current state at `X` for each t-batch.
            sampler: A sampler used for sampling from the posterior of the cost
                model (required if `use_mean=False`, ignored if `use_mean=True`).

        Returns:
            A `num_fantasies x batch_shape`-dim Tensor of cost-weighted utilities.
        """
        if not self._use_mean and sampler is None:
            raise RuntimeError("Must provide `sampler` if `use_mean=False`")

        cost_posterior = self.cost_model.posterior(X)
        if self._use_mean:
            cost = cost_posterior.mean  # batch_shape x q x m'
        else:
            # This will be of shape num_fantasies x batch_shape x q x m'
            cost = sampler(cost_posterior)
            # TODO: Make sure this doesn't change base samples in-place
        cost = self.cost_objective(cost)

        # Ensure non-negativity of the cost
        if settings.debug.on():
            if torch.any(cost < -1e-7):
                warnings.warn(
                    "Encountered negative cost values in InverseCostWeightedUtility",
                    CostAwareWarning,
                )
        # clamp (away from zero) and sum cost across elements of the q-batch -
        # this will be of shape `num_fantasies x batch_shape` or `batch_shape`
        cost = cost.clamp_min(self._min_cost).sum(dim=-1)

        # if we are doing inverse weighting on the sample level, clamp numerator.
        if not self._use_mean:
            deltas = deltas.clamp_min(0.0)

        # compute and return the ratio on the sample level - If `use_mean=True`
        # this operation involves broadcasting the cost across fantasies
        return deltas / cost
