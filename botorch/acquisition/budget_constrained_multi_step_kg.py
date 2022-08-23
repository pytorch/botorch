#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Type, Union

import torch
from botorch.acquisition.feasibility_weighted_projected_mean import (
    FeasibilityWeightedProjectedMean,
)
from botorch.acquisition.multi_step_lookahead import qMultiStepLookahead
from botorch.models.model import Model
from botorch.sampling.samplers import MCSampler
from botorch.utils.objective import soft_eval_constraint
from torch import Tensor
from torch.nn import Module


class BudgetConstrainedMultiStepLookaheadKG(qMultiStepLookahead):
    r"""Budget-Constrained Multi-Step Knowledge Gradient."""

    def __init__(
        self,
        model: Model,
        budget: Union[float, Tensor],
        num_fantasies: Optional[List[int]] = None,
        project: Optional[Type[Callable]] = None,
        samplers: Optional[List[MCSampler]] = None,
        X_pending: Optional[Tensor] = None,
        maximize: bool = True,
        collapse_fantasy_base_samples: bool = True,
    ) -> None:
        r"""Budget-Constrained Multi-Step Knowledge Gradient. (one-shot optimization).

        Args:
            model: A fitted two-output model, where the first output corresponds to the
                objective, and the second one to the cost.
            budget: Budget constraint used in Lagrangian term.
            num_fantasies: A list `[f_1, ..., f_k]` containing the number of fantasy
                points to use for the `k` look-ahead steps.
            project: Callable that maps `X_k` to a tensor of the same shape projected to
                the desired target set (e.g. target fidelities in case of multi-fidelity
                optimization).
            samplers: A list of MCSampler objects to be used for sampling fantasies in
                each stage.
            X_pending: A `m x d`-dim Tensor of `m` design points that have points that
                have been submitted for function evaluation but have not yet been
                evaluated. Concatenated into `X` upon forward call. Copied and set to
                have no gradient.
            collapse_fantasy_base_samples: If True, collapse_batch_dims of the Samplers
                will be applied on fantasy batch dimensions as well, meaning that base
                samples are the same in all subtrees starting from the same level.
            maximize: A bool indicating if the problem is a maximization problem or a
                minimization problem used to define the acquisition function accordingly.
        """
        self.budget = budget
        self.maximize = maximize

        if project is None:

            def project(X):
                return X

        self.project = project

        n_lookahead_steps = len(num_fantasies)

        batch_sizes = [1 for _ in range(n_lookahead_steps)]
        valfunc_cls = [None]
        valfunc_argfacs = [None]
        for n in range(1, n_lookahead_steps + 1):
            valfunc_cls.append(FeasibilityWeightedProjectedMean)
            valfunc_argfacs.append(
                SmoothedFeasibilityWeightedProjectedMeanArgfac(
                    final_step=n == n_lookahead_steps,
                    budget=self.budget,
                    project=self.project,
                    maximize=self.maximize,
                )
            )

        super().__init__(
            model=model,
            batch_sizes=batch_sizes,
            num_fantasies=num_fantasies,
            samplers=samplers,
            valfunc_cls=valfunc_cls,
            valfunc_argfacs=valfunc_argfacs,
            X_pending=X_pending,
            collapse_fantasy_base_samples=collapse_fantasy_base_samples,
        )


class FeasibilityWeightedProjectedMeanArgfac(Module):
    r"""Extracts a tensor of zeros and ones indicating whether the current
    state is the first non feasible state."""

    def __init__(
        self,
        final_step: bool,
        budget: Union[float, Tensor],
        project: Optional[Type[Callable]] = None,
    ) -> None:
        super().__init__()
        self.final_step = final_step
        self.budget = budget
        self.project = project

    def forward(self, model: Model, X: Tensor) -> Dict[str, Any]:
        y = torch.transpose(model.train_targets, -2, -1)
        y_original_scale = model.outcome_transform.untransform(y)[0]
        log_costs = y_original_scale[..., 1]
        costs = torch.exp(log_costs)
        previous_budget = self.budget - costs[..., :-1].sum(dim=-1)
        first_non_feasible_state_ind = torch.where(previous_budget >= 0.0, 1.0, 0.0)

        if not self.final_step:
            current_budget = self.budget - costs.sum(dim=-1)
            first_non_feasible_state_ind *= torch.where(current_budget < 0.0, 1.0, 0.0)

        params = {
            "first_non_feasible_state_ind": first_non_feasible_state_ind,
            "project": self.project,
        }
        return params


class SmoothedFeasibilityWeightedProjectedMeanArgfac(Module):
    r"""Extracts a tensor of zeros and ones indicating whether the current
    state is the first non feasible state."""

    def __init__(
        self,
        final_step: bool,
        budget: Union[float, Tensor],
        project: Optional[Type[Callable]] = None,
        maximize: bool = True,
    ) -> None:
        super().__init__()
        self.final_step = final_step
        self.budget = budget
        self.project = project
        self.maximize = maximize

    def forward(self, model: Model, X: Tensor) -> Dict[str, Any]:
        y = torch.transpose(model.train_targets, -2, -1)
        y_original_scale = model.outcome_transform.untransform(y)[0]
        log_costs = y_original_scale[..., 1]
        costs = torch.exp(log_costs)
        previous_budget = self.budget - costs[..., :-1].sum(dim=-1)
        smoothed_first_non_feasible_state_ind = soft_eval_constraint(
            lhs=-previous_budget
        )

        if not self.final_step:
            current_budget = self.budget - costs.sum(dim=-1)
            smoothed_first_non_feasible_state_ind = (
                smoothed_first_non_feasible_state_ind.mul(
                    soft_eval_constraint(lhs=current_budget)
                )
            )

        params = {
            "first_non_feasible_state_ind": smoothed_first_non_feasible_state_ind,
            "project": self.project,
            "maximize": self.maximize,
        }
        return params
