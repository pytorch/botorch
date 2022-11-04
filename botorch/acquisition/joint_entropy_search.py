#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Acquisition function for joint entropy search (JES). The code utilizes the
implementation designed for the multi-objective batch setting.

References:

.. [Tu2022]
    B. Tu, A. Gandy, N. Kantas and B.Shafei. Joint Entropy Search for Multi-Objective
    Bayesian Optimization. Advances in Neural Information Processing Systems, 35.
    2022.

"""

from __future__ import annotations

from typing import Any, Optional

from botorch.acquisition.multi_objective.joint_entropy_search import (
    qLowerBoundMultiObjectiveJointEntropySearch,
)
from botorch.acquisition.multi_objective.utils import compute_sample_box_decomposition
from botorch.models.model import Model
from botorch.utils.transforms import concatenate_pending_points, t_batch_mode_transform
from torch import Tensor


class qLowerBoundJointEntropySearch(qLowerBoundMultiObjectiveJointEntropySearch):
    r"""The acquisition function for the Joint Entropy Search, where the batches
    `q > 1` are supported through the lower bound formulation.

    This acquisition function computes the mutual information between the observation
    at a candidate point `X` and the optimal input-output pair.
    """

    def __init__(
        self,
        model: Model,
        optimal_inputs: Tensor,
        optimal_outputs: Tensor,
        maximize: bool = True,
        hypercell_bounds: Tensor = None,
        X_pending: Optional[Tensor] = None,
        estimation_type: str = "LB",
        num_samples: int = 64,
        **kwargs: Any,
    ) -> None:
        r"""Joint entropy search acquisition function.

        Args:
            model: A fitted single-outcome model.
            optimal_inputs: A `num_samples x d`-dim tensor containing the sampled
                optimal inputs of dimension `d`. We assume for simplicity that each
                sample only contains one optimal set of inputs.
            optimal_outputs: A `num_samples x 1`-dim Tensor containing the optimal
                set of objectives of dimension `1`.
            maximize: If true, we consider a maximization problem.
            hypercell_bounds:  A `num_samples x 2 x J x 1`-dim Tensor containing the
                hyper-rectangle bounds for integration, where `J` is the number of
                hyper-rectangles. By default, the problem is assumed to be
                unconstrained and therefore the region of integration for a sample
                `(x*, y*)` is a `J=1` hyper-rectangle of the form  `(-infty, y^*]`
                for a maximization problem and `[y^*, +infty)` for a minimization
                problem. In the constrained setting, the region of integration also
                includes the infeasible space.
            X_pending: A `m x d`-dim Tensor of `m` design points that have been
                submitted for function evaluation, but have not yet been evaluated.
            estimation_type: A string to determine which entropy estimate is
                computed: "0", "LB", "LB2", or "MC". In the single-objective
                setting, "LB" is equivalent to "LB2".
            num_samples: The number of Monte Carlo samples used for the Monte Carlo
                estimate.
        """
        if hypercell_bounds is None:
            hypercell_bounds = compute_sample_box_decomposition(
                pareto_fronts=optimal_outputs.unsqueeze(-2), maximize=maximize
            )

        super().__init__(
            model=model,
            pareto_sets=optimal_inputs.unsqueeze(-2),
            pareto_fronts=optimal_outputs.unsqueeze(-2),
            hypercell_bounds=hypercell_bounds,
            X_pending=X_pending,
            estimation_type=estimation_type,
            condition_on_samples=True,
            num_samples=num_samples,
        )

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluates qLowerBoundJointEntropySearch at the design points `X`.

        Args:
            X: A `batch_shape x q x d`-dim Tensor of `batch_shape` t-batches with `q`
            `d`-dim design points each.

        Returns:
            A `batch_shape`-dim Tensor of acquisition values at the given design
            points `X`.
        """

        return self._compute_lower_bound_information_gain(X)
