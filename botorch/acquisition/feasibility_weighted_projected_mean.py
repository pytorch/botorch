#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Callable, Optional, Type, Union

from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from botorch.acquisition.objective import PosteriorTransform
from botorch.models.model import Model
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor


class FeasibilityWeightedProjectedMean(AnalyticAcquisitionFunction):
    r"""Feasibility-Weighted Proejcted Mean.

    This acquisition should be used as part of the budget-constrained
    multi-step KG acquisition function only.
    """

    def __init__(
        self,
        model: Model,
        first_non_feasible_state_ind: Union[float, Tensor],
        project: Optional[Type[Callable]],
        posterior_transform: Optional[PosteriorTransform] = None,
        maximize: bool = True,
    ) -> None:
        r"""First non-feasible projected mean acquisition function.

        Args:
            model: A fitted two-outcome model, where the first output corresponds
                to the objective and the second one to the log-cost.
            first_non_feasible_state_ind: Either a scalar or a `b`-dim Tensor (batch mode)
            of zeros and ones indicating for every item in the batch whether it is feasible.
            project: Callable that maps `X_k` to a tensor of the same shape projected to
                the desired target set (e.g. target fidelities in case of multi-fidelity
                optimization).
            objective: The objective under which the output is evaluated.
            maximize: A bool indicating if the problem is a maximization problem or a
                minimization problem used to define the acquisition function accordingly.
        """
        # use AcquisitionFunction constructor to avoid check for objective
        super(AnalyticAcquisitionFunction, self).__init__(model=model)
        self.first_non_feasible_state_ind = first_non_feasible_state_ind
        self.project = project
        self.maximize = maximize
        if posterior_transform is not None:
            raise NotImplementedError("Posterior transforms currently not supported")

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate Constrained Expected Improvement on the candidate set X.

        Args:
            X: A `(b) x 1 x d`-dim Tensor of `(b)` t-batches of `d`-dim design
                points each.

        Returns:
            A `(b)`-dim Tensor of Expected Improvement values at the given
            design points `X`.
        """
        posterior = self.model.posterior(X=X, posterior_transform=None)
        means = posterior.mean.squeeze(dim=-2)  # (b) x 2
        val = self.first_non_feasible_state_ind * means[..., 0]  # (b)
        if not self.maximize:
            val = -val
        return val
