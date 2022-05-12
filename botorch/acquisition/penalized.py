#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Modules to add regularization to acquisition functions.
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from botorch.acquisition.objective import GenericMCObjective
from botorch.exceptions import UnsupportedError
from torch import Tensor


class L2Penalty(torch.nn.Module):
    r"""L2 penalty class to be added to any arbitrary acquisition function
    to construct a PenalizedAcquisitionFunction."""

    def __init__(self, init_point: Tensor):
        r"""Initializing L2 regularization.

        Args:
            init_point: The "1 x dim" reference point against which
                we want to regularize.
        """
        super().__init__()
        self.init_point = init_point

    def forward(self, X: Tensor) -> Tensor:
        r"""
        Args:
            X: A "batch_shape x q x dim" representing the points to be evaluated.

        Returns:
            A tensor of size "batch_shape" representing the acqfn for each q-batch.
        """
        regularization_term = (
            torch.norm((X - self.init_point), p=2, dim=-1).max(dim=-1).values ** 2
        )
        return regularization_term


class L1Penalty(torch.nn.Module):
    r"""L1 penalty class to be added to any arbitrary acquisition function
    to construct a PenalizedAcquisitionFunction."""

    def __init__(self, init_point: Tensor):
        r"""Initializing L1 regularization.

        Args:
            init_point: The "1 x dim" reference point against which
                we want to regularize.
        """
        super().__init__()
        self.init_point = init_point

    def forward(self, X: Tensor) -> Tensor:
        r"""
        Args:
            X: A "batch_shape x q x dim" representing the points to be evaluated.

        Returns:
            A tensor of size "batch_shape" representing the acqfn for each q-batch.
        """
        regularization_term = (
            torch.norm((X - self.init_point), p=1, dim=-1).max(dim=-1).values
        )
        return regularization_term


class GaussianPenalty(torch.nn.Module):
    r"""Gaussian penalty class to be added to any arbitrary acquisition function
    to construct a PenalizedAcquisitionFunction."""

    def __init__(self, init_point: Tensor, sigma: float):
        r"""Initializing Gaussian regularization.

        Args:
            init_point: The "1 x dim" reference point against which
                we want to regularize.
            sigma: The parameter used in gaussian function.
        """
        super().__init__()
        self.init_point = init_point
        self.sigma = sigma

    def forward(self, X: Tensor) -> Tensor:
        r"""
        Args:
            X: A "batch_shape x q x dim" representing the points to be evaluated.

        Returns:
            A tensor of size "batch_shape" representing the acqfn for each q-batch.
        """
        sq_diff = torch.norm((X - self.init_point), p=2, dim=-1) ** 2
        pdf = torch.exp(sq_diff / 2 / self.sigma**2)
        regularization_term = pdf.max(dim=-1).values
        return regularization_term


class GroupLassoPenalty(torch.nn.Module):
    r"""Group lasso penalty class to be added to any arbitrary acquisition function
    to construct a PenalizedAcquisitionFunction."""

    def __init__(self, init_point: Tensor, groups: List[List[int]]):
        r"""Initializing Group-Lasso regularization.

        Args:
            init_point: The "1 x dim" reference point against which we want
                to regularize.
            groups: Groups of indices used in group lasso.
        """
        super().__init__()
        self.init_point = init_point
        self.groups = groups

    def forward(self, X: Tensor) -> Tensor:
        r"""
        X should be batch_shape x 1 x dim tensor. Evaluation for q-batch is not
        implemented yet.
        """
        if X.shape[-2] != 1:
            raise NotImplementedError(
                "group-lasso has not been implemented for q>1 yet."
            )

        regularization_term = group_lasso_regularizer(
            X=X.squeeze(-2) - self.init_point, groups=self.groups
        )
        return regularization_term


class PenalizedAcquisitionFunction(AcquisitionFunction):
    r"""Single-outcome acquisition function regularized by the given penalty.

    The usage is similar to:
        raw_acqf = NoisyExpectedImprovement(...)
        penalty = GroupLassoPenalty(...)
        acqf = PenalizedAcquisitionFunction(raw_acqf, penalty)
    """

    def __init__(
        self,
        raw_acqf: AcquisitionFunction,
        penalty_func: torch.nn.Module,
        regularization_parameter: float,
    ) -> None:
        r"""Initializing Group-Lasso regularization.

        Args:
            raw_acqf: The raw acquisition function that is going to be regularized.
            penalty_func: The regularization function.
            regularization_parameter: Regularization parameter used in optimization.
        """
        super().__init__(model=raw_acqf.model)
        self.raw_acqf = raw_acqf
        self.penalty_func = penalty_func
        self.regularization_parameter = regularization_parameter

    def forward(self, X: Tensor) -> Tensor:
        raw_value = self.raw_acqf(X=X)
        penalty_term = self.penalty_func(X)
        return raw_value - self.regularization_parameter * penalty_term

    @property
    def X_pending(self) -> Optional[Tensor]:
        return self.raw_acqf.X_pending

    def set_X_pending(self, X_pending: Optional[Tensor] = None) -> None:
        if not isinstance(self.raw_acqf, AnalyticAcquisitionFunction):
            self.raw_acqf.set_X_pending(X_pending=X_pending)
        else:
            raise UnsupportedError(
                "The raw acquisition function is Analytic and does not account "
                "for X_pending yet."
            )


def group_lasso_regularizer(X: Tensor, groups: List[List[int]]) -> Tensor:
    r"""Computes the group lasso regularization function for the given point.

    Args:
        X: A bxd tensor representing the points to evaluate the regularization at.
        groups: List of indices of different groups.

    Returns:
        Computed group lasso norm of at the given points.
    """
    return torch.sum(
        torch.stack(
            [math.sqrt(len(g)) * torch.norm(X[..., g], p=2, dim=-1) for g in groups],
            dim=-1,
        ),
        dim=-1,
    )


class L1PenaltyObjective(torch.nn.Module):
    r"""
    L1 penalty objective class. An instance of this class can be added to any
    arbitrary objective to construct a PenalizedMCObjective.
    """

    def __init__(self, init_point: Tensor):
        r"""Initializing L1 penalty objective.

        Args:
            init_point: The "1 x dim" reference point against which
                we want to regularize.
        """
        super().__init__()
        self.init_point = init_point

    def forward(self, X: Tensor) -> Tensor:
        r"""
        Args:
            X: A "batch_shape x q x dim" representing the points to be evaluated.

        Returns:
            A "1 x batch_shape x q" tensor representing the penalty for each point.
            The first dimension corresponds to the dimension of MC samples.
        """
        return torch.norm((X - self.init_point), p=1, dim=-1).unsqueeze(dim=0)


class PenalizedMCObjective(GenericMCObjective):
    r"""Penalized MC objective.

    Allows to construct a penaltized MC-objective by adding a penalty term to
    the original objective.

        mc_acq(X) = objective(X) + penalty_objective(X)

    Note: PenalizedMCObjective allows adding penalty at the MCObjective level,
    different from the AcquisitionFunction level in PenalizedAcquisitionFunction.

    Example:
        >>> regularization_parameter = 0.01
        >>> init_point = torch.zeros(3) # assume data dim is 3
        >>> objective = lambda Y, X: torch.sqrt(Y).sum(dim=-1)
        >>> l1_penalty_objective = L1PenaltyObjective(init_point=init_point)
        >>> l1_penalized_objective = PenalizedMCObjective(
                objective, l1_penalty_objective, regularization_parameter
            )
        >>> samples = sampler(posterior)
                objective, l1_penalty_objective, regularization_parameter
    """

    def __init__(
        self,
        objective: Callable[[Tensor, Optional[Tensor]], Tensor],
        penalty_objective: torch.nn.Module,
        regularization_parameter: float,
    ) -> None:
        r"""Penalized MC objective.

        Args:
            objective: A callable `f(samples, X)` mapping a
                `sample_shape x batch-shape x q x m`-dim Tensor `samples` and
                an optional `batch-shape x q x d`-dim Tensor `X` to a
                `sample_shape x batch-shape x q`-dim Tensor of objective values.
            penalty_objective: A torch.nn.Module `f(X)` that takes in a
                `batch-shape x q x d`-dim Tensor `X` and outputs a
                `1 x batch-shape x q`-dim Tensor of penalty objective values.
            regularization_parameter: weight of the penalty (regularization) term
        """
        super().__init__(objective=objective)
        self.penalty_objective = penalty_objective
        self.regularization_parameter = regularization_parameter

    def forward(self, samples: Tensor, X: Optional[Tensor] = None) -> Tensor:
        r"""Evaluate the penalized objective on the samples.

        Args:
            samples: A `sample_shape x batch_shape x q x m`-dim Tensors of
                samples from a model posterior.
            X: A `batch_shape x q x d`-dim tensor of inputs. Relevant only if
                the objective depends on the inputs explicitly.

        Returns:
            A `sample_shape x batch_shape x q`-dim Tensor of objective values
            with penalty added for each point.
        """
        obj = super().forward(samples=samples, X=X)
        penalty_obj = self.penalty_objective(X)
        return obj - self.regularization_parameter * penalty_obj
