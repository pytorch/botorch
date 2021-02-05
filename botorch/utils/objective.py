#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Helpers for handling objectives.
"""

from __future__ import annotations

from typing import Callable, List, Optional

import torch
from torch import Tensor


def get_objective_weights_transform(
    weights: Optional[Tensor],
) -> Callable[[Tensor, Optional[Tensor]], Tensor]:
    r"""Create a linear objective callable from a set of weights.

    Create a callable mapping a Tensor of size `b x q x m` and an (optional)
    Tensor of size `b x q x d` to a Tensor of size `b x q`, where `m` is the
    number of outputs of the model using scalarization via the objective weights.
    This callable supports broadcasting (e.g. for calling on a tensor of shape
    `mc_samples x b x q x m`). For `m = 1`, the objective weight is used to
    determine the optimization direction.

    Args:
        weights: a 1-dimensional Tensor containing a weight for each task.
            If not provided, the identity mapping is used.

    Returns:
        Transform function using the objective weights.

    Example:
        >>> weights = torch.tensor([0.75, 0.25])
        >>> transform = get_objective_weights_transform(weights)
    """
    # if no weights provided, just extract the single output
    if weights is None:
        return lambda Y: Y.squeeze(-1)

    def _objective(Y: Tensor, X: Optional[Tensor] = None):
        r"""Evaluate objective.

        Note: einsum multiples Y by weights and sums over the `m`-dimension.
        Einsum is ~2x faster than using `(Y * weights.view(1, 1, -1)).sum(dim-1)`.

        Args:
            Y: A `... x b x q x m` tensor of function values.

        Returns:
            A `... x b x q`-dim tensor of objective values.
        """
        return torch.einsum("...m, m", [Y, weights])

    return _objective


def apply_constraints_nonnegative_soft(
    obj: Tensor,
    constraints: List[Callable[[Tensor], Tensor]],
    samples: Tensor,
    eta: float,
) -> Tensor:
    r"""Applies constraints to a non-negative objective.

    This function uses a sigmoid approximation to an indicator function for
    each constraint.

    Args:
        obj: A `n_samples x b x q` Tensor of objective values.
        constraints: A list of callables, each mapping a Tensor of size `b x q x m`
            to a Tensor of size `b x q`, where negative values imply feasibility.
            This callable must support broadcasting. Only relevant for multi-
            output models (`m` > 1).
        samples: A `b x q x m` Tensor of samples drawn from the posterior.
        eta: The temperature parameter for the sigmoid function.

    Returns:
        A `n_samples x b x q`-dim tensor of feasibility-weighted objectives.
    """
    obj = obj.clamp_min(0)  # Enforce non-negativity with constraints
    for constraint in constraints:
        obj = obj.mul(soft_eval_constraint(constraint(samples), eta=eta))
    return obj


def soft_eval_constraint(lhs: Tensor, eta: float = 1e-3) -> Tensor:
    r"""Element-wise evaluation of a constraint in a 'soft' fashion

    `value(x) = 1 / (1 + exp(x / eta))`

    Args:
        lhs: The left hand side of the constraint `lhs <= 0`.
        eta: The temperature parameter of the softmax function. As eta
            grows larger, this approximates the Heaviside step function.

    Returns:
        Element-wise 'soft' feasibility indicator of the same shape as `lhs`.
        For each element `x`, `value(x) -> 0` as `x` becomes positive, and
        `value(x) -> 1` as x becomes negative.
    """
    if eta <= 0:
        raise ValueError("eta must be positive")
    return torch.sigmoid(-lhs / eta)


def apply_constraints(
    obj: Tensor,
    constraints: List[Callable[[Tensor], Tensor]],
    samples: Tensor,
    infeasible_cost: float,
    eta: float = 1e-3,
) -> Tensor:
    r"""Apply constraints using an infeasible_cost `M` for negative objectives.

    This allows feasibility-weighting an objective for the case where the
    objective can be negative by usingthe following strategy:
    (1) add `M` to make obj nonnegative
    (2) apply constraints using the sigmoid approximation
    (3) shift by `-M`

    Args:
        obj: A `n_samples x b x q` Tensor of objective values.
        constraints: A list of callables, each mapping a Tensor of size `b x q x m`
            to a Tensor of size `b x q`, where negative values imply feasibility.
            This callable must support broadcasting. Only relevant for multi-
            output models (`m` > 1).
        samples: A `b x q x m` Tensor of samples drawn from the posterior.
        infeasible_cost: The infeasible value.
        eta: The temperature parameter of the sigmoid function.

    Returns:
        A `n_samples x b x q`-dim tensor of feasibility-weighted objectives.
    """
    # obj has dimensions n_samples x b x q
    obj = obj.add(infeasible_cost)  # now it is nonnegative
    obj = apply_constraints_nonnegative_soft(
        obj=obj, constraints=constraints, samples=samples, eta=eta
    )
    return obj.add(-infeasible_cost)
