#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Helpers for handling objectives.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from botorch.utils.safe_math import log_fatmoid, logexpit
from botorch.utils.transforms import normalize_indices
from torch import Tensor


def get_objective_weights_transform(
    weights: Tensor | None,
) -> Callable[[Tensor, Tensor | None], Tensor]:
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

    def _objective(Y: Tensor, X: Tensor | None = None):
        r"""Evaluate objective.

        Note: einsum multiples Y by weights and sums over the `m`-dimension.
        Einsum is ~2x faster than using `(Y * weights.view(1, 1, -1)).sum(dim-1)`.

        Args:
            Y: A `... x b x q x m` tensor of function values.
            X: Ignored.

        Returns:
            A `... x b x q`-dim tensor of objective values.
        """
        # if no weights provided, just extract the single output
        if weights is None:
            return Y.squeeze(-1)
        return torch.einsum("...m, m", [Y, weights])

    return _objective


def apply_constraints_nonnegative_soft(
    obj: Tensor,
    constraints: list[Callable[[Tensor], Tensor]],
    samples: Tensor,
    eta: Tensor | float,
) -> Tensor:
    r"""Applies constraints to a non-negative objective.

    This function uses a sigmoid approximation to an indicator function for
    each constraint.

    Args:
        obj: A `n_samples x b x q (x m')`-dim Tensor of objective values.
        constraints: A list of callables, each mapping a Tensor of size `b x q x m`
            to a Tensor of size `b x q`, where negative values imply feasibility.
            This callable must support broadcasting. Only relevant for multi-
            output models (`m` > 1).
        samples: A `n_samples x b x q x m` Tensor of samples drawn from the posterior.
        eta: The temperature parameter for the sigmoid function. Can be either a float
            or a 1-dim tensor. In case of a float the same eta is used for every
            constraint in constraints. In case of a tensor the length of the tensor
            must match the number of provided constraints. The i-th constraint is
            then estimated with the i-th eta value.

    Returns:
        A `n_samples x b x q (x m')`-dim tensor of feasibility-weighted objectives.
    """
    w = compute_smoothed_feasibility_indicator(
        constraints=constraints, samples=samples, eta=eta
    )
    if obj.dim() == samples.dim():
        w = w.unsqueeze(-1)  # Need to unsqueeze to accommodate the outcome dimension.
    return obj.clamp_min(0).mul(w)  # Enforce non-negativity of obj, apply constraints.


def compute_feasibility_indicator(
    constraints: list[Callable[[Tensor], Tensor]] | None,
    samples: Tensor,
    marginalize_dim: int | None = None,
) -> Tensor:
    r"""Computes the feasibility of a list of constraints given posterior samples.

    Args:
        constraints: A list of callables, each mapping a batch_shape x q x m`-dim Tensor
            to a `batch_shape x q`-dim Tensor, where negative values imply feasibility.
        samples: A batch_shape x q x m`-dim Tensor of posterior samples.
        marginalize_dim: A batch dimension that should be marginalized.
            For example, this is useful when using a batched fully Bayesian
            model.

    Returns:
        A `batch_shape x q`-dim tensor of Boolean feasibility values.
    """
    ind = torch.ones(samples.shape[:-1], dtype=torch.bool, device=samples.device)
    if constraints is not None:
        for constraint in constraints:
            ind = ind.logical_and(constraint(samples) <= 0)
    if ind.ndim >= 3 and marginalize_dim is not None:
        # make sure marginalize_dim is not negative
        if marginalize_dim < 0:
            # add 1 to the normalize marginalize_dim since we have already
            # removed the output dim
            marginalize_dim = 1 + normalize_indices([marginalize_dim], d=ind.ndim)[0]

        ind = ind.float().mean(dim=marginalize_dim).round().bool()
    return ind


def compute_smoothed_feasibility_indicator(
    constraints: list[Callable[[Tensor], Tensor]],
    samples: Tensor,
    eta: Tensor | float,
    log: bool = False,
    fat: bool = False,
) -> Tensor:
    r"""Computes the smoothed feasibility indicator of a list of constraints.

    Given posterior samples, using a sigmoid to smoothly approximate the feasibility
    indicator of each individual constraint to ensure differentiability and high
    gradient signal. The `fat` and `log` options improve the numerical behavior of
    the smooth approximation.

    NOTE: *Negative* constraint values are associated with feasibility.

    Args:
        constraints: A list of callables, each mapping a Tensor of size `b x q x m`
            to a Tensor of size `b x q`, where negative values imply feasibility.
            This callable must support broadcasting. Only relevant for multi-
            output models (`m` > 1).
        samples: A `n_samples x b x q x m` Tensor of samples drawn from the posterior.
        eta: The temperature parameter for the sigmoid function. Can be either a float
            or a 1-dim tensor. In case of a float the same eta is used for every
            constraint in constraints. In case of a tensor the length of the tensor
            must match the number of provided constraints. The i-th constraint is
            then estimated with the i-th eta value.
        log: Toggles the computation of the log-feasibility indicator.
        fat: Toggles the computation of the fat-tailed feasibility indicator.

    Returns:
        A `n_samples x b x q`-dim tensor of feasibility indicator values.
    """
    if type(eta) is not Tensor:
        eta = torch.full((len(constraints),), eta)
    if len(eta) != len(constraints):
        raise ValueError(
            "Number of provided constraints and number of provided etas do not match."
        )
    if not (eta > 0).all():
        raise ValueError("eta must be positive.")
    is_feasible = torch.zeros_like(samples[..., 0])
    log_sigmoid = log_fatmoid if fat else logexpit
    for constraint, e in zip(constraints, eta):
        is_feasible = is_feasible + log_sigmoid(-constraint(samples) / e)

    return is_feasible if log else is_feasible.exp()


def compute_probabilities_of_feasibility_indicator(
    probabilties_of_feasibility: list[Callable[[Tensor], Tensor]],
    samples: Tensor,
    log: bool = False,
) -> Tensor:
    is_feasible = torch.zeros_like(samples[..., 0])
    for constraint in probabilties_of_feasibility:
        # 1 means feasible, zero means not feasible
        is_feasible = is_feasible + constraint(samples).log()
    return is_feasible if log else is_feasible.exp()


def apply_constraints(
    obj: Tensor,
    constraints: list[Callable[[Tensor], Tensor]],
    samples: Tensor,
    infeasible_cost: float,
    eta: Tensor | float = 1e-3,
) -> Tensor:
    r"""Apply constraints using an infeasible_cost `M` for negative objectives.

    This allows feasibility-weighting an objective for the case where the
    objective can be negative by using the following strategy:
    (1) Add `M` to make obj non-negative;
    (2) Apply constraints using the sigmoid approximation;
    (3) Shift by `-M`.

    Args:
        obj: A `n_samples x b x q (x m')`-dim Tensor of objective values.
        constraints: A list of callables, each mapping a Tensor of size `b x q x m`
            to a Tensor of size `b x q`, where negative values imply feasibility.
            This callable must support broadcasting. Only relevant for multi-
            output models (`m` > 1).
        samples: A `n_samples x b x q x m` Tensor of samples drawn from the posterior.
        infeasible_cost: The infeasible value.
        eta: The temperature parameter of the sigmoid function. Can be either a float
            or a 1-dim tensor. In case of a float the same eta is used for every
            constraint in constraints. In case of a tensor the length of the tensor
            must match the number of provided constraints. The i-th constraint is
            then estimated with the i-th eta value.

    Returns:
        A `n_samples x b x q (x m')`-dim tensor of feasibility-weighted objectives.
    """
    # obj has dimensions n_samples x b x q (x m')
    obj = obj.add(infeasible_cost)  # now it is nonnegative
    obj = apply_constraints_nonnegative_soft(
        obj=obj,
        constraints=constraints,
        samples=samples,
        eta=eta,
    )
    return obj.add(-infeasible_cost)
