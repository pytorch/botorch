#!/usr/bin/env python3

from typing import Callable, List, Optional

import torch
from torch import Tensor


def get_objective_weights_transform(
    weights: Optional[Tensor]
) -> Callable[[Tensor], Tensor]:
    """Greate a linear objective callable froma set of weights.

    Create a callable mapping a Tensor of size `b x q x t` to a Tensor of size
    `b x q`, where `t` is the number of outputs (tasks) of the model using
    scalarization via the objective weights. This callable supports broadcasting
    (e.g. for calling on a tensor of shape `mc_samples x b x q x t`). For `t = 1`,
    the objective weight is used to determine the optimization direction.

    Args:
        weights: a 1-dimensional Tensor containing a weight for each task.
            If not provided, the identity mapping is used.

    Returns:
        Callable[Tensor, Tensor]: transform function using the objective weights

    """
    # if no weights provided, just extract the single output
    if weights is None:
        return lambda Y: Y.squeeze(-1)
    # TODO: replace with einsum once pytorch performance issues are resolved
    return lambda Y: torch.sum(Y * weights.view(1, 1, -1), dim=-1)


def apply_constraints_nonnegative_soft_(
    obj: Tensor,
    constraints: List[Callable[[Tensor], Tensor]],
    samples: Tensor,
    eta: float,
) -> None:
    """Applies constraints to a nonnegative objective using a sigmoid approximation
    to an indicator function for each constraint. `obj` is modified in-place.

    Args:
        obj: A `n_samples x b x q` Tensor of objective values.
        constraints: A list of callables, each mapping a Tensor of size `b x q x t`
            to a Tensor of size `b x q`, where negative values imply feasibility.
            This callable must support broadcasting. Only relevant for multi-
            output models (`t` > 1).
        samples: A `b x q x t` Tensor of samples drawn from the posterior.
        eta: The temperature parameter for the sigmoid function.
    """
    if constraints is not None:
        obj.clamp_min_(0)  # Enforce non-negativity with constraints
        for constraint in constraints:
            obj.mul_(soft_eval_constraint(constraint(samples), eta=eta))


def soft_eval_constraint(lhs: Tensor, eta: float = 1e-3) -> Tensor:
    """Element-wise evaluation of a constraint in a 'soft' fashion

    `value(x) = 1 / (1 + exp(x / eta))`

    Args:
        lhs (Tensor): The left hand side of the constraint `lhs <= 0`.
        eta (float): The temperature parameter of the softmax function. As eta
            grows larger, this approximates the Heaviside step function.

    Returns:
        Tensor: element-wise 'soft' feasibility indicator of the same shape as lhs.
            For each element x, value(x) -> 0 as x becomes positive, and value(x) -> 1
            as x becomes negative.

    """
    if eta <= 0:
        raise ValueError("eta must be positive")
    return torch.sigmoid(-lhs / eta)


def apply_constraints_(
    obj: Tensor,
    constraints: List[Callable[[Tensor], Tensor]],
    samples: Tensor,
    infeasible_cost: float,
) -> None:
    """Apply constraints using an infeasible_cost `M` for the case where
    the objective can be negative via the strategy: (1) add `M` to make obj nonnegative,
    (2) apply constraints using the sigmoid approximation, (3) shift by `-M`. `obj`
    is modified in-place.

    Args:
        obj: A `n_samples x b x q` Tensor of objective values.
        constraints: A list of callables, each mapping a Tensor of size `b x q x t`
            to a Tensor of size `b x q`, where negative values imply feasibility.
            This callable must support broadcasting. Only relevant for multi-
            output models (`t` > 1).
        samples: A `b x q x t` Tensor of samples drawn from the posterior.
        infeasible_cost: The infeasible value.
    """
    if constraints is not None:
        # obj has dimensions n_samples x b x q
        obj.add_(infeasible_cost)  # now it is nonnegative
        apply_constraints_nonnegative_soft_(
            obj=obj, constraints=constraints, samples=samples, eta=1e-3
        )
        obj.add_(-infeasible_cost)
