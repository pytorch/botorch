#!/usr/bin/env python3

"""
Batch acquisition functions using the reparameterization trick in combination
with MC sampling.

.. [Wilson2017reparam]
    Wilson, J. T., Moriconi, R., Hutter, F., & Deisenroth, M. P. (2017). The
    reparameterization trick for acquisition functions. arXiv preprint
    arXiv:1712.00424.
"""

from math import pi, sqrt
from typing import Callable, List, Optional

import torch
from torch import Tensor

from ...models import Model
from ...utils.transforms import squeeze_last_dim
from ..batch_utils import batch_mode_transform


def apply_constraints_(
    obj: Tensor,
    constraints: List[Callable[[Tensor], Tensor]],
    samples: Tensor,
    M: float,
) -> None:
    """Applies constraints using an infeasiblity penalty.

    Assigns a penalty of `-M` when not feasible, where obj is modified in-place.

    Args:
        obj: A `b x q` Tensor of objective values.
        constraints: A list of callables, each mapping a Tensor of size `b x q x t`
            to a Tensor of size `b x q`, where negative values imply feasibility.
            This callable must support broadcasting. Only relevant for multi-
            output models (`t` > 1).
        samples: A `b x q x t` Tensor of samples drawn from the posterior.
        M: The infeasible value.
    """
    if constraints is not None:
        # obj has dimensions n_samples x b x q
        # all_feasible has dimensions n_samples x b x q and tracks whether or not
        # each design point is feasible, i.e., for all constraints
        all_con_feasible = torch.ones(obj.shape, dtype=torch.uint8, device=obj.device)
        for constraint in constraints:
            # con has dimensions n_samples x b x q
            con = constraint(samples)
            all_con_feasible.mul_(con <= 0)
        obj[~all_con_feasible] = -M


def get_infeasible_cost(
    X: Tensor, model: Model, objective: Callable[[Tensor], Tensor] = squeeze_last_dim
) -> float:
    """Get infeasible cost for a model and objective.

    Computes an infeasible cost M such that -M is almost always < min_x f(x),
        so that feasible points are preferred.

    Args:
        X: A `m x d` Tensor of `m` design points to use in evaluating the minimum.
        model: A fitted model.

    Returns:
        The infeasible cost M value.
    """
    posterior = model.posterior(X)
    lb = objective(posterior.mean - 6 * posterior.variance.sqrt()).min()
    M = -lb.clamp_max(0.0)
    return M.item()


@batch_mode_transform
def batch_expected_improvement(
    X: Tensor,
    model: Model,
    best_f: float,
    objective: Callable[[Tensor], Tensor] = squeeze_last_dim,
    constraints: Optional[List[Callable[[Tensor], Tensor]]] = None,
    M: float = 0.0,
    mc_samples: int = 500,
    base_samples: Optional[Tensor] = None,
) -> Tensor:
    """q-Expected Improvement acquisition function.

    Args:
        X: A `(b) x q x d` Tensor with `b` t-batches of `q` design points each.
            X is two-dimensional, assume `b = 1`.
        model: A fitted model.
        best_f: The best (feasible) function value observed so far (assumed
            noiseless).
        objective: A callable mapping a Tensor of size `b x q x t` to a
            Tensor of size `b x q`, where `t` is the number of outputs of
            the model. Note: the callable must support broadcasting.
            If omitted, squeeze the output dimension (applicable to single-
            output models only).
        constraints: A list of callables, each mapping a Tensor of size
            `b x q x t` to a Tensor of size `b x q`, where negative values
            imply feasibility. Note: the callable must support broadcasting.
            Only relevant for multi-output models (`t` > 1).
        M: The infeasibility cost. Should be set s.t. `-M < min_x obj(x)`.
        mc_samples: The number of (quasi-) Monte-Carlo samples to use for
            approximating the expectation.
        base_samples: A fixed Tensor of N(0,1) random variables used for
            deterministic optimization.

    Returns:
        Tensor: The q-EI value of the design X for each of the `b` t-batches.
    """
    posterior = model.posterior(X)
    samples = posterior.rsample(
        sample_shape=torch.Size([mc_samples]), base_samples=base_samples
    )
    obj = objective(samples)
    # apply the constraints to the objective first; then, compare with best_f
    # (which could be -M if no feasible point has been found)
    apply_constraints_(obj=obj, constraints=constraints, samples=samples, M=M)
    obj = (obj - best_f).clamp_min_(0)
    q_ei = obj.max(dim=2)[0].mean(dim=0)
    return q_ei


@batch_mode_transform
def batch_noisy_expected_improvement(
    X: Tensor,
    model: Model,
    X_observed: Tensor,
    objective: Callable[[Tensor], Tensor] = squeeze_last_dim,
    constraints: Optional[List[Callable[[Tensor], Tensor]]] = None,
    M: float = 0.0,
    mc_samples: int = 500,
    base_samples: Optional[Tensor] = None,
) -> Tensor:
    """q-Noisy Expected Improvement acquisition function.

    Args:
        X: A `b x q x d` Tensor with `b` t-batches of `q` design points each.
            If X is two-dimensional, assume `b = 1`.
        model: A fitted model.
        X_observed: A `q' x d`-dim Tensor of `q'` design points that have
            already been observed and would be considered as the best design
            point.
        objective: A callable mapping a Tensor of size `b x q x t` to a
            Tensor of size `b x q`, where `t` is the number of outputs of
            the model. This callable must support broadcasting. If omitted,
            squeeze the output dimension (applicable to single-output models
            only).
        constraints: A list of callables, each mapping a Tensor of size
            `b x q x t` to a Tensor of size `b x q`, where negative values
            imply feasibility. This callable must support broadcasting. Only
            relevant for multi-output models (`t` > 1).
        M: The infeasibility cost. Should be set s.t. `-M < min_x obj(x)`.
        mc_samples: The number of (quasi-) Monte-Carlo samples to use for
            approximating the expectation.
        base_samples: A fixed Tensor of N(0,1) random variables used for
            deterministic optimization.

    Returns:
        Tensor: The q-NoisyEI value of the design X for each of the `b` t-batches.
    """
    q = X.shape[-2]
    # predict posterior (joint across points and tasks)
    posterior = model.posterior(torch.cat([X, X_observed], dim=-2))
    samples = posterior.rsample(
        sample_shape=torch.Size([mc_samples]), base_samples=base_samples
    )
    # compute objective value
    obj = objective(samples)
    apply_constraints_(obj=obj, constraints=constraints, samples=samples, M=M)
    diff = (
        (obj[:, :, :q].max(dim=2)[0] - obj[:, :, q:].max(dim=2)[0])
        .clamp_min_(0)
        .mean(dim=0)
    )
    return diff


@batch_mode_transform
def batch_probability_of_improvement(
    X: Tensor,
    model: Model,
    best_f: Tensor,
    mc_samples: int = 500,
    base_samples: Optional[Tensor] = None,
) -> Tensor:
    """q-Probability of Improvement acquisition function.

    Args:
        X: A `(b) x q x d` Tensor with `b` t-batches of `q` design points each.
            If `X` is two-dimensional, assume `b = 1`.
        model: A fitted model.
        best_f: The best (feasible) function value observed so far (assumed
            noiseless).
        mc_samples: The number of (quasi-) Monte-Carlo samples to use for
            approximating the probability of improvement.
        base_samples: A Tensor of N(0,1) random variables used for
            deterministic optimization.

    Returns:
        Tensor: The q-PI value of the design X for each of the `b` t-batches.
    """
    posterior = model.posterior(X)
    samples = posterior.rsample(
        sample_shape=torch.Size([mc_samples]), base_samples=base_samples
    ).max(dim=2)[0]
    val = torch.sigmoid(samples - best_f).mean(dim=0)
    return val


@batch_mode_transform
def batch_simple_regret(
    X: Tensor,
    model: Model,
    mc_samples: int = 500,
    base_samples: Optional[Tensor] = None,
) -> Tensor:
    """q-Simple Regret acquisition function.

    Args:
        X: A `(b) x q x d` Tensor with `b` t-batches of `q` design points each.
            If `X` is two-dimensional, assume `b = 1`.
        model: A fitted model.
        mc_samples: The number of (quasi-) Monte-Carlo samples to use for
            approximating the probability of improvement.
        base_samples: A Tensor of N(0,1) random variables used for
            deterministic optimization.

    Returns:
        Tensor: The q-simple regret value of the design X for each of the `b`
            t-batches.
    """
    posterior = model.posterior(X)
    val = (
        posterior.rsample(
            sample_shape=torch.Size([mc_samples]), base_samples=base_samples
        )
        .max(dim=2)[0]
        .mean(dim=0)
    )
    return val


@batch_mode_transform
def batch_upper_confidence_bound(
    X: Tensor,
    model: Model,
    beta: float,
    mc_samples: int = 500,
    base_samples: Optional[Tensor] = None,
) -> Tensor:
    """q-Upper Confidence Bound acquisition function.

    Args:
        X: A `(b) x q x d` Tensor with `b` t-batches of `q` design points each.
            If `X` is two-dimensional, assume `b = 1`.
        model: A fitted model.
        beta: Controls tradeoff between mean and standard deviation in UCB.
        mc_samples: The number of (quasi-) Monte-Carlo samples to use for
            approximating the probability of improvement.
        base_samples: A Tensor of `N(0,1)` random variables used for
            deterministic optimization.

    Returns:
        Tensor: The constrained q-UCB value of the design X for each of the
            `b` t-batches.
    """
    posterior = model.posterior(X)
    samples = posterior.rsample(
        sample_shape=torch.Size([mc_samples]), base_samples=base_samples
    )
    mean = posterior.mean
    ucb_mc_samples = (sqrt(beta * pi / 2) * (samples - mean).abs() + mean).max(dim=2)[0]
    return ucb_mc_samples.mean(dim=0)
