#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

r"""
Utilities for acquisition functions.
"""

import math
from typing import Callable, Optional

import torch
from torch import Tensor

from ..exceptions.errors import UnsupportedError
from ..models.model import Model
from ..sampling.samplers import IIDNormalSampler, SobolQMCNormalSampler
from ..utils.transforms import squeeze_last_dim
from . import monte_carlo  # noqa F401
from . import analytic
from .acquisition import AcquisitionFunction
from .objective import IdentityMCObjective, MCAcquisitionObjective


def get_acquisition_function(
    acquisition_function_name: str,
    model: Model,
    objective: MCAcquisitionObjective,
    X_observed: Tensor,
    X_pending: Optional[Tensor] = None,
    mc_samples: int = 500,
    qmc: bool = True,
    seed: Optional[int] = None,
    **kwargs,
) -> "monte_carlo.MCAcquisitionFunction":
    r"""Convenience function for initializing botorch acquisition functions.

    Args:
        acquisition_function_name: Name of the acquisition function.
        model: A fitted model.
        objective: A MCAcquisitionObjective.
        X_observed: A `m1 x d`-dim Tensor of `m1` design points that have
            already been observed.
        X_pending: A `m2 x d`-dim Tensor of `m2` design points whose evaluation
            is pending.
        mc_samples: The number of samples to use for (q)MC evaluation of the
            acquisition function.
        qmc: If True, use quasi-Monte-Carlo sampling (instead of iid).
        seed: If provided, perform deterministic optimization (i.e. the
            function to optimize is fixed and not stochastic).

    Returns:
        The requested acquisition function.

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> obj = LinearMCObjective(weights=torch.tensor([1.0, 2.0]))
        >>> acqf = get_acquisition_function("qEI", model, obj, train_X)
    """
    # initialize the sampler
    if qmc:
        sampler = SobolQMCNormalSampler(num_samples=mc_samples, seed=seed)
    else:
        sampler = IIDNormalSampler(num_samples=mc_samples, seed=seed)
    # instantiate and return the requested acquisition function
    if acquisition_function_name == "qEI":
        best_f = objective(model.posterior(X_observed).mean).max().item()
        return monte_carlo.qExpectedImprovement(
            model=model,
            best_f=best_f,
            sampler=sampler,
            objective=objective,
            X_pending=X_pending,
        )
    elif acquisition_function_name == "qPI":
        best_f = objective(model.posterior(X_observed).mean).max().item()
        return monte_carlo.qProbabilityOfImprovement(
            model=model,
            best_f=best_f,
            sampler=sampler,
            objective=objective,
            X_pending=X_pending,
            tau=kwargs.get("tau", 1e-3),
        )
    elif acquisition_function_name == "qNEI":
        return monte_carlo.qNoisyExpectedImprovement(
            model=model,
            X_baseline=X_observed,
            sampler=sampler,
            objective=objective,
            X_pending=X_pending,
            prune_baseline=kwargs.get("prune_baseline", False),
        )
    elif acquisition_function_name == "qSR":
        return monte_carlo.qSimpleRegret(
            model=model, sampler=sampler, objective=objective, X_pending=X_pending
        )
    elif acquisition_function_name == "qUCB":
        if "beta" not in kwargs:
            raise ValueError("`beta` must be specified in kwargs for qUCB.")
        return monte_carlo.qUpperConfidenceBound(
            model=model,
            beta=kwargs["beta"],
            sampler=sampler,
            objective=objective,
            X_pending=X_pending,
        )
    raise NotImplementedError(
        f"Unknown acquisition function {acquisition_function_name}"
    )


def get_infeasible_cost(
    X: Tensor, model: Model, objective: Callable[[Tensor], Tensor] = squeeze_last_dim
) -> float:
    r"""Get infeasible cost for a model and objective.

    Computes an infeasible cost `M` such that `-M < min_x f(x)` almost always,
        so that feasible points are preferred.

    Args:
        X: A `n x d` Tensor of `n` design points to use in evaluating the
            minimum. These points should cover the design space well. The more
            points the better the estimate, at the expense of added computation.
        model: A fitted botorch model.
        objective: The objective with which to evaluate the model output.

    Returns:
        The infeasible cost `M` value.

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> objective = lambda Y: Y[..., -1] ** 2
        >>> M = get_infeasible_cost(train_X, model, obj)
    """
    posterior = model.posterior(X)
    lb = objective(posterior.mean - 6 * posterior.variance.clamp_min(0).sqrt()).min()
    M = -lb.clamp_max(0.0)
    return M.item()


def is_nonnegative(acq_function: AcquisitionFunction) -> bool:
    r"""Determine whether a given acquisition function is non-negative.

    Args:
        acq_function: The `AcquisitionFunction` instance.

    Returns:
        True if `acq_function` is non-negative, False if not, or if the behavior
        is unknown (for custom acquisition functions).

    Example:
        >>> qEI = qExpectedImprovement(model, best_f=0.1)
        >>> is_nonnegative(qEI)  # returns True
    """
    return isinstance(
        acq_function,
        (
            analytic.ExpectedImprovement,
            analytic.ConstrainedExpectedImprovement,
            analytic.ProbabilityOfImprovement,
            analytic.NoisyExpectedImprovement,
            monte_carlo.qExpectedImprovement,
            monte_carlo.qNoisyExpectedImprovement,
            monte_carlo.qProbabilityOfImprovement,
        ),
    )


def prune_inferior_points(
    model: Model,
    X: Tensor,
    objective: Optional[MCAcquisitionObjective] = None,
    num_samples: int = 2048,
    max_frac: float = 1.0,
) -> Tensor:
    r"""Prune points from an input tensor that are unlikely to be the best point.

    Given a model, an objective, and an input tensor `X`, this function returns
    the subset of points in `X` that have some probability of being the best
    point under the objective. This function uses sampling to estimate the
    probabilities, the higher the number of points `n` in `X` the higher the
    number of samples `num_samples` should be to obtain accurate estimates.

    Args:
        model: A fitted model. Batched models are currently not supported.
        X: An input tensor of shape `n x d`. Batched inputs are currently not
            supported.
        objective: The objective under which to evaluate the posterior.
        num_samples: The number of samples used to compute empirical
            probabilities of being the best point.
        max_frac: The maximum fraction of points to retain. Must satisfy
            `0 < max_frac <= 1`. Ensures that the number of elements in the
            returned tensor does not exceed `ceil(max_frac * n)`.

    Returns:
        A `n' x d` with subset of points in `X`, where

            n' = min(N_nz, ceil(max_frac * n))

        with `N_nz` the number of points in `X` that have non-zero (empirical,
        under `num_samples` samples) probability of being the best point.
    """
    if X.ndim > 2:
        # TODO: support batched inputs (req. dealing with ragged tensors)
        raise UnsupportedError(
            "Batched inputs `X` are currently unsupported by prune_inferior_points"
        )
    max_points = math.ceil(max_frac * X.size(-2))
    if max_points < 1 or max_points > X.size(-2):
        raise ValueError(f"max_frac must take values in (0, 1], is {max_frac}")
    sampler = SobolQMCNormalSampler(num_samples=num_samples)
    with torch.no_grad():
        posterior = model.posterior(X=X)
        samples = sampler(posterior)
    if objective is None:
        objective = IdentityMCObjective()
    obj_vals = objective(samples)
    if obj_vals.ndim > 2:
        # TODO: support batched inputs (req. dealing with ragged tensors)
        raise UnsupportedError(
            "Batched models are currently unsupported by prune_inferior_points"
        )
    is_best = torch.argmax(obj_vals, dim=-1)
    idcs, counts = torch.unique(is_best, return_counts=True)

    if len(idcs) > max_points:
        counts, order_idcs = torch.sort(counts, descending=True)
        idcs = order_idcs[:max_points]

    return X[idcs]
