#!/usr/bin/env python3

"""
Utilities for acquisition functions.
"""

from typing import Callable, Optional

import torch
from torch import Tensor

from ..models.model import Model
from ..utils.transforms import squeeze_last_dim
from .monte_carlo import (
    MCAcquisitionFunction,
    qExpectedImprovement,
    qNoisyExpectedImprovement,
    qProbabilityOfImprovement,
    qSimpleRegret,
    qUpperConfidenceBound,
)
from .objective import MCAcquisitionObjective
from .sampler import IIDNormalSampler, SobolQMCNormalSampler


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
) -> MCAcquisitionFunction:
    """Convenience function for initializing Acquisition Functions.

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
        AcquisitionFunction: the acquisition function
    """
    # initialize the sampler
    if qmc:
        sampler = SobolQMCNormalSampler(num_samples=mc_samples, seed=seed)
    else:
        sampler = IIDNormalSampler(num_samples=mc_samples, seed=seed)
    # instantiate and return the requested acquisition function
    if acquisition_function_name == "qEI":
        best_f = objective(model.posterior(X_observed).mean).max().item()
        return qExpectedImprovement(
            model=model, best_f=best_f, sampler=sampler, objective=objective
        )
    elif acquisition_function_name == "qPI":
        best_f = objective(model.posterior(X_observed).mean).max().item()
        return qProbabilityOfImprovement(
            model=model,
            best_f=best_f,
            sampler=sampler,
            objective=objective,
            tau=kwargs.get("tau", 1e-3),
        )
    elif acquisition_function_name == "qNEI":
        if X_pending is None:
            X_baseline = X_observed
        else:
            X_baseline = torch.cat([X_observed, X_pending], dim=-2)
        return qNoisyExpectedImprovement(
            model=model, X_baseline=X_baseline, sampler=sampler, objective=objective
        )
    elif acquisition_function_name == "qSR":
        return qSimpleRegret(model=model, sampler=sampler, objective=objective)
    elif acquisition_function_name == "qUCB":
        if "beta" not in kwargs:
            raise ValueError("`beta` must be specified in kwargs for qUCB.")
        return qUpperConfidenceBound(
            model=model, beta=kwargs["beta"], sampler=sampler, objective=objective
        )
    raise NotImplementedError(
        f"Unknown acquisition function {acquisition_function_name}"
    )


def get_infeasible_cost(
    X: Tensor, model: Model, objective: Callable[[Tensor], Tensor] = squeeze_last_dim
) -> float:
    """Get infeasible cost for a model and objective.

    Computes an infeasible cost M such that -M is almost always < min_x f(x),
        so that feasible points are preferred.

    Args:
        X: A `m x d` Tensor of `m` design points to use in evaluating the
            minimum. These points should cover the design space well. The more
            points the better the estimate, at the expense of added computation.
        model: A fitted model.
        objective: The objective with which to evaluate the model output.

    Returns:
        The infeasible cost M value.
    """
    posterior = model.posterior(X)
    lb = objective(posterior.mean - 6 * posterior.variance.sqrt()).min()
    M = -lb.clamp_max(0.0)
    return M.item()
