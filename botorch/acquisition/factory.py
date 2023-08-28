#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Utilities for acquisition functions.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Union

import torch
from botorch.acquisition import monte_carlo
from botorch.acquisition.multi_objective import monte_carlo as moo_monte_carlo
from botorch.acquisition.objective import MCAcquisitionObjective, PosteriorTransform
from botorch.acquisition.utils import compute_best_feasible_objective
from botorch.models.model import Model
from botorch.sampling.get_sampler import get_sampler
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
    NondominatedPartitioning,
)
from torch import Tensor


def get_acquisition_function(
    acquisition_function_name: str,
    model: Model,
    objective: MCAcquisitionObjective,
    X_observed: Tensor,
    posterior_transform: Optional[PosteriorTransform] = None,
    X_pending: Optional[Tensor] = None,
    constraints: Optional[List[Callable[[Tensor], Tensor]]] = None,
    eta: Optional[Union[Tensor, float]] = 1e-3,
    mc_samples: int = 512,
    seed: Optional[int] = None,
    *,
    # optional parameters that are only needed for certain acquisition functions
    tau: float = 1e-3,
    prune_baseline: bool = True,
    marginalize_dim: Optional[int] = None,
    cache_root: bool = True,
    beta: Optional[float] = None,
    ref_point: Union[None, List[float], Tensor] = None,
    Y: Optional[Tensor] = None,
    alpha: float = 0.0,
) -> monte_carlo.MCAcquisitionFunction:
    r"""Convenience function for initializing botorch acquisition functions.

    Args:
        acquisition_function_name: Name of the acquisition function.
        model: A fitted model.
        objective: A MCAcquisitionObjective.
        X_observed: A `m1 x d`-dim Tensor of `m1` design points that have
            already been observed.
        posterior_transform: A PosteriorTransform (optional).
        X_pending: A `m2 x d`-dim Tensor of `m2` design points whose evaluation
            is pending.
        constraints: A list of callables, each mapping a Tensor of dimension
            `sample_shape x batch-shape x q x m` to a Tensor of dimension
            `sample_shape x batch-shape x q`, where negative values imply
            feasibility. Used for all acquisition functions except qSR and qUCB.
        eta: The temperature parameter for the sigmoid function used for the
            differentiable approximation of the constraints. In case of a float the
            same eta is used for every constraint in constraints. In case of a
            tensor the length of the tensor must match the number of provided
            constraints. The i-th constraint is then estimated with the i-th
            eta value. Used for all acquisition functions except qSR and qUCB.
        mc_samples: The number of samples to use for (q)MC evaluation of the
            acquisition function.
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
    sampler = get_sampler(
        posterior=model.posterior(X_observed[:1]),
        sample_shape=torch.Size([mc_samples]),
        seed=seed,
    )
    if posterior_transform is not None and acquisition_function_name in [
        "qEHVI",
        "qNEHVI",
    ]:
        raise NotImplementedError(
            "PosteriorTransforms are not yet implemented for multi-objective "
            "acquisition functions."
        )
    # instantiate and return the requested acquisition function
    if acquisition_function_name in ("qEI", "qLogEI", "qPI"):
        # Since these are the non-noisy variants, use the posterior mean at the observed
        # inputs directly to compute the best feasible value without sampling.
        Y = model.posterior(X_observed, posterior_transform=posterior_transform).mean
        obj = objective(samples=Y, X=X_observed)
        best_f = compute_best_feasible_objective(
            samples=Y,
            obj=obj,
            constraints=constraints,
            model=model,
            objective=objective,
            posterior_transform=posterior_transform,
            X_baseline=X_observed,
        )
    if acquisition_function_name == "qEI":
        return monte_carlo.qExpectedImprovement(
            model=model,
            best_f=best_f,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
            constraints=constraints,
            eta=eta,
        )
    if acquisition_function_name == "qLogEI":
        # putting the import here to avoid circular imports
        # ideally, the entire function should be moved out of this file,
        # but since it is used for legacy code to be deprecated, we keep it here.
        from botorch.acquisition.logei import qLogExpectedImprovement

        return qLogExpectedImprovement(
            model=model,
            best_f=best_f,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
            constraints=constraints,
            eta=eta,
        )
    elif acquisition_function_name == "qPI":
        return monte_carlo.qProbabilityOfImprovement(
            model=model,
            best_f=best_f,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
            tau=tau,
            constraints=constraints,
            eta=eta,
        )
    elif acquisition_function_name == "qNEI":
        return monte_carlo.qNoisyExpectedImprovement(
            model=model,
            X_baseline=X_observed,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
            prune_baseline=prune_baseline,
            marginalize_dim=marginalize_dim,
            cache_root=cache_root,
            constraints=constraints,
            eta=eta,
        )
    elif acquisition_function_name == "qLogNEI":
        from botorch.acquisition.logei import qLogNoisyExpectedImprovement

        return qLogNoisyExpectedImprovement(
            model=model,
            X_baseline=X_observed,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
            prune_baseline=prune_baseline,
            marginalize_dim=marginalize_dim,
            cache_root=cache_root,
            constraints=constraints,
            eta=eta,
        )
    elif acquisition_function_name == "qSR":
        return monte_carlo.qSimpleRegret(
            model=model,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
        )
    elif acquisition_function_name == "qUCB":
        if beta is None:
            raise ValueError("`beta` must be not be None for qUCB.")
        return monte_carlo.qUpperConfidenceBound(
            model=model,
            beta=beta,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
        )
    elif acquisition_function_name == "qEHVI":
        if Y is None:
            raise ValueError("`Y` must not be None for qEHVI")
        if ref_point is None:
            raise ValueError("`ref_point` must not be None for qEHVI")
        # get feasible points
        if constraints is not None:
            feas = torch.stack([c(Y) <= 0 for c in constraints], dim=-1).all(dim=-1)
            Y = Y[feas]
        obj = objective(Y)
        if alpha > 0:
            partitioning = NondominatedPartitioning(
                ref_point=torch.as_tensor(ref_point, dtype=Y.dtype, device=Y.device),
                Y=obj,
                alpha=alpha,
            )
        else:
            partitioning = FastNondominatedPartitioning(
                ref_point=torch.as_tensor(ref_point, dtype=Y.dtype, device=Y.device),
                Y=obj,
            )
        return moo_monte_carlo.qExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point,
            partitioning=partitioning,
            sampler=sampler,
            objective=objective,
            constraints=constraints,
            eta=eta,
            X_pending=X_pending,
        )
    elif acquisition_function_name == "qNEHVI":
        if ref_point is None:
            raise ValueError("`ref_point` must not be None for qNEHVI")
        return moo_monte_carlo.qNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point,
            X_baseline=X_observed,
            sampler=sampler,
            objective=objective,
            constraints=constraints,
            eta=eta,
            prune_baseline=prune_baseline,
            alpha=alpha,
            X_pending=X_pending,
            marginalize_dim=marginalize_dim,
            cache_root=cache_root,
        )
    raise NotImplementedError(
        f"Unknown acquisition function {acquisition_function_name}"
    )
