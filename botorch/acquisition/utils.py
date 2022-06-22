#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Utilities for acquisition functions.
"""

from __future__ import annotations

import math
import warnings
from typing import Callable, Dict, List, Optional

import torch
from botorch import settings
from botorch.acquisition import analytic, monte_carlo, multi_objective  # noqa F401
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.multi_objective import monte_carlo as moo_monte_carlo
from botorch.acquisition.objective import (
    IdentityMCObjective,
    MCAcquisitionObjective,
    PosteriorTransform,
)
from botorch.exceptions.errors import UnsupportedError
from botorch.exceptions.warnings import SamplingWarning
from botorch.models.fully_bayesian import MCMC_DIM
from botorch.models.model import Model
from botorch.sampling.samplers import IIDNormalSampler, MCSampler, SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
    NondominatedPartitioning,
)
from botorch.utils.transforms import is_fully_bayesian
from torch import Tensor
from torch.quasirandom import SobolEngine


def get_acquisition_function(
    acquisition_function_name: str,
    model: Model,
    objective: MCAcquisitionObjective,
    X_observed: Tensor,
    posterior_transform: Optional[PosteriorTransform] = None,
    X_pending: Optional[Tensor] = None,
    constraints: Optional[List[Callable[[Tensor], Tensor]]] = None,
    mc_samples: int = 500,
    qmc: bool = True,
    seed: Optional[int] = None,
    **kwargs,
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
            feasibility. Used when constraint_transforms are not passed
            as part of the objective.
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
    if posterior_transform is not None and acquisition_function_name in [
        "qEHVI",
        "qNEHVI",
    ]:
        raise NotImplementedError(
            "PosteriorTransforms are not yet implemented for multi-objective "
            "acquisition functions."
        )
    # instantiate and return the requested acquisition function
    if acquisition_function_name in ("qEI", "qPI"):
        obj = objective(
            model.posterior(X_observed, posterior_transform=posterior_transform).mean
        )
        best_f = obj.max(dim=-1).values
    if acquisition_function_name == "qEI":
        return monte_carlo.qExpectedImprovement(
            model=model,
            best_f=best_f,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
        )
    elif acquisition_function_name == "qPI":
        return monte_carlo.qProbabilityOfImprovement(
            model=model,
            best_f=best_f,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
            tau=kwargs.get("tau", 1e-3),
        )
    elif acquisition_function_name == "qNEI":
        return monte_carlo.qNoisyExpectedImprovement(
            model=model,
            X_baseline=X_observed,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
            prune_baseline=kwargs.get("prune_baseline", False),
            marginalize_dim=kwargs.get("marginalize_dim"),
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
        if "beta" not in kwargs:
            raise ValueError("`beta` must be specified in kwargs for qUCB.")
        return monte_carlo.qUpperConfidenceBound(
            model=model,
            beta=kwargs["beta"],
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
        )
    elif acquisition_function_name == "qEHVI":
        # pyre-fixme [16]: `Model` has no attribute `train_targets`
        try:
            ref_point = kwargs["ref_point"]
        except KeyError:
            raise ValueError("`ref_point` must be specified in kwargs for qEHVI")
        try:
            Y = kwargs["Y"]
        except KeyError:
            raise ValueError("`Y` must be specified in kwargs for qEHVI")
        # get feasible points
        if constraints is not None:
            feas = torch.stack([c(Y) <= 0 for c in constraints], dim=-1).all(dim=-1)
            Y = Y[feas]
        obj = objective(Y)
        alpha = kwargs.get("alpha", 0.0)
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
            X_pending=X_pending,
        )
    elif acquisition_function_name == "qNEHVI":
        if "ref_point" not in kwargs:
            raise ValueError("`ref_point` must be specified in kwargs for qNEHVI")
        return moo_monte_carlo.qNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=kwargs["ref_point"],
            X_baseline=X_observed,
            sampler=sampler,
            objective=objective,
            constraints=constraints,
            prune_baseline=kwargs.get("prune_baseline", True),
            alpha=kwargs.get("alpha", 0.0),
            X_pending=X_pending,
            marginalize_dim=kwargs.get("marginalize_dim"),
            cache_root=kwargs.get("cache_root", True),
        )
    raise NotImplementedError(
        f"Unknown acquisition function {acquisition_function_name}"
    )


def get_infeasible_cost(
    X: Tensor,
    model: Model,
    objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
    posterior_transform: Optional[PosteriorTransform] = None,
) -> Tensor:
    r"""Get infeasible cost for a model and objective.

    For each outcome, computes an infeasible cost `M` such that
    `-M < min_x f(x)` almost always, so that feasible points are preferred.

    Args:
        X: A `n x d` Tensor of `n` design points to use in evaluating the
            minimum. These points should cover the design space well. The more
            points the better the estimate, at the expense of added computation.
        model: A fitted botorch model with `m` outcomes.
        objective: The objective with which to evaluate the model output.
        posterior_transform: A PosteriorTransform (optional).

    Returns:
        An `m`-dim tensor of infeasible cost values.

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> objective = lambda Y: Y[..., -1] ** 2
        >>> M = get_infeasible_cost(train_X, model, obj)
    """
    if objective is None:

        def objective(Y: Tensor, X: Optional[Tensor] = None):
            return Y.squeeze(-1)

    posterior = model.posterior(X, posterior_transform=posterior_transform)
    lb = objective(posterior.mean - 6 * posterior.variance.clamp_min(0).sqrt())
    if lb.ndim < posterior.mean.ndim:
        lb = lb.unsqueeze(-1)
    # Take outcome-wise min. Looping in to handle batched models.
    while lb.dim() > 1:
        lb = lb.min(dim=-2).values
    return -(lb.clamp_max(0.0))


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
            multi_objective.analytic.ExpectedHypervolumeImprovement,
            multi_objective.monte_carlo.qExpectedHypervolumeImprovement,
            multi_objective.monte_carlo.qNoisyExpectedHypervolumeImprovement,
        ),
    )


def prune_inferior_points(
    model: Model,
    X: Tensor,
    objective: Optional[MCAcquisitionObjective] = None,
    posterior_transform: Optional[PosteriorTransform] = None,
    num_samples: int = 2048,
    max_frac: float = 1.0,
    sampler: Optional[MCSampler] = None,
    marginalize_dim: Optional[int] = None,
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
        posterior_transform: A PosteriorTransform (optional).
        num_samples: The number of samples used to compute empirical
            probabilities of being the best point.
        max_frac: The maximum fraction of points to retain. Must satisfy
            `0 < max_frac <= 1`. Ensures that the number of elements in the
            returned tensor does not exceed `ceil(max_frac * n)`.
        sampler: If provided, will use this customized sampler instead of
            automatically constructing one with `num_samples`.
        marginalize_dim: A batch dimension that should be marginalized.
            For example, this is useful when using a batched fully Bayesian
            model.

    Returns:
        A `n' x d` with subset of points in `X`, where

            n' = min(N_nz, ceil(max_frac * n))

        with `N_nz` the number of points in `X` that have non-zero (empirical,
        under `num_samples` samples) probability of being the best point.
    """
    if marginalize_dim is None and is_fully_bayesian(model):
        # TODO: Properly deal with marginalizing fully Bayesian models
        marginalize_dim = MCMC_DIM

    if X.ndim > 2:
        # TODO: support batched inputs (req. dealing with ragged tensors)
        raise UnsupportedError(
            "Batched inputs `X` are currently unsupported by prune_inferior_points"
        )
    max_points = math.ceil(max_frac * X.size(-2))
    if max_points < 1 or max_points > X.size(-2):
        raise ValueError(f"max_frac must take values in (0, 1], is {max_frac}")
    with torch.no_grad():
        posterior = model.posterior(X=X, posterior_transform=posterior_transform)
    if sampler is None:
        if posterior.base_sample_shape.numel() > SobolEngine.MAXDIM:
            if settings.debug.on():
                warnings.warn(
                    f"Sample dimension q*m={posterior.base_sample_shape.numel()} "
                    f"exceeding Sobol max dimension ({SobolEngine.MAXDIM}). "
                    "Using iid samples instead.",
                    SamplingWarning,
                )
            sampler = IIDNormalSampler(num_samples=num_samples)
        else:
            sampler = SobolQMCNormalSampler(num_samples=num_samples)
    samples = sampler(posterior)
    if objective is None:
        objective = IdentityMCObjective()
    obj_vals = objective(samples, X=X)
    if obj_vals.ndim > 2:
        if obj_vals.ndim == 3 and marginalize_dim is not None:
            obj_vals = obj_vals.mean(dim=marginalize_dim)
        else:
            # TODO: support batched inputs (req. dealing with ragged tensors)
            raise UnsupportedError(
                "Models with multiple batch dims are currently unsupported by"
                " prune_inferior_points."
            )
    is_best = torch.argmax(obj_vals, dim=-1)
    idcs, counts = torch.unique(is_best, return_counts=True)

    if len(idcs) > max_points:
        counts, order_idcs = torch.sort(counts, descending=True)
        idcs = order_idcs[:max_points]

    return X[idcs]


def project_to_target_fidelity(
    X: Tensor, target_fidelities: Optional[Dict[int, float]] = None
) -> Tensor:
    r"""Project `X` onto the target set of fidelities.

    This function assumes that the set of feasible fidelities is a box, so
    projecting here just means setting each fidelity parameter to its target
    value.

    Args:
        X: A `batch_shape x q x d`-dim Tensor of with `q` `d`-dim design points
            for each t-batch.
        target_fidelities: A dictionary mapping a subset of columns of `X` (the
            fidelity parameters) to their respective target fidelity value. If
            omitted, assumes that the last column of X is the fidelity parameter
            with a target value of 1.0.

    Return:
        A `batch_shape x q x d`-dim Tensor `X_proj` with fidelity parameters
            projected to the provided fidelity values.
    """
    if target_fidelities is None:
        target_fidelities = {-1: 1.0}
    d = X.size(-1)
    # normalize to positive indices
    tfs = {k if k >= 0 else d + k: v for k, v in target_fidelities.items()}
    ones = torch.ones(*X.shape[:-1], device=X.device, dtype=X.dtype)
    # here we're looping through the feature dimension of X - this could be
    # slow for large `d`, we should optimize this for that case
    X_proj = torch.stack(
        [X[..., i] if i not in tfs else tfs[i] * ones for i in range(d)], dim=-1
    )
    return X_proj


def expand_trace_observations(
    X: Tensor, fidelity_dims: Optional[List[int]] = None, num_trace_obs: int = 0
) -> Tensor:
    r"""Expand `X` with trace observations.

    Expand a tensor of inputs with "trace observations" that are obtained during
    the evaluation of the candidate set. This is used in multi-fidelity
    optimization. It can be though of as augmenting the `q`-batch with additional
    points that are the expected trace observations.

    Let `f_i` be the `i`-th fidelity parameter. Then this functions assumes that
    for each element of the q-batch, besides the fidelity `f_i`, we will observe
    additonal fidelities `f_i1, ..., f_iK`, where `K = num_trace_obs`, during
    evaluation of the candidate set `X`. Specifically, this function assumes
    that `f_ij = (K-j) / (num_trace_obs + 1) * f_i` for all `i`. That is, the
    expansion is performed in parallel for all fidelities (it does not expand
    out all possible combinations).

    Args:
        X: A `batch_shape x q x d`-dim Tensor of with `q` `d`-dim design points
            (incl. the fidelity parameters) for each t-batch.
        fidelity_dims: The indices of the fidelity parameters. If omitted,
            assumes that the last column of X contains the fidelity parameters.
        num_trace_obs: The number of trace observations to use.

    Return:
        A `batch_shape x (q + num_trace_obs x q) x d` Tensor `X_expanded` that
            expands `X` with trace observations.
    """
    if num_trace_obs == 0:  # No need to expand if we don't use trace observations
        return X

    if fidelity_dims is None:
        fidelity_dims = [-1]

    # The general strategy in the following is to expand `X` to the desired
    # shape, and then multiply it (point-wise) with a tensor of scaling factors
    reps = [1] * (X.ndim - 2) + [1 + num_trace_obs, 1]
    X_expanded = X.repeat(*reps)  # batch_shape x (q + num_trace_obs x q) x d
    scale_fac = torch.ones_like(X_expanded)
    s_pad = 1 / (num_trace_obs + 1)
    # tensor of  num_trace_obs scaling factors equally space between 1-s_pad and s_pad
    sf = torch.linspace(1 - s_pad, s_pad, num_trace_obs, device=X.device, dtype=X.dtype)
    # repeat each element q times
    q = X.size(-2)
    sf = torch.repeat_interleave(sf, q)  # num_trace_obs * q
    # now expand this to num_trace_obs x q x num_fidelities
    sf = sf.unsqueeze(-1).expand(X_expanded.size(-2) - q, len(fidelity_dims))
    # change relevant entries of the scaling tensor
    scale_fac[..., q:, fidelity_dims] = sf
    return scale_fac * X_expanded


def project_to_sample_points(X: Tensor, sample_points: Tensor) -> Tensor:
    r"""Augment `X` with sample points at which to take weighted average.

    Args:
        X: A `batch_shape x 1 x d`-dim Tensor of with one d`-dim design points
            for each t-batch.
        sample_points: `p x d'`-dim Tensor (`d' < d`) of `d'`-dim sample points at
            which to compute the expectation. The `d'`-dims refer to the trailing
            columns of X.
    Returns:
        A `batch_shape x p x d` Tensor where the q-batch includes the `p` sample points.
    """
    batch_shape = X.shape[:-2]
    p, d_prime = sample_points.shape
    X_new = X.repeat(*(1 for _ in batch_shape), p, 1)  # batch_shape x p x d
    X_new[..., -d_prime:] = sample_points
    return X_new
