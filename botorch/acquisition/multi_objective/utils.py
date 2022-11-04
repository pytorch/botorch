#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Utilities for multi-objective acquisition functions.
"""

from __future__ import annotations

import math
import warnings
from typing import Callable, List, Optional

import torch
from botorch import settings
from botorch.acquisition import monte_carlo  # noqa F401
from botorch.acquisition.multi_objective.objective import (
    IdentityMCMultiOutputObjective,
    MCMultiOutputObjective,
)
from botorch.exceptions.errors import UnsupportedError
from botorch.exceptions.warnings import BotorchWarning, SamplingWarning
from botorch.models.fully_bayesian import MCMC_DIM
from botorch.models.model import Model
from botorch.sampling.samplers import IIDNormalSampler, SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.box_decomposition import (
    BoxDecomposition,
)
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.transforms import is_fully_bayesian, normalize_indices
from torch import Tensor
from torch.quasirandom import SobolEngine


def get_default_partitioning_alpha(num_objectives: int) -> float:
    r"""Determines an approximation level based on the number of objectives.

    If `alpha` is 0, FastNondominatedPartitioning should be used. Otherwise,
    an approximate NondominatedPartitioning should be used with approximation
    level `alpha`.

    Args:
        num_objectives: the number of objectives.

    Returns:
        The approximation level `alpha`.
    """
    if num_objectives <= 4:
        return 0.0
    elif num_objectives > 6:
        warnings.warn("EHVI works best for less than 7 objectives.", BotorchWarning)
    return 10 ** (-8 + num_objectives)


def prune_inferior_points_multi_objective(
    model: Model,
    X: Tensor,
    ref_point: Tensor,
    objective: Optional[MCMultiOutputObjective] = None,
    constraints: Optional[List[Callable[[Tensor], Tensor]]] = None,
    num_samples: int = 2048,
    max_frac: float = 1.0,
    marginalize_dim: Optional[int] = None,
) -> Tensor:
    r"""Prune points from an input tensor that are unlikely to be pareto optimal.

    Given a model, an objective, and an input tensor `X`, this function returns
    the subset of points in `X` that have some probability of being pareto
    optimal, better than the reference point, and feasible. This function uses
    sampling to estimate the probabilities, the higher the number of points `n`
    in `X` the higher the number of samples `num_samples` should be to obtain
    accurate estimates.

    Args:
        model: A fitted model. Batched models are currently not supported.
        X: An input tensor of shape `n x d`. Batched inputs are currently not
            supported.
        ref_point: The reference point.
        objective: The objective under which to evaluate the posterior.
        constraints: A list of callables, each mapping a Tensor of dimension
            `sample_shape x batch-shape x q x m` to a Tensor of dimension
            `sample_shape x batch-shape x q`, where negative values imply
            feasibility.
        num_samples: The number of samples used to compute empirical
            probabilities of being the best point.
        max_frac: The maximum fraction of points to retain. Must satisfy
            `0 < max_frac <= 1`. Ensures that the number of elements in the
            returned tensor does not exceed `ceil(max_frac * n)`.
        marginalize_dim: A batch dimension that should be marginalized.
            For example, this is useful when using a batched fully Bayesian
            model.

    Returns:
        A `n' x d` with subset of points in `X`, where

            n' = min(N_nz, ceil(max_frac * n))

        with `N_nz` the number of points in `X` that have non-zero (empirical,
        under `num_samples` samples) probability of being pareto optimal.
    """
    if marginalize_dim is None and is_fully_bayesian(model):
        # TODO: Properly deal with marginalizing fully Bayesian models
        marginalize_dim = MCMC_DIM

    if X.ndim > 2:
        # TODO: support batched inputs (req. dealing with ragged tensors)
        raise UnsupportedError(
            "Batched inputs `X` are currently unsupported by "
            "prune_inferior_points_multi_objective"
        )
    max_points = math.ceil(max_frac * X.size(-2))
    if max_points < 1 or max_points > X.size(-2):
        raise ValueError(f"max_frac must take values in (0, 1], is {max_frac}")
    with torch.no_grad():
        posterior = model.posterior(X=X)
    if posterior.event_shape.numel() > SobolEngine.MAXDIM:
        if settings.debug.on():
            warnings.warn(
                f"Sample dimension q*m={posterior.event_shape.numel()} exceeding Sobol "
                f"max dimension ({SobolEngine.MAXDIM}). Using iid samples instead.",
                SamplingWarning,
            )
        sampler = IIDNormalSampler(num_samples=num_samples)
    else:
        sampler = SobolQMCNormalSampler(num_samples=num_samples)
    samples = sampler(posterior)
    if objective is None:
        objective = IdentityMCMultiOutputObjective()
    obj_vals = objective(samples, X=X)
    if obj_vals.ndim > 3:
        if obj_vals.ndim == 4 and marginalize_dim is not None:
            obj_vals = obj_vals.mean(dim=marginalize_dim)
        else:
            # TODO: support batched inputs (req. dealing with ragged tensors)
            raise UnsupportedError(
                "Models with multiple batch dims are currently unsupported by"
                " prune_inferior_points_multi_objective."
            )
    if constraints is not None:
        infeas = torch.stack([c(samples) > 0 for c in constraints], dim=0).any(dim=0)
        if infeas.ndim == 3 and marginalize_dim is not None:
            # make sure marginalize_dim is not negative
            if marginalize_dim < 0:
                # add 1 to the normalize marginalize_dim since we have already
                # removed the output dim
                marginalize_dim = (
                    1 + normalize_indices([marginalize_dim], d=infeas.ndim)[0]
                )

            infeas = infeas.float().mean(dim=marginalize_dim).round().bool()
        # set infeasible points to be the ref point
        obj_vals[infeas] = ref_point
    pareto_mask = is_non_dominated(obj_vals, deduplicate=False) & (
        obj_vals > ref_point
    ).all(dim=-1)
    probs = pareto_mask.to(dtype=X.dtype).mean(dim=0)
    idcs = probs.nonzero().view(-1)
    if idcs.shape[0] > max_points:
        counts, order_idcs = torch.sort(probs, descending=True)
        idcs = order_idcs[:max_points]
    effective_n_w = obj_vals.shape[-2] // X.shape[-2]
    idcs = (idcs / effective_n_w).long().unique()
    return X[idcs]


def compute_sample_box_decomposition(
    pareto_fronts: Tensor,
    partitioning: BoxDecomposition = DominatedPartitioning,
    maximize: bool = True,
    num_constraints: Optional[int] = 0,
) -> Tensor:
    r"""Compute the box decomposition associated with some sampled Pareto fronts.
    This also handles the single-objective settings, where we have a sampled set
    of optimal single-objective outputs.

    The hypercell bounds is a Tensor of shape `num_pareto_samples x 2 x J x (M + K)`,
    where `J`= max(num_boxes) i.e. the smallest number of boxes needed to
    partition all the Pareto samples.

    To take advantage of batch computations, we pad the bounds with a `2 x (M + K)`
    -dim Tensor `[ref_point, ref_point]`, when the number of boxes required is smaller
    than `max(num_boxes)`, where `ref_point = NEG_INF`.

    For constraints, we assume an input `x` is feasible if `f_k(x) <= 0`.

    Args:
        pareto_fronts: A `num_pareto_samples x num_pareto_points x M` dim Tensor
            containing the sampled optimal set of objectives.
        partitioning: A `BoxDecomposition` module that is used to obtain the
            hyper-rectangle bounds for integration. In the unconstrained case, this
            gives the partition of the dominated space. In the constrained case, this
            gives the partition of the feasible dominated space union the infeasible
            space.
        maximize: If true, the box-decomposition is computed assuming maximization.
        num_constraints: The number of constraints `K`.

    Returns:
        A `num_pareto_samples x 2 x J x (M + K)`-dim Tensor containing the bounds for
        the hyper-rectangles.
    """
    tkwargs = {"dtype": pareto_fronts.dtype, "device": pareto_fronts.device}
    # We will later compute `norm.log_prob(NEG_INF)`, this is `-inf` if `NEG_INF` is
    # too small.
    NEG_INF = -1e10

    if pareto_fronts.ndim != 3:
        raise UnsupportedError(
            "Currently this only supports Pareto fronts of the shape "
            "`num_pareto_samples x num_pareto_points x num_objectives`."
        )

    num_pareto_samples = pareto_fronts.shape[0]
    M = pareto_fronts.shape[-1]
    K = num_constraints
    ref_point = torch.ones(M, **tkwargs) * NEG_INF
    weight = 1.0 if maximize else -1.0

    if M == 1:
        # Only consider a Pareto front with one element.
        extreme_values = weight * torch.max(weight * pareto_fronts, dim=-2).values
        ref_point = weight * ref_point.expand(extreme_values.shape)

        if maximize:
            hypercell_bounds = torch.stack(
                [ref_point, extreme_values], axis=-2
            ).unsqueeze(-1)
        else:
            hypercell_bounds = torch.stack(
                [extreme_values, ref_point], axis=-2
            ).unsqueeze(-1)
    else:
        box_bounds = []
        num_boxes = []

        # Iterate through the samples and compute the box decompositions.
        for i in range(num_pareto_samples):
            # Dominated partitioning assumes maximization.
            # If minimizing we consider negative the Pareto front.
            box_bounds_i = partitioning(
                ref_point, weight * pareto_fronts[i, :, :]
            ).hypercell_bounds

            # Reverse the transformation.
            if not maximize:
                box_bounds_i = weight * torch.flip(box_bounds_i, dims=[0])

            num_boxes = num_boxes + [box_bounds_i.shape[-2]]
            box_bounds = box_bounds + [box_bounds_i]

        # Create a Tensor to contain the padded box bounds.
        hypercell_bounds = NEG_INF * torch.ones(
            (num_pareto_samples, 2, max(num_boxes), M), **tkwargs
        )

        for i in range(num_pareto_samples):
            box_bounds_i = box_bounds[i]
            num_boxes_i = num_boxes[i]
            hypercell_bounds[i, :, 0:num_boxes_i, :] = box_bounds_i

    # Add an extra box for the inequality constraint.
    if K > 0:
        # `num_pareto_samples x 2 x (J - 1) x K`
        feasible_boxes = torch.zeros(
            hypercell_bounds.shape[:-1] + torch.Size([K]), **tkwargs
        )

        feasible_boxes[..., 0, :, :] = NEG_INF
        # `num_pareto_samples x 2 x (J - 1) x (M + K)`
        hypercell_bounds = torch.cat([hypercell_bounds, feasible_boxes], dim=-1)

        # `num_pareto_samples x 2 x 1 x (M + K)`
        infeasible_box = torch.zeros(
            hypercell_bounds.shape[:-2] + torch.Size([1, M + K]), **tkwargs
        )

        infeasible_box[..., 1, :, :] = -NEG_INF
        # `num_pareto_samples x 2 x J x (M + K)`
        hypercell_bounds = torch.cat([hypercell_bounds, infeasible_box], dim=-2)

    # `num_pareto_samples x 2 x J x (M + K)`
    return hypercell_bounds
