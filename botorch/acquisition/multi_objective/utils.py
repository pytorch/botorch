#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
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
from botorch.exceptions.errors import BotorchError, UnsupportedError
from botorch.exceptions.warnings import BotorchWarning
from botorch.exceptions.warnings import SamplingWarning
from botorch.models.model import Model
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.sampling.samplers import IIDNormalSampler, SobolQMCNormalSampler
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.transforms import normalize_indices
from gpytorch.distributions.multitask_multivariate_normal import (
    MultitaskMultivariateNormal,
)
from gpytorch.lazy.block_diag_lazy_tensor import BlockDiagLazyTensor
from gpytorch.lazy.lazy_tensor import LazyTensor
from gpytorch.utils.cholesky import psd_safe_cholesky
from gpytorch.utils.errors import NanError
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

    return X[idcs]


def extract_batch_covar(mt_mvn: MultitaskMultivariateNormal) -> LazyTensor:
    r"""Extract a batched independent covariance matrix from a MTMVN.

    Args:
        mt_mvn: A multi-task multivariate normal with a block diagonal
            covariance matrix.

    Returns:
        A lazy covariance matrix consisting of a batch of the blocks of the
            diagonal of the MultitaskMultivariateNormal.

    """
    lazy_covar = mt_mvn.lazy_covariance_matrix
    if not isinstance(lazy_covar, BlockDiagLazyTensor):
        raise BotorchError(f"Expected BlockDiagLazyTensor, but got {type(lazy_covar)}.")
    return lazy_covar.base_lazy_tensor


def _reshape_base_samples(
    base_samples: Tensor, sample_shape: torch.Size, posterior: GPyTorchPosterior
) -> Tensor:
    r"""Manipulate shape of base_samples to match `MultivariateNormal.rsample`.

    This ensure that base_samples are used in the same way as in
    gpytorch.distributions.MultivariateNormal. For CBD, it is important to ensure
    that the same base samples are used for the in-sample points here and in the
    cached box decompositions.

    Args:
        base_samples: The base samples.
        sample_shape: The sample shape.
        posterior: The joint posterior is over (X_baseline, X).

    Returns:
        Reshaped and expanded base samples.
    """
    loc = posterior.mvn.loc
    peshape = posterior.event_shape
    base_samples = base_samples.view(
        sample_shape + torch.Size([1 for _ in range(loc.ndim - 1)]) + peshape[-2:]
    ).expand(sample_shape + loc.shape[:-1] + peshape[-2:])
    base_samples = base_samples.reshape(
        -1, *loc.shape[:-1], posterior.mvn.lazy_covariance_matrix.shape[-1]
    )
    base_samples = base_samples.permute(*range(1, loc.dim() + 1), 0)
    return base_samples.reshape(
        *peshape[:-2],
        peshape[-1],
        peshape[-2],
        *sample_shape,
    )


def sample_cached_cholesky(
    posterior: GPyTorchPosterior,
    baseline_L: Tensor,
    q: int,
    base_samples: Tensor,
    sample_shape: torch.Size,
    max_tries: int = 6,
) -> Tensor:
    r"""Get posterior samples at the `q` new points from the joint multi-output posterior.

    TODO: support single output posteriors.

    Args:
        posterior: The joint posterior is over (X_baseline, X).
        baseline_L: The baseline lower triangular cholesky factor.
        q: The number of new points in X.
        base_samples: The base samples.
        sample_shape: The sample shape.
        max_tries: The number of tries for computing the Cholesky
            decomposition with increasing jitter.

    Returns:
        A `sample_shape x batch_shape x q x m`-dim tensor of posterior
            samples at the new points.
    """
    # compute bottom left covariance block
    if isinstance(posterior.mvn, MultitaskMultivariateNormal):
        lazy_covar = extract_batch_covar(mt_mvn=posterior.mvn)
    else:
        raise NotImplementedError(
            "Single-output MultivariateNormal distributions are "
            "not currently supported."
        )
    # Get the `q` new rows of the batched covariance matrix
    bottom_rows = lazy_covar[..., -q:, :].evaluate()
    # The covariance in block form is:
    # [K(X_baseline, X_baseline), K(X_baseline, X)]
    # [K(X, X_baseline), K(X, X)]
    # bl := K(X, X_baseline)
    # br := K(X, X)
    # Get bottom right block of new covariance
    bl, br = torch.split(bottom_rows, bottom_rows.shape[-1] - q, dim=-1)
    # Solve Ax = b
    # where A = K(X_baseline, X_baseline) and b = K(X, X_baseline)^T
    # and bl_chol := x^T
    # bl_chol is the new `(batch_shape) x q x n`-dim bottom left block
    # of the cholesky decomposition
    bl_chol = torch.triangular_solve(
        bl.transpose(-2, -1), baseline_L, upper=False
    ).solution.transpose(-2, -1)
    # Compute the new bottom right block of the Cholesky decomposition via:
    # Cholesky(K(X, X) - bl_chol @ bl_chol^T)
    br_to_chol = br - bl_chol @ bl_chol.transpose(-2, -1)
    # TODO: technically we should make sure that we add a consistent
    # nugget to the cached covariance and the new block
    br_chol = psd_safe_cholesky(br_to_chol, max_tries=max_tries)
    # Create a `(batch_shape) x q x (n+q)`-dim tensor containing the
    # `q` new bottom rows of the Cholesky decomposition
    new_Lq = torch.cat([bl_chol, br_chol], dim=-1)
    mean = posterior.mvn.mean
    base_samples = _reshape_base_samples(
        base_samples=base_samples, sample_shape=sample_shape, posterior=posterior
    )
    new_mean = mean[..., -q:, :]
    res = (
        new_Lq.matmul(base_samples)
        .permute(-1, *range(mean.dim() - 2), -2, -3)
        .contiguous()
        .add(new_mean)
    )
    contains_nans = torch.isnan(res).any()
    contains_infs = torch.isinf(res).any()
    if contains_nans or contains_infs:
        suffix_args = []
        if contains_nans:
            suffix_args.append("nans")
        if contains_infs:
            suffix_args.append("infs")
        suffix = " and ".join(suffix_args)
        raise NanError(f"Samples contain {suffix}.")
    return res
