#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import warnings
from typing import Dict, Optional, Union

import torch
from botorch import settings
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.knowledge_gradient import (
    _get_value_function,
    qKnowledgeGradient,
)
from botorch.acquisition.utils import is_nonnegative
from botorch.exceptions.warnings import BadInitialCandidatesWarning, SamplingWarning
from botorch.models.model import Model
from botorch.utils.sampling import batched_multinomial, draw_sobol_samples, manual_seed
from botorch.utils.transforms import standardize
from torch import Tensor
from torch.quasirandom import SobolEngine


def gen_batch_initial_conditions(
    acq_function: AcquisitionFunction,
    bounds: Tensor,
    q: int,
    num_restarts: int,
    raw_samples: int,
    options: Optional[Dict[str, Union[bool, float, int]]] = None,
) -> Tensor:
    r"""Generate a batch of initial conditions for random-restart optimziation.

    Args:
        acq_function: The acquisition function to be optimized.
        bounds: A `2 x d` tensor of lower and upper bounds for each column of `X`.
        q: The number of candidates to consider.
        num_restarts: The number of starting points for multistart acquisition
            function optimization.
        raw_samples: The number of raw samples to consider in the initialization
            heuristic.
        options: Options for initial condition generation. For valid options see
            `initialize_q_batch` and `initialize_q_batch_nonneg`. If `options`
            contains a `nonnegative=True` entry, then `acq_function` is
            assumed to be non-negative (useful when using custom acquisition
            functions). In addition, an "init_batch_limit" option can be passed
            to specify the batch limit for the initialization. This is useful
            for avoiding memory limits when computing the batch posterior over
            raw samples.

    Returns:
        A `num_restarts x q x d` tensor of initial conditions.

    Example:
        >>> qEI = qExpectedImprovement(model, best_f=0.2)
        >>> bounds = torch.tensor([[0.], [1.]])
        >>> Xinit = gen_batch_initial_conditions(
        >>>     qEI, bounds, q=3, num_restarts=25, raw_samples=500
        >>> )
    """
    options = options or {}
    seed: Optional[int] = options.get("seed")
    batch_limit: Optional[int] = options.get(
        "init_batch_limit", options.get("batch_limit")
    )
    batch_initial_arms: Tensor
    factor, max_factor = 1, 5
    init_kwargs = {}
    device = bounds.device
    bounds = bounds.cpu()
    if "eta" in options:
        init_kwargs["eta"] = options.get("eta")
    if options.get("nonnegative") or is_nonnegative(acq_function):
        init_func = initialize_q_batch_nonneg
        if "alpha" in options:
            init_kwargs["alpha"] = options.get("alpha")
    else:
        init_func = initialize_q_batch

    q = 1 if q is None else q
    # the dimension the samples are drawn from
    effective_dim = bounds.shape[-1] * q
    if effective_dim > SobolEngine.MAXDIM and settings.debug.on():
        warnings.warn(
            f"Sample dimension q*d={effective_dim} exceeding Sobol max dimension "
            f"({SobolEngine.MAXDIM}). Using iid samples instead.",
            SamplingWarning,
        )

    while factor < max_factor:
        with warnings.catch_warnings(record=True) as ws:
            n = raw_samples * factor
            if effective_dim <= SobolEngine.MAXDIM:
                X_rnd = draw_sobol_samples(bounds=bounds, n=n, q=q, seed=seed)
            else:
                with manual_seed(seed):
                    # load on cpu
                    X_rnd_nlzd = torch.rand(n * effective_dim, dtype=bounds.dtype)
                    X_rnd_nlzd = X_rnd_nlzd.view(n, q, bounds.shape[-1])
                X_rnd = bounds[0] + (bounds[1] - bounds[0]) * X_rnd_nlzd
            with torch.no_grad():
                if batch_limit is None:
                    batch_limit = X_rnd.shape[0]
                Y_rnd_list = []
                start_idx = 0
                while start_idx < X_rnd.shape[0]:
                    end_idx = min(start_idx + batch_limit, X_rnd.shape[0])
                    Y_rnd_curr = acq_function(
                        X_rnd[start_idx:end_idx].to(device=device)
                    ).cpu()
                    Y_rnd_list.append(Y_rnd_curr)
                    start_idx += batch_limit
                Y_rnd = torch.cat(Y_rnd_list)
            batch_initial_conditions = init_func(
                X=X_rnd, Y=Y_rnd, n=num_restarts, **init_kwargs
            ).to(device=device)
            if not any(issubclass(w.category, BadInitialCandidatesWarning) for w in ws):
                return batch_initial_conditions
            if factor < max_factor:
                factor += 1
                if seed is not None:
                    seed += 1  # make sure to sample different X_rnd
    warnings.warn(
        "Unable to find non-zero acquisition function values - initial conditions "
        "are being selected randomly.",
        BadInitialCandidatesWarning,
    )
    return batch_initial_conditions


def gen_one_shot_kg_initial_conditions(
    acq_function: qKnowledgeGradient,
    bounds: Tensor,
    q: int,
    num_restarts: int,
    raw_samples: int,
    options: Optional[Dict[str, Union[bool, float, int]]] = None,
) -> Optional[Tensor]:
    r"""Generate a batch of smart initializations for qKnowledgeGradient.

    This function generates initial conditions for optimizing one-shot KG using
    the maximizer of the posterior objective. Intutively, the maximizer of the
    fantasized posterior will often be close to a maximizer of the current
    posterior. This function uses that fact to generate the initital conditions
    for the fantasy points. Specifically, a fraction of `1 - frac_random` (see
    options) is generated by sampling from the set of maximizers of the
    posterior objective (obtained via random restart optimization) according to
    a softmax transformation of their respective values. This means that this
    initialization strategy internally solves an acquisition function
    maximization problem. The remaining `frac_random` fantasy points as well as
    all `q` candidate points are chosen according to the standard initialization
    strategy in `gen_batch_initial_conditions`.

    Args:
        acq_function: The qKnowledgeGradient instance to be optimized.
        bounds: A `2 x d` tensor of lower and upper bounds for each column of
            task features.
        q: The number of candidates to consider.
        num_restarts: The number of starting points for multistart acquisition
            function optimization.
        raw_samples: The number of raw samples to consider in the initialization
            heuristic.
        options: Options for initial condition generation. These contain all
            settings for the standard heuristic initialization from
            `gen_batch_initial_conditions`. In addition, they contain
            `frac_random` (the fraction of fully random fantasy points),
            `num_inner_restarts` and `raw_inner_samples` (the number of random
            restarts and raw samples for solving the posterior objective
            maximization problem, respectively) and `eta` (temperature parameter
            for sampling heuristic from posterior objective maximizers).

    Returns:
        A `num_restarts x q' x d` tensor that can be used as initial conditions
        for `optimize_acqf()`. Here `q' = q + num_fantasies` is the total number
        of points (candidate points plus fantasy points).

    Example:
        >>> qKG = qKnowledgeGradient(model, num_fantasies=64)
        >>> bounds = torch.tensor([[0., 0.], [1., 1.]])
        >>> Xinit = gen_one_shot_kg_initial_conditions(
        >>>     qKG, bounds, q=3, num_restarts=10, raw_samples=512,
        >>>     options={"frac_random": 0.25},
        >>> )
    """
    options = options or {}
    frac_random: float = options.get("frac_random", 0.1)
    if not 0 < frac_random < 1:
        raise ValueError(
            f"frac_random must take on values in (0,1). Value: {frac_random}"
        )
    q_aug = acq_function.get_augmented_q_batch_size(q=q)

    # TODO: Avoid unnecessary computation by not generating all candidates
    ics = gen_batch_initial_conditions(
        acq_function=acq_function,
        bounds=bounds,
        q=q_aug,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        options=options,
    )

    # compute maximizer of the value function
    value_function = _get_value_function(
        model=acq_function.model,
        objective=acq_function.objective,
        sampler=acq_function.inner_sampler,
        project=getattr(acq_function, "project", None),
    )
    from botorch.optim.optimize import optimize_acqf

    fantasy_cands, fantasy_vals = optimize_acqf(
        acq_function=value_function,
        bounds=bounds,
        q=1,
        num_restarts=options.get("num_inner_restarts", 20),
        raw_samples=options.get("raw_inner_samples", 1024),
        return_best_only=False,
    )

    # sampling from the optimizers
    n_value = int((1 - frac_random) * (q_aug - q))  # number of non-random ICs
    eta = options.get("eta", 2.0)
    weights = torch.exp(eta * standardize(fantasy_vals))
    idx = torch.multinomial(weights, num_restarts * n_value, replacement=True)

    # set the respective initial conditions to the sampled optimizers
    ics[..., -n_value:, :] = fantasy_cands[idx, 0].view(num_restarts, n_value, -1)
    return ics


def gen_value_function_initial_conditions(
    acq_function: AcquisitionFunction,
    bounds: Tensor,
    num_restarts: int,
    raw_samples: int,
    current_model: Model,
    options: Optional[Dict[str, Union[bool, float, int]]] = None,
) -> Tensor:
    r"""Generate a batch of smart initializations for optimizing
    the value function of qKnowledgeGradient.

    This function generates initial conditions for optimizing the inner problem of
    KG, i.e. its value function, using the maximizer of the posterior objective.
    Intutively, the maximizer of the fantasized posterior will often be close to a
    maximizer of the current posterior. This function uses that fact to generate the
    initital conditions for the fantasy points. Specifically, a fraction of `1 -
    frac_random` (see options) of raw samples is generated by sampling from the set of
    maximizers of the posterior objective (obtained via random restart optimization)
    according to a softmax transformation of their respective values. This means that
    this initialization strategy internally solves an acquisition function
    maximization problem. The remaining raw samples are generated using
    `draw_sobol_samples`. All raw samples are then evaluated, and the initial
    conditions are selected according to the standard initialization strategy in
    'initialize_q_batch' individually for each inner problem.

    Args:
        acq_function: The value function instance to be optimized.
        bounds: A `2 x d` tensor of lower and upper bounds for each column of
            task features.
        num_restarts: The number of starting points for multistart acquisition
            function optimization.
        raw_samples: The number of raw samples to consider in the initialization
            heuristic.
        current_model: The model of the KG acquisition function that was used to
            generate the fantasy model of the value function.
        options: Options for initial condition generation. These contain all
            settings for the standard heuristic initialization from
            `gen_batch_initial_conditions`. In addition, they contain
            `frac_random` (the fraction of fully random fantasy points),
            `num_inner_restarts` and `raw_inner_samples` (the number of random
            restarts and raw samples for solving the posterior objective
            maximization problem, respectively) and `eta` (temperature parameter
            for sampling heuristic from posterior objective maximizers).

    Returns:
        A `num_restarts x batch_shape x q x d` tensor that can be used as initial
        conditions for `optimize_acqf()`. Here `batch_shape` is the batch shape
        of value function model.

    Example:
        >>> fant_X = torch.rand(5, 1, 2)
        >>> fantasy_model = model.fantasize(fant_X, SobolQMCNormalSampler(16))
        >>> value_function = PosteriorMean(fantasy_model)
        >>> bounds = torch.tensor([[0., 0.], [1., 1.]])
        >>> Xinit = gen_value_function_initial_conditions(
        >>>     value_function, bounds, num_restarts=10, raw_samples=512,
        >>>     options={"frac_random": 0.25},
        >>> )
    """
    options = options or {}
    seed: Optional[int] = options.get("seed")
    frac_random: float = options.get("frac_random", 0.6)
    if not 0 < frac_random < 1:
        raise ValueError(
            f"frac_random must take on values in (0,1). Value: {frac_random}"
        )

    # compute maximizer of the current value function
    value_function = _get_value_function(
        model=current_model,
        objective=acq_function.objective,
        sampler=getattr(acq_function, "sampler", None),
        project=getattr(acq_function, "project", None),
    )
    from botorch.optim.optimize import optimize_acqf

    fantasy_cands, fantasy_vals = optimize_acqf(
        acq_function=value_function,
        bounds=bounds,
        q=1,
        num_restarts=options.get("num_inner_restarts", 20),
        raw_samples=options.get("raw_inner_samples", 1024),
        return_best_only=False,
        options={
            k: v
            for k, v in options.items()
            if k
            not in ("frac_random", "num_inner_restarts", "raw_inner_samples", "eta")
        },
    )

    batch_shape = acq_function.model.batch_shape
    # sampling from the optimizers
    n_value = int((1 - frac_random) * raw_samples)  # number of non-random ICs
    if n_value > 0:
        eta = options.get("eta", 2.0)
        weights = torch.exp(eta * standardize(fantasy_vals))
        idx = batched_multinomial(
            weights=weights.expand(*batch_shape, -1),
            num_samples=n_value,
            replacement=True,
        ).permute(-1, *range(len(batch_shape)))
        resampled = fantasy_cands[idx]
    else:
        resampled = torch.empty(
            0, *batch_shape, 1, bounds.shape[-1], dtype=bounds.dtype
        )
    # add qMC samples
    randomized = draw_sobol_samples(
        bounds=bounds, n=raw_samples - n_value, q=1, batch_shape=batch_shape, seed=seed
    )
    # full set of raw samples
    X_rnd = torch.cat([resampled, randomized], dim=0)

    # evaluate the raw samples
    with torch.no_grad():
        Y_rnd = acq_function(X_rnd)

    # select the restart points using the heuristic
    return initialize_q_batch(
        X=X_rnd, Y=Y_rnd, n=num_restarts, eta=options.get("eta", 2.0)
    )


def initialize_q_batch(X: Tensor, Y: Tensor, n: int, eta: float = 1.0) -> Tensor:
    r"""Heuristic for selecting initial conditions for candidate generation.

    This heuristic selects points from `X` (without replacement) with probability
    proportional to `exp(eta * Z)`, where `Z = (Y - mean(Y)) / std(Y)` and `eta`
    is a temperature parameter.

    When using an acquisiton function that is non-negative and possibly zero
    over large areas of the feature space (e.g. qEI), you should use
    `initialize_q_batch_nonneg` instead.

    Args:
        X: A `b x batch_shape x q x d` tensor of `b` - `batch_shape` samples of
            `q`-batches from a d`-dim feature space. Typically, these are generated
            using qMC sampling.
        Y: A tensor of `b x batch_shape` outcomes associated with the samples.
            Typically, this is the value of the batch acquisition function to be
            maximized.
        n: The number of initial condition to be generated. Must be less than `b`.
        eta: Temperature parameter for weighting samples.

    Returns:
        A `n x batch_shape x q x d` tensor of `n` - `batch_shape` `q`-batch initial
        conditions, where each batch of `n x q x d` samples is selected independently.

    Example:
        >>> # To get `n=10` starting points of q-batch size `q=3`
        >>> # for model with `d=6`:
        >>> qUCB = qUpperConfidenceBound(model, beta=0.1)
        >>> Xrnd = torch.rand(500, 3, 6)
        >>> Xinit = initialize_q_batch(Xrnd, qUCB(Xrnd), 10)
    """
    n_samples = X.shape[0]
    batch_shape = X.shape[1:-2] or torch.Size()
    if n > n_samples:
        raise RuntimeError(
            f"n ({n}) cannot be larger than the number of "
            f"provided samples ({n_samples})"
        )
    elif n == n_samples:
        return X

    Ystd = Y.std(dim=0)
    if torch.any(Ystd == 0):
        warnings.warn(
            "All acquisition values for raw samples points are the same for "
            "at least one batch. Choosing initial conditions at random.",
            BadInitialCandidatesWarning,
        )
        return X[torch.randperm(n=n_samples, device=X.device)][:n]

    max_val, max_idx = torch.max(Y, dim=0)
    Z = (Y - Y.mean(dim=0)) / Ystd
    etaZ = eta * Z
    weights = torch.exp(etaZ)
    while torch.isinf(weights).any():
        etaZ *= 0.5
        weights = torch.exp(etaZ)
    if batch_shape == torch.Size():
        idcs = torch.multinomial(weights, n)
    else:
        idcs = batched_multinomial(
            weights=weights.permute(*range(1, len(batch_shape) + 1), 0), num_samples=n
        ).permute(-1, *range(len(batch_shape)))
    # make sure we get the maximum
    if max_idx not in idcs:
        idcs[-1] = max_idx
    if batch_shape == torch.Size():
        return X[idcs]
    else:
        return X.gather(
            dim=0, index=idcs.view(*idcs.shape, 1, 1).expand(n, *X.shape[1:])
        )


def initialize_q_batch_nonneg(
    X: Tensor, Y: Tensor, n: int, eta: float = 1.0, alpha: float = 1e-4
) -> Tensor:
    r"""Heuristic for selecting initial conditions for non-neg. acquisition functions.

    This function is similar to `initialize_q_batch`, but designed specifically
    for acquisition functions that are non-negative and possibly zero over
    large areas of the feature space (e.g. qEI). All samples for which
    `Y < alpha * max(Y)` will be ignored (assuming that `Y` contains at least
    one positive value).

    Args:
        X: A `b x q x d` tensor of `b` samples of `q`-batches from a `d`-dim.
            feature space. Typically, these are generated using qMC.
        Y: A tensor of `b` outcomes associated with the samples. Typically, this
            is the value of the batch acquisition function to be maximized.
        n: The number of initial condition to be generated. Must be less than `b`.
        eta: Temperature parameter for weighting samples.
        alpha: The threshold (as a fraction of the maximum observed value) under
            which to ignore samples. All input samples for which
            `Y < alpha * max(Y)` will be ignored.

    Returns:
        A `n x q x d` tensor of `n` `q`-batch initial conditions.

    Example:
        >>> # To get `n=10` starting points of q-batch size `q=3`
        >>> # for model with `d=6`:
        >>> qEI = qExpectedImprovement(model, best_f=0.2)
        >>> Xrnd = torch.rand(500, 3, 6)
        >>> Xinit = initialize_q_batch(Xrnd, qEI(Xrnd), 10)
    """
    n_samples = X.shape[0]
    if n > n_samples:
        raise RuntimeError("n cannot be larger than the number of provided samples")
    elif n == n_samples:
        return X

    max_val, max_idx = torch.max(Y, dim=0)
    if torch.any(max_val <= 0):
        warnings.warn(
            "All acquisition values for raw sampled points are nonpositive, so "
            "initial conditions are being selected randomly.",
            BadInitialCandidatesWarning,
        )
        return X[torch.randperm(n=n_samples, device=X.device)][:n]

    # make sure there are at least `n` points with positive acquisition values
    pos = Y > 0
    num_pos = pos.sum().item()
    if num_pos < n:
        # select all positive points and then fill remaining quota with randomly
        # selected points
        remaining_indices = (~pos).nonzero(as_tuple=False).view(-1)
        rand_indices = torch.randperm(remaining_indices.shape[0], device=Y.device)
        sampled_remaining_indices = remaining_indices[rand_indices[: n - num_pos]]
        pos[sampled_remaining_indices] = 1
        return X[pos]
    # select points within alpha of max_val, iteratively decreasing alpha by a
    # factor of 10 as necessary
    alpha_pos = Y >= alpha * max_val
    while alpha_pos.sum() < n:
        alpha = 0.1 * alpha
        alpha_pos = Y >= alpha * max_val
    alpha_pos_idcs = torch.arange(len(Y), device=Y.device)[alpha_pos]
    weights = torch.exp(eta * (Y[alpha_pos] / max_val - 1))
    idcs = alpha_pos_idcs[torch.multinomial(weights, n)]
    if max_idx not in idcs:
        idcs[-1] = max_idx
    return X[idcs]
