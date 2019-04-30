#!/usr/bin/env python3

import typing  # noqa F401
import warnings

import torch
from torch import Tensor

from ..exceptions.warnings import BadInitialCandidatesWarning


def initialize_q_batch(X: Tensor, Y: Tensor, n: int, eta: float = 1.0) -> Tensor:
    r"""Heuristic for selecting initial conditions for candidate generation.

    This heuristic selects points from `X` (without replacement) with probability
    proportional to `exp(eta * Z)`, where `Z = (Y - mean(Y)) / std(Y)` and `eta`
    is a temperature parameter.

    When using an acquisiton function that is non-negative and possibly zero
    over large areas of the feature space (e.g. qEI), you should use
    `initialize_q_batch_nonneg` instead.

    Args:
        X: A `b x q x d` tensor of `b` samples of `q`-batches from a `d`-dim.
            feature space. Typically, these are generated using qMC sampling.
        Y: A tensor of `b` outcomes associated with the samples. Typically, this
            is the value of the batch acquisition function to be maximized.
        n: The number of initial condition to be generated. Must be less than `b`.
        eta: Temperature parameter for weighting samples.

    Returns:
        A `n x q x d` tensor of `n` `q`-batch initial conditions.

    Example:
        # To get `n=10` starting points of q-batch size `q=3` for model with `d=6`:
        >>> qUCB = qUpperConfidenceBound(model, beta=0.1)
        >>> Xrnd = torch.rand(500, 3, 6)
        >>> Xinit = initialize_q_batch(Xrnd, qUCB(Xrnd), 10)
    """
    n_samples = X.shape[0]
    if n > n_samples:
        raise RuntimeError(
            f"n ({n}) cannot be larger than the number of "
            f"provided samples ({n_samples})"
        )
    elif n == n_samples:
        return X

    Ystd = Y.std()
    if Ystd == 0:
        warnings.warn(
            "All acqusition values for raw samples points are the same. "
            "Choosing initial conditions at random.",
            BadInitialCandidatesWarning,
        )
        return X[torch.randperm(n=n_samples, device=X.device)][:n]

    max_val, max_idx = torch.max(Y, dim=0)
    Z = Y - Y.mean() / Ystd
    weights = torch.exp(eta * Z)
    idcs = torch.multinomial(weights, n)
    # make sure we get the maximum
    if max_idx not in idcs:
        idcs[-1] = max_idx
    return X[idcs]


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
        # To get `n=10` starting points of q-batch size `q=3` for model with `d=6`:
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
        remaining_indices = (~pos).nonzero().view(-1)
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
