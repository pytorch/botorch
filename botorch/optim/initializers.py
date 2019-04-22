#!/usr/bin/env python3

from typing import Dict, Optional, Union
from warnings import warn

import torch
from torch import Tensor

from ..exceptions.warnings import BadInitialCandidatesWarning


def initialize_q_batch(
    X: Tensor,
    Y: Tensor,
    n: int,
    options: Optional[Dict[str, Union[bool, float]]] = None,
) -> Tensor:
    r"""Heuristic for picking initial candidates for candidate generation.

    Args:
        X: A `b x q x d` tensor of `b` samples of `q`-batches from a `d`-dim.
            feature space. Typically, these are generated using qMC.
        Y: A tensor of `b` outcomes associated with the samples. Typically, this
            is the value of the batch acquisition function to be maximized.
        n: The number of initial condition to be generated. Must be smaller than `b`.
        options: A dictionary for specifying options:
            - alpha: The threshold (as a fraction of the maximum observed value)
            under which to ignore samples. All samples for which `Y < alpha max(Y)`
            will be ignored. Default: 1e-5
            - eta: Temperature parameter for weighting samples. Default: 1.0.
            If `eta == 0`, any non-zero function values are equally likely to be
            selected.

    Returns:
        A `n x q x d` tensor of `n` `q`-batch initial conditions.
    """
    options = options or {}
    alpha = options.get("alpha", 1e-4)
    eta = options.get("eta", 1.0)
    n_samples = X.shape[0]

    if n > n_samples:
        raise RuntimeError("n cannot be larger than the number of provided samples")
    elif n == n_samples:
        return X

    max_val, max_idx = torch.max(Y, dim=0)
    if torch.any(max_val <= 0):
        warn(
            "All acquisition values for raw sampled points are nonpositive, so "
            "initial conditions are being selected randomly.",
            BadInitialCandidatesWarning,
        )
        return X[torch.randperm(n=n_samples, device=X.device)][:n]

    # make sure there are at least `n` points with positive acquisition values
    positive = Y > 0
    num_positive = positive.sum()
    if num_positive < n:
        # select all positive points and then fill remaining quota with randomly
        # selected points
        remaining_indices = (~positive).nonzero().view(-1)
        rand_indices = torch.randperm(remaining_indices.shape[0], device=Y.device)
        sampled_remaining_indices = remaining_indices[rand_indices[: n - num_positive]]
        positive[sampled_remaining_indices] = 1
        return X[positive]
    # select points within alpha of max_val, iteratively decreasing alpha by a
    # factor of 10 as necessary
    alpha_positive = Y >= alpha * max_val
    while alpha_positive.sum() < n:
        alpha = 0.1 * alpha
        alpha_positive = Y >= alpha * max_val
    alpha_positive_idcs = torch.arange(len(Y), device=Y.device)[alpha_positive]
    weights = torch.exp(eta * (Y[alpha_positive] / max_val - 1))
    idcs = alpha_positive_idcs[torch.multinomial(weights, n)]
    if max_idx not in idcs:
        idcs[-1] = max_idx
    return X[idcs]
