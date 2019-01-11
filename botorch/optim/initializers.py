#!/usr/bin/env python3

import math
from itertools import permutations
from typing import Callable, Dict, Optional, Union

import numpy as np
import torch
from torch import Tensor


def initialize_q_batch(
    X: Tensor,
    Y: Tensor,
    n: int,
    sim_measure: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
    options: Optional[Dict[str, Union[bool, float]]] = None,
) -> Tensor:
    """Heuristic for picking initial candidates for candidate generation.

    Picks initial (q-batch) conditions by trading off high function values `Y`
    with a uniform distribution of the candidates in the feature space.
    The latter is characterized by a permutation-invariant similarity measure
    computed from the pair-wise similarity between points (as measured by
    sim_measure, if provided, otherwise by the euclidean distance).
    `eta_Y` and `eta_sim` are temperature parameters:
        - As `eta_Y -> inf`, the heuristic becomes greedy w.r.t. the outcome
            values. If `eta_Y == 0`, outcome values are ignored.
        - As `eta_sim -> inf`, the heuristic becomes greedy w.r.t. "diversity",
            i.e. favoring points whose similarity to previous points is small.
            If `eta_sim == 0`, similarity between points is ignored (this is
            typically much faster).

    Args:
        X: A `b x q x d` tensor of `b` samples of `q`-batches from a `d`-dim.
            feature space. Typically, these are generated using qMC.
        Y: A tensor of `b` outcomes associated with the samples. Typically, this
            is the value of the batch acquisition function to be maximized.
        n: The number of initial condition to be generated. Must be smaller than `b`.
        sim_measure: A callable `(X, x) -> C`, where `X`, `x`, and `C` are Tensors
            of size `b x q x d`, `q x d`, and `n_samples x q x q`, respectively,
            where `C[k, i, j]` characterizes the similiarity betweem the points
            `X[k, i]` and `x[j]`. This is used to trade off an uniform spread of
            points (w.r.t. the similarity measure) and a high function value.
            Typically, this is based on a gpytorch covar_module, e.g.,
            `sim_measure = lambda X, x: covar_module(X, x).evaluate()`.
            If not provided, use `1 / (1 + gamma * ||x - x'||_2)`, where `gamma`
            is defined in options. Default: 1.0
        options: A dictionary for specifying options:
            - eta_Y: The temperature parameter for the outcomes Y. Default: 2.0
            - eta_sim: The temperature parameter for the (negative) covariances.
                Default: 2.0. Note: If `eta_sim == 0`, no distances will be
                computed, and points will be selected only based on their
                function values `Y`.
            - max_perms: Maximum number of permutataions to use for computing
                the similarity measure (the maximum of the sum of point-wise
                covariances over all permutations of the `q` points). Complexity
                (space and time) is linear in max_perms. If `max_perms > q!`,
                this function computes the maximum across all possible `q!`
                permutations, otherwise it uses `max_perms` random permutations.
                Default: `4! = 24`
            - gamma: The parameter in 1 / (1 + gamma * ||x - x'||_2)` used as
                similarity measure in cas sim_measure is not provided.

    Returns:
        A `n x q x d` tensor of `n` `q`-batch initial conditions.

    """
    options = options or {}
    eta_Y = options.get("eta_Y", 2.0)
    eta_sim = options.get("eta_sim", 2.0)
    gamma = options.get("gamma", 1.0)
    max_perms = options.get("max_perms", 24)
    n_samples, q = X.shape[0], X.shape[1]

    if n > n_samples:
        raise RuntimeError("n cannot be larger than the number of provided samples")
    elif n == n_samples:
        return X

    similarity: Callable[[Tensor, Tensor], Tensor]
    if sim_measure is not None:
        similarity = sim_measure
    else:
        # 1 / (1 + euclidean distance)
        def similarity(X: Tensor, x: Tensor) -> Tensor:
            distance = torch.norm(X.unsqueeze(-2) - x, p=2, dim=-1)
            return (1 + gamma * distance).reciprocal()

    # push value through exp (w/ temperature) to get probabilities for sampling
    Y_std = Y.std()
    if Y_std > 0:
        prob_vals = torch.exp(eta_Y * (Y - Y.mean()) / Y_std)
    else:
        prob_vals = torch.ones_like(Y)
    prob_vals /= prob_vals.mean()  # for numerical stabilty

    if eta_sim == 0:
        # similarity is irrelevant, just sample from the prob_vals directly
        indcs = torch.multinomial(prob_vals / prob_vals.sum(), n, replacement=False)
        return X[indcs]

    arng = torch.arange(q, device=X.device)  # indexing helper
    available = torch.ones(n_samples, dtype=torch.uint8, device=X.device)

    # all permutations of q points, used to compute a measure of similarity
    # between two q-batches of points in X that is permutation-invariant (w.r.t.
    # the order of the q elements)
    total_perms = math.factorial(q)
    all_perms = permutations(range(q), q)
    if max_perms >= total_perms:
        # evaluate all permutations
        perms = list(all_perms)
    else:
        # evaluate a random subset of permutations
        indcs = np.random.randint(total_perms, size=max_perms)
        # note that all_perms is a generator, so we never construct the full
        # set of permutations in memory
        perms = [perm for i, perm in enumerate(all_perms) if i in indcs]
    perms = torch.tensor(perms, device=X.device)

    # store the points we pick
    xs = []

    # for the first index we just pick the globally best sample
    max_idx = Y.argmax().item()
    available[max_idx] = 0
    xs.append(X[max_idx])

    # keep track of the maximum similarity measure between points seen so far
    max_sims = torch.full((n_samples,), -float("inf"), device=X.device, dtype=X.dtype)

    # loop until we have the requested number of points
    with torch.no_grad():
        for _ in range(n - 1):
            # compute covariance between current pick and all other samples
            # this will depend on the order of the points in the q-batch
            loc_sims = similarity(X, xs[-1])
            # TODO (T38994517): Investigate whether eff. complexity here is only q^2
            sims_raw = torch.stack(
                [loc_sims[..., arng, perm].sum(-1) for perm in perms], -1
            ).max(-1)[0]
            # standardize similarities
            sims_raw_std = sims_raw.std()
            if sims_raw_std > 0:
                sims = (sims_raw - sims_raw.mean()) / sims_raw_std
            else:
                sims = sims_raw - sims_raw.mean()
            max_sims = torch.stack([max_sims, sims], 0).max(0)[0]

            # weights from the similarity measure (less similar points -> higher weight)
            sim_weights = torch.exp(-eta_sim * (max_sims - max_sims.mean()))
            sim_weights /= sim_weights.mean()

            # probabilities according to which we sample from X
            prob = prob_vals * sim_weights
            prob[~available] = 0
            prob /= prob.sum()

            # pick the next candidate
            max_idx = torch.multinomial(prob, 1).item()
            available[max_idx] = 0
            xs.append(X[max_idx])

    return torch.stack(xs)
