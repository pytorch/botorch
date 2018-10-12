#!/usr/bin/env python3

from typing import Callable

import torch
from gpytorch import Module
from torch import Tensor


def q_batch_initialization(
    model: Module,
    gen_function: Callable[[int], Tensor],
    q: int,
    utility: Callable[[Tensor, Module], Tensor],
    multiplier: int = 50,
) -> Tensor:
    """Generate initial points for batch optimiziation.

    Replicates the procedure from Wilson, et al. (Section B.1)

    Args:
        model: A fitted GP model
        gen_function: A function n -> X_cand producing n (typically random)
            feasible candidates as a `n x d` tensor X_cand
        q: The number of candidates in each q-batch
        utility: A callable (X_cand, model) -> U computing the utility of each
            candidate in X_cand under the provided model.
        multiplier: This factor determines how many points to generate from
            gen_function for each q-batch (for a total of q * multiplier points)

    Returns:
        The set of candidates with the highest value under the utility
    """
    # TODO: There is a better heuristic for this in AE, adopt that
    bulk_X = gen_function(q * multiplier)
    _, best_indices = torch.topk(-utility(bulk_X, model), k=q)
    return bulk_X[best_indices]
