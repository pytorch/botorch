#!/usr/bin/env python3

from typing import Callable

import torch
from gpytorch import Module
from torch import Tensor


def q_batch_initialization(
    gen_function: Callable[[int], Tensor],
    acq_function: Module,
    q: int,
    multiplier: int = 50,
    torch_batches: int = 1,
) -> Tensor:
    """Generate initial points for batch optimiziation.

    Args:
        gen_function: A function n -> X_cand producing n (typically random)
            feasible candidates as a `n x d` tensor X_cand
        acq_function:  An acquisition function Module assumed
            to support the given value of q that returns
            a Tensor when evaluated at a given q-batch
        q: The number of candidates in each q-batch
        multiplier: This factor determines how many q-batches
            to generate from gen_function
        torch_batches:  This factor determines how many q-batches
            in total will be returned by the function.  Must
            be less than multiplier.

    Returns:
        A Tensor X of size torch_batches x q x d where
        X[i,:] has one of the torch_batches highest values of
        acq_function(X[i,:])
    """
    # TODO: remove cache clearing once upstream issues regarding non-batch
    # evaluation followed by batch evaluation are resolved T36825603
    # clear caches
    acq_function.model.train()
    acq_function.model.eval()
    bulk_X = torch.cat([gen_function(q).unsqueeze(0) for i in range(multiplier)], dim=0)
    # TODO: bulk_X is multiplier x q x d. Replace below
    # when acq_functions all support t-batches
    val_X = torch.cat(
        [acq_function(bulk_X[i, ...]).reshape(1) for i in range(bulk_X.shape[0])]
    )
    _, best_indices = torch.topk(val_X, k=torch_batches)
    # TODO: remove cache clearing once upstream issues regarding non-batch
    # evaluation followed by batch evaluation are resolved T36825603
    # clear caches
    acq_function.model.train()
    acq_function.model.eval()
    return bulk_X[best_indices]
