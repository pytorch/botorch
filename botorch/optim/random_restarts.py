#! /usr/bin/env python3

from typing import Callable, Dict, Optional, Type, Union

import torch
from botorch.gen import gen_candidates
from botorch.optim.initializers import q_batch_initialization
from gpytorch import Module
from torch import Tensor
from torch.optim import LBFGS
from torch.optim.optimizer import Optimizer


def random_restarts(
    gen_function: Callable[[int], Tensor],
    acq_function: Module,
    q: int,
    num_starting_points: int = 1,
    multiplier: int = 50,
    lower_bounds: Optional[Tensor] = None,
    upper_bounds: Optional[Tensor] = None,
    optimizer: Type[Optimizer] = torch.optim.Adam,
    options: Optional[Dict[str, Union[float, str]]] = None,
    max_iter: int = 50,
    verbose: bool = True,
    fixed_features: Optional[Dict[int, Optional[float]]] = None,
):
    """
    Generate a set of candidates via multi-start optimization.
    Args:
        gen_function: A function n -> X_cand producing n (typically random)
            feasible candidates as a `n x d` tensor X_cand
        acq_function:  An acquisition function Module assumed
            to support the given value of q that returns
            a Tensor when evaluated at a given q-batch
        q: The number of candidates in each q-batch
        num_starting_points:  Number of starting points for multistart acquisition
            function optimization.  Must be less than multiplier.
        multiplier: This factor determines how many q-batches
            to generate from gen_function
        lower_bounds: minimum values for each column of initial_candidates
        upper_bounds: maximum values for each column of initial_candidates
        optimizer (Optimizer): The pytorch optimizer to use to perform
            candidate search
        options:  options used to control the optimization
        max_iter (int):  maximum number of iterations
        verbose (bool):  whether to provide verbose output
        fixed_features:  This is a dictionary of feature indices
            to values, where all generated candidates will have features
            fixed to these values.  If the dictionary value is None, then that
            feature will just be fixed to the clamped value and
            not optimized.  Assumes values to be compatible with
            lower_bounds and upper_bounds!

    Returns:
        The set of generated candidates

    """
    # randomly sample restart points
    # performs non-batch evaluation
    initial_candidates = q_batch_initialization(
        gen_function=gen_function,
        acq_function=acq_function,
        q=q,
        multiplier=multiplier,
        torch_batches=num_starting_points,
    )
    # performs batch evaluation
    candidates, batch_acquisition = gen_candidates(
        initial_candidates=initial_candidates,
        acquisition_function=acq_function,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        optimizer=optimizer,
        options=options,
        max_iter=max_iter,
        verbose=verbose,
        fixed_features=fixed_features,
    )
    return candidates[torch.max(batch_acquisition.view(-1), dim=0)[1].item()]
