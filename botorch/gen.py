#!/usr/bin/env python3

from typing import Callable, Dict, List, Optional, Type, Union

import torch
from botorch.utils import check_convergence, columnwise_clamp, fix_features
from torch import Tensor
from torch.optim import Optimizer


def gen_candidates(
    initial_candidates: Tensor,
    acquisition_function: Callable,
    lower_bounds: Optional[Tensor] = None,
    upper_bounds: Optional[Tensor] = None,
    optimizer: Type[Optimizer] = torch.optim.Adam,
    options: Optional[Dict[str, Union[float, str]]] = None,
    max_iter: int = 50,
    verbose: bool = True,
    fixed_features: Optional[Dict[int, Optional[float]]] = None,
) -> Tensor:
    """Generate a set of candidates via optimization from a given set of
    starting points.

    Args:
        initial_candidates: starting points for optimization
        acquisition_function: acquisition function to be used
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
    options = options or {}
    clamped_candidates = columnwise_clamp(
        initial_candidates, lower_bounds, upper_bounds
    ).requires_grad_(True)
    candidates = fix_features(clamped_candidates, fixed_features)
    bayes_optimizer = optimizer(
        params=[clamped_candidates], lr=options.get("lr", 0.025)
    )
    param_trajectory: Dict[str, List[Tensor]] = {"candidates": []}
    loss_trajectory: List[float] = []
    i = 0
    converged = False
    while not converged:
        i += 1
        loss = -acquisition_function(candidates)
        if verbose:
            print("Iter: {} - Value: {:.3f}".format(i, -loss.item()))
        loss_trajectory.append(loss.item())
        param_trajectory["candidates"].append(candidates.clone())

        def closure():
            bayes_optimizer.zero_grad()
            loss = -acquisition_function(candidates)
            loss.backward()
            return loss

        bayes_optimizer.step(closure)
        clamped_candidates.data = columnwise_clamp(
            clamped_candidates, lower_bounds, upper_bounds
        )
        candidates = fix_features(clamped_candidates, fixed_features)
        converged = check_convergence(
            loss_trajectory=loss_trajectory,
            param_trajectory=param_trajectory,
            options=options,
            max_iter=max_iter,
        )

    return candidates
