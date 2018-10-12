#!/usr/bin/env python3

from typing import Callable, Dict, List, Optional, Union

import torch
from botorch.utils import check_convergence, columnwise_clamp
from torch import Tensor
from torch.optim import Optimizer


def gen_candidates(
    initial_candidates: Tensor,
    acquisition_function: Callable,
    lower_bounds: Optional[Tensor] = None,
    upper_bounds: Optional[Tensor] = None,
    optimizer: Optimizer = torch.optim.Adam,
    options: Optional[Dict[str, Union[float, str]]] = None,
    max_iter: int = 50,
    verbose: bool = True,
) -> Tensor:
    """Generate a set of candidates via optimization from a given set of
    starting points.

    Args:
        initial_candidates: starting points for optimization
        acquisition_function: acquisition function to be used
        lower_bounds: minimum values for each column of initial_candidates
        upper_bounds: maximum values for each column of initial_candidates
        inner_optimization_steps (int): the number of optimization steps to
            perform when searching for the candidate
        candidate_optimizer (Callable): The pytorch optimizer to use to perform
            candidate search
        learning_rate (float): The learning rate to use for stochastic gradient
            optimization.

    Returns:
        The set of generated candidates

    """
    options = options or {}
    candidates = columnwise_clamp(
        initial_candidates, lower_bounds, upper_bounds
    ).requires_grad_(True)
    bayes_optimizer = optimizer(params=[candidates], lr=options.get("lr", 0.025))
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
        param_trajectory["candidates"].append(candidates)

        def closure():
            bayes_optimizer.zero_grad()
            loss = -acquisition_function(candidates)
            loss.backward()
            return loss

        bayes_optimizer.step(closure)
        candidates.data = columnwise_clamp(candidates, lower_bounds, upper_bounds)
        converged = check_convergence(
            loss_trajectory=loss_trajectory,
            param_trajectory=param_trajectory,
            options=options,
            max_iter=max_iter,
        )

    return candidates
