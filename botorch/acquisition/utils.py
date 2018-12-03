#!/usr/bin/env python3

from typing import Callable, Dict, List, Optional, Union

import torch
from torch.nn import Module

from ..models import Model
from .batch_modules import (
    qExpectedImprovement,
    qNoisyExpectedImprovement,
    qProbabilityOfImprovement,
    qUpperConfidenceBound,
)


def get_acquisition_function(
    acquisition_function_name: str,
    model: Model,
    X_observed: torch.Tensor,
    objective: Callable[[torch.Tensor], torch.Tensor] = lambda Y: Y,
    constraints: Optional[List[Callable[[torch.Tensor], torch.Tensor]]] = None,
    X_pending: Optional[torch.Tensor] = None,
    acquisition_function_args: Optional[Dict[str, Union[float, int]]] = None,
) -> Module:
    """
    Initializes and returns the AcquisitionFunction module.
    Args:
        acquisition_function_name: a string representing the acquisition
            function name
        model: the model
        X_observed: A q' x n Tensor of q' design points that have already been
            observed and would be considered as the best design point. This is used
            by qEI and qPI determine the best_f, and this is used directly by qNEI.
        objective: A callable mapping a Tensor of size `b x q x t x mc_samples`
            to a Tensor of size `b x q x mc_samples`, where `t` is the number of
            outputs (tasks) of the model. If omitted, use the identity map
            (applicable to single-task models only).
            Assumed to be non-negative when the constaints are used!
        constraints: A list of callables, each mapping a Tensor of size
            `b x q x t x mc_samples` to a Tensor of size `b x q x mc_samples`,
            where negative values imply feasibility. Only relevant for multi-task
            models (`t` > 1).
        X_pending:  A (k x d) feature tensor X for k pending
            observations.
        acquisition_function_args: A map containing extra arguments for initializing
            the acquisition function module. For UCB, this should include beta.
    Returns:
        AcquisitionFunction: the acquisition function
    """
    if acquisition_function_name == "qEI":
        return qExpectedImprovement(
            model,
            best_f=model.posterior(X_observed).mean.max().item(),
            objective=objective,
            constraints=constraints,
            X_pending=X_pending,
        )
    elif acquisition_function_name == "qPI":
        return qProbabilityOfImprovement(
            model,
            best_f=model.posterior(X_observed).mean.max().item(),
            objective=objective,
            constraints=constraints,
            X_pending=X_pending,
        )
    elif acquisition_function_name == "qNEI":
        return qNoisyExpectedImprovement(
            model,
            X_observed,
            objective=objective,
            constraints=constraints,
            X_pending=X_pending,
        )
    elif acquisition_function_name == "qUCB":
        if acquisition_function_args and "beta" in acquisition_function_args:
            return qUpperConfidenceBound(
                model, beta=acquisition_function_args["beta"], X_pending=X_pending
            )
        else:
            print("Beta must be specified in acquisition_function_args for qUCB.")
            raise ValueError
    else:
        raise NotImplementedError
