#!/usr/bin/env python3

from typing import Callable, Dict, List, Optional, Union

from torch import Tensor
from torch.nn import Module

from ..models import Model
from ..utils import squeeze_last_dim
from .batch_modules import (
    qExpectedImprovement,
    qKnowledgeGradient,
    qKnowledgeGradientNoDiscretization,
    qNoisyExpectedImprovement,
    qProbabilityOfImprovement,
    qUpperConfidenceBound,
)


def get_acquisition_function(
    acquisition_function_name: str,
    model: Model,
    X_observed: Tensor,
    objective: Callable[[Tensor], Tensor] = squeeze_last_dim,
    constraints: Optional[List[Callable[[Tensor], Tensor]]] = None,
    infeasible_cost: float = 0.0,
    X_pending: Optional[Tensor] = None,
    seed: Optional[int] = None,
    acquisition_function_args: Optional[Dict[str, Union[bool, float, int]]] = None,
) -> Module:
    """Initializes and returns an AcquisitionFunction module.

    Args:
        acquisition_function_name: Name of the acquisition function.
        model: A fitted model.
        X_observed: A `q' x d`-dim Tensor of `q'`` design points that have
            already been observed and would be considered as the best design
            point.
        objective: A callable mapping a Tensor of size `b x q x t` to a Tensor
            of size `b x q`, where `t` is the number of outputs of the model.
            This callable must support broadcasting. If omitted, squeeze the
            output dimension (applicable to single- output models only).
        constraints: A list of callables, each mapping a Tensor of size
            `b x q x t` to a Tensor of size `b x q`, where negative values
            imply feasibility. Note: the callable must support broadcasting.
            Only relevant for multi-output models (`t` > 1).
        infeasible_cost: The infeasibility cost `M`. Should be set s.t.
            `-M < min_x obj(x)`.
        X_pending: A `m x d`-dim Tensor with `m` design points that are
            pending for evaluation.
        seed: If provided, perform deterministic optimization (i.e. the
            function to optimize is fixed and not stochastic).
        qmc: If True, use quasi-Monte-Carlo sampling (instead of iid).
        acquisition_function_args: A map containing extra arguments for
            initializing the acquisition function module. E.g. for q-UCB, this
            can be used to specify the `beta` parameter.

    Returns:
        AcquisitionFunction: the acquisition function
    """
    acquisition_function_args = acquisition_function_args or {}
    if acquisition_function_name == "qEI":
        return qExpectedImprovement(
            model=model,
            best_f=objective(model.posterior(X_observed).mean).max().item(),
            objective=objective,
            constraints=constraints,
            infeasible_cost=infeasible_cost,
            X_pending=X_pending,
            seed=seed,
            **acquisition_function_args,
        )
    elif acquisition_function_name == "qPI":
        return qProbabilityOfImprovement(
            model=model,
            best_f=objective(model.posterior(X_observed).mean).max().item(),
            objective=objective,
            constraints=constraints,
            X_pending=X_pending,
            seed=seed,
            **acquisition_function_args,
        )
    elif acquisition_function_name == "qNEI":
        return qNoisyExpectedImprovement(
            model=model,
            X_observed=X_observed,
            objective=objective,
            constraints=constraints,
            infeasible_cost=infeasible_cost,
            X_pending=X_pending,
            seed=seed,
            **acquisition_function_args,
        )
    elif acquisition_function_name == "qKG":
        return qKnowledgeGradient(
            model=model,
            X_observed=X_observed,
            objective=objective,
            constraints=constraints,
            X_pending=X_pending,
            seed=seed,
            **acquisition_function_args,
        )
    elif acquisition_function_name == "qUCB":
        if "beta" not in acquisition_function_args:
            raise ValueError(
                "beta must be specified in acquisition_function_args for qUCB."
            )
        return qUpperConfidenceBound(
            model=model, X_pending=X_pending, seed=seed, **acquisition_function_args
        )

    elif acquisition_function_name == "qKGNoDiscretization":
        return qKnowledgeGradientNoDiscretization(
            model=model,
            objective=objective,
            constraints=constraints,
            X_pending=X_pending,
            seed=seed,
            **acquisition_function_args,
        )
    else:
        raise NotImplementedError(
            f"Unknown acquistition function {acquisition_function_name}"
        )
