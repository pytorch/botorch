#!/usr/bin/env python3

from typing import Dict, List, NamedTuple, Optional, Union


class OptimizeConfig(NamedTuple):
    """
    Config for closed loop optimization.
    """

    joint_optimization: bool = True  # whether to do joint or sequential optimization
    fine_tune: bool = False  # whether to perform a secondary joint optimization
    initial_points: int = 10
    q: int = 5
    n_batch: int = 10
    num_starting_points: int = 1
    num_raw_samples: int = 500  # number of samples for random restart heuristic
    max_retries: int = 0  # number of retries, in the case of exceptions
    warm_start: bool = True
    candidate_optim_options: Dict[str, Union[float, int, str]] = {
        "maxiter": 100,
        "ftol": 1e-5,
    }
    model_fit_options: Dict[str, Union[float, int]] = {"maxiter": 50}


class AcquisitionFunctionConfig(NamedTuple):
    """
    Config for the acquisition function
    """

    name: str
    args: Optional[Dict[str, Union[bool, float, int]]] = None
