#!/usr/bin/env python3

from typing import List, NamedTuple, Optional

from torch import Tensor


class ClosedLoopOutput(NamedTuple):
    """Container for closed loop output
        q_i = q-batch size on iteration i
        t = number of tasks
    """

    Xs: List[Tensor]  # iteration x q_i x d
    Ys: List[Tensor]  # iteration x q_i x t
    Ycovs: List[Tensor]  # iteration x q_i x t x t
    best: List[Tensor]  # iteration x d
    best_model_objective: List[float]  # iteration
    best_model_feasibility: List[float]  # iteration
    costs: List[float]  # iteration
    runtime: float
    weights: Optional[List[Tensor]] = None  # iteration x q_i


class BenchmarkOutput(NamedTuple):
    """Container for collected closed loop output across iterations"""

    Xs: List[List[Tensor]]  # run x iteration x q_i x d
    Ys: List[List[Tensor]]  # run x iteration x q_i x t
    Ycovs: List[List[Tensor]]  # run x iteration x q_i x t x t
    best: List[List[Tensor]]  # run x iteration x d
    best_model_objective: List[List[float]]  # run x iteration
    best_model_feasibility: List[List[float]]  # run x iteration
    costs: List[List[float]]  # run x iteration
    runtime: List[float]  # run
    best_true_objective: List[List[float]]  # run x iteration
    best_true_feasibility: List[List[float]]  # run x iteration
    regrets: List[List[float]]  # run x iteration
    best_regrets: List[List[float]]  # run x iteration
    weights: Optional[List[List[Tensor]]] = None  # run x iteration x q_i
