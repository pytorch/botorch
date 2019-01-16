#!/usr/bin/env python3

from typing import List, NamedTuple, Optional

from torch import Tensor

from ..models.model import Model


# TODO: replace NamedTuple output containers with dataclass in py3.7+: T39170426


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
    runtimes: List[float]  # iteration
    weights: Optional[List[Tensor]] = None  # iteration x q_i


class BenchmarkOutput(NamedTuple):
    """Container for closed loop output collected across runs"""

    Xs: List[List[Tensor]]  # run x iteration x q_i x d
    Ys: List[List[Tensor]]  # run x iteration x q_i x t
    Ycovs: List[List[Tensor]]  # run x iteration x q_i x t x t
    best: List[List[Tensor]]  # run x iteration x d
    best_model_objective: List[List[float]]  # run x iteration
    best_model_feasibility: List[List[float]]  # run x iteration
    costs: List[List[float]]  # run x iteration
    runtimes: List[List[float]]  # run x iteration
    best_true_objective: List[Tensor]  # run x iteration
    best_true_feasibility: List[Tensor]  # run x iteration
    regrets: List[Tensor]  # run x iteration (regret of the q-batch)
    cumulative_regrets: List[Tensor]  # run x iteration
    weights: Optional[List[List[Tensor]]] = None  # run x iteration x q_i


class AggregatedBenchmarkOutput(NamedTuple):
    """Container for a summary of benchmark output across trials."""

    num_trials: int
    mean_runtime: Tensor
    var_runtime: Tensor
    batch_iterations: Tensor
    mean_best_model_objective: Tensor
    var_best_model_objective: Tensor
    mean_best_model_feasibility: Tensor
    var_best_model_feasibility: Tensor
    mean_best_true_objective: Tensor
    var_best_true_objective: Tensor
    mean_best_true_feasibility: Tensor
    var_best_true_feasibility: Tensor
    mean_cost: Tensor
    var_cost: Tensor
    mean_regret: Tensor
    var_regret: Tensor
    mean_cumulative_regret: Tensor
    var_cumulative_regret: Tensor


class _ModelBestPointOutput(NamedTuple):
    model: Model
    best_point: Tensor
    obj: Tensor
    feas: Tensor
    retry: int
