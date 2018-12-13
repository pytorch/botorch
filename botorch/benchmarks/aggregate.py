#!/usr/bin/env python3

import torch

from .output import AggregatedBenchmarkOutput, BenchmarkOutput


def aggregate_benchmark(output: BenchmarkOutput) -> AggregatedBenchmarkOutput:
    """
    Aggregate closed loop results across trials and return a summary containing
    the mean and variance of each metric.

    Args:
        output: the collected output across trials
    Returns:
        AggregatedBenchmarkOutput: summary
    """
    X = output.Xs[0][0]
    tkwargs = {"dtype": X.dtype, "device": X.device}  # pyre-ignore [16]
    best_model_feasibility = torch.tensor(output.best_model_feasibility, **tkwargs)

    runtime = torch.tensor(output.runtime, **tkwargs)
    trial_X = output.Xs[0]
    batch_iterations = torch.tensor(
        [len(trial_X[i]) for i in range(len(trial_X))],
        dtype=torch.long,
        device=tkwargs["device"],
    ).cumsum(dim=0)
    best_model_objective = torch.tensor(output.best_model_objective, **tkwargs)
    pfeas = best_model_feasibility.mean(dim=0)
    pfeas_var = pfeas * (1 - pfeas)
    costs = torch.tensor(output.costs, **tkwargs)
    best_true_objective = torch.stack(output.best_true_objective, dim=0)
    best_true_feasibility = torch.stack(output.best_true_feasibility, dim=0)
    regrets = torch.stack(output.regrets, dim=0)
    best_regrets = torch.stack(output.best_regrets, dim=0)
    true_pfeas = best_true_feasibility.mean(dim=0)  # pyre-ignore [16]
    true_pfeas_var = true_pfeas * (1 - true_pfeas) / best_true_feasibility.shape[0]

    return AggregatedBenchmarkOutput(
        num_trials=runtime.shape[0],
        mean_runtime=runtime.mean(dim=0).item(),
        var_runtime=runtime.var(dim=0).item(),
        batch_iterations=batch_iterations,
        mean_best_model_objective=best_model_objective.mean(dim=0),
        var_best_model_objective=best_model_objective.var(dim=0),
        mean_best_model_feasibility=pfeas,
        var_best_model_feasibility=pfeas_var,
        mean_cost=costs.mean(dim=0),
        var_cost=costs.var(dim=0),
        mean_regret=regrets.mean(dim=0),  # pyre-ignore [16]
        var_regret=regrets.var(dim=0),  # pyre-ignore [16]
        mean_best_regret=best_regrets.mean(dim=0),  # pyre-ignore [16]
        var_best_regret=best_regrets.var(dim=0),  # pyre-ignore [16]
        mean_best_true_objective=best_true_objective.mean(dim=0),  # pyre-ignore [16]
        var_best_true_objective=best_true_objective.var(dim=0),  # pyre-ignore [16]
        mean_best_true_feasibility=true_pfeas,
        var_best_true_feasibility=true_pfeas_var,
    )
