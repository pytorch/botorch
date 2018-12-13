#!/usr/bin/env python3

from time import time
from typing import Callable, List, NamedTuple, Optional, Tuple

import gpytorch
import torch
from botorch import fit_model
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from torch import Tensor

from ..acquisition.batch_modules import BatchAcquisitionFunction
from ..acquisition.utils import get_acquisition_function
from ..models.gp_regression import SingleTaskGP
from ..models.model import Model
from ..optim.random_restarts import random_restarts
from .output import BenchmarkOutput, ClosedLoopOutput


class OptimizeConfig(NamedTuple):
    """Config for closed loop optimization"""

    acquisition_function_name: str = "qEI"
    initial_points: int = 10
    q: int = 5
    n_batch: int = 10
    candidate_gen_max_iter: int = 25
    model_max_iter: int = 50
    num_starting_points: int = 1
    max_retries: int = 0  # number of retries, in the case of exceptions


def greedy(
    X: Tensor,
    model: Model,
    objective: Callable[[Tensor], Tensor] = lambda Y: Y,
    constraints: Optional[List[Callable[[Tensor], Tensor]]] = None,
    mc_samples: int = 10000,
) -> Tuple[Tensor, float, float]:
    """
    Fetch the best point, best objective, and feasibility based on the joint
    posterior of the evaluated points.

    Args:
        X: q x d tensor of points
        model: model: A fitted model.
        objective: A callable mapping a Tensor of size `b x q x (t)`
            to a Tensor of size `b x q`, where `t` is the number of
            outputs (tasks) of the model. Note: the callable must support broadcasting.
            If omitted, use the identity map (applicable to single-task models only).
            Assumed to be non-negative when the constraints are used!
        constraints: A list of callables, each mapping a Tensor of size
            `b x q x t` to a Tensor of size `b x q`, where negative values imply
            feasibility. Note: the callable must support broadcasting. Only
            relevant for multi-task models (`t` > 1).
        mc_samples: The number of Monte-Carlo samples to draw from the model
            posterior.
    Returns:
        Tensor: `1 x d` best point
        float: best objective
        float: feasibility of best point

    """
    posterior = model.posterior(X)
    with gpytorch.settings.fast_pred_var():
        # mc_samples x b x q x (t)
        samples = posterior.rsample(sample_shape=torch.Size([mc_samples])).unsqueeze(1)
    # TODO: handle non-positive definite objectives
    obj = objective(samples).clamp_min_(0)  # pyre-ignore [16]
    obj_raw = objective(samples)
    feas_raw = torch.ones_like(obj_raw)
    if constraints is not None:
        for constraint in constraints:
            feas_raw.mul_((constraint(samples) < 0).type_as(obj))  # pyre-ignore [16]
        obj.mul_(feas_raw)
    _, best_idx = torch.max(obj.mean(dim=0), dim=-1)
    return (
        X[best_idx].view(-1, X.shape[-1]).detach(),
        obj_raw.mean(dim=0)[0, best_idx].item(),  # pyre-ignore [16]
        feas_raw.mean(dim=0)[0, best_idx].item(),  # pyre-ignore [16]
    )


def run_closed_loop(
    func: Callable[[Tensor], List[Tensor]],
    gen_function: Callable[[int], Tensor],
    config: OptimizeConfig,
    objective: Callable[[Tensor], Tensor] = lambda Y: Y,
    constraints: Optional[List[Callable[[Tensor], Tensor]]] = None,
    lower_bounds: Optional[Tensor] = None,
    upper_bounds: Optional[Tensor] = None,
    verbose: bool = False,
    seed: Optional[int] = None,
) -> ClosedLoopOutput:
    """
    Uses Bayesian Optimization to optimize func.

    Args:
        func: function to optimize (maximize by default)
        gen_function: A function n -> X_cand producing n (typically random)
            feasible candidates as a `n x d` tensor X_cand
        config: configuration for the optimization
        objective: A callable mapping a Tensor of size `b x q x (t)`
            to a Tensor of size `b x q`, where `t` is the number of
            outputs (tasks) of the model. Note: the callable must support broadcasting.
            If omitted, use the identity map (applicable to single-task models only).
            Assumed to be non-negative when the constraints are used!
        constraints: A list of callables, each mapping a Tensor of size
            `b x q x t` to a Tensor of size `b x q`, where negative values imply
            feasibility. Note: the callable must support broadcasting. Only
            relevant for multi-task models (`t` > 1).
        lower_bounds: minimum values for each column of initial_candidates
        upper_bounds: maximum values for each column of initial_candidates
        verbose: whether to provide verbose output
        seed: if seed is provided, do deterministic optimization where the function to
            optimize is fixed and not stochastic.
    Returns:
        ClosedLoopOutput: outputs from optimization

    # TODO: Add support for multi-task models.
    # TODO: Add support for known observation noise.
    """
    # TODO: remove exception handling wrapper when model fitting is stabilized
    Xs = []
    Ys = []
    Ycovs = []
    best = []
    best_model_objective = []
    best_model_feasibility = []
    costs = []
    runtime = 0.0
    retry = 0

    start_time = time()
    X = gen_function(config.initial_points)
    model = None
    train_X = None
    for iteration in range(config.n_batch):
        failed = True
        while retry <= config.max_retries and failed:
            try:
                if verbose:
                    print("Iteration:", iteration + 1)
                if iteration > 0:
                    # type check for pyre
                    assert isinstance(model, Model)
                    acquisition_function: BatchAcquisitionFunction = get_acquisition_function(
                        acquisition_function_name=config.acquisition_function_name,
                        model=model,
                        X_observed=train_X,
                        objective=objective,
                        constraints=constraints,
                        X_pending=None,
                        seed=seed,
                    )
                    if verbose:
                        print("---- acquisition optimization")
                    candidates = random_restarts(
                        gen_function=gen_function,
                        acq_function=acquisition_function,
                        q=config.q,
                        num_starting_points=config.num_starting_points,
                        multiplier=100,
                        options={"maxiter": config.candidate_gen_max_iter},
                    )
                    X = acquisition_function.extract_candidates(candidates).detach()
                if verbose:
                    print("---- evaluate")
                Y, Ycov = func(X)
                Xs.append(X.detach())
                Ys.append(Y.detach())
                Ycovs.append(Ycov.detach())
                train_X = torch.cat(Xs, dim=0).detach()
                train_Y = torch.cat(Ys, dim=0).detach()
                if verbose:
                    print("---- train")
                # TODO: copy over the state_dict from the existing model before
                # optimization begins
                likelihood = GaussianLikelihood()  # pyre-ignore [16]
                model = SingleTaskGP(train_X, train_Y, likelihood)
                mll = ExactMarginalLogLikelihood(likelihood, model)
                mll.to(dtype=train_X.dtype, device=train_X.device)
                mll = fit_model(mll, options={"maxiter": config.model_max_iter})
                if verbose:
                    print("---- identify")
                best_point, obj, feas = greedy(
                    X=train_X, model=model, objective=objective, constraints=constraints
                )
                best.append(best_point)
                best_model_objective.append(obj)
                best_model_feasibility.append(feas)
                costs.append(1.0)
                failed = False
            except Exception:
                retry += 1
                if verbose:
                    print("---- Failed {} times ----".format(retry))
                Xs = []
                Ys = []
                Ycovs = []
                best = []
                best_model_objective = []
                best_model_feasibility = []
                costs = []
                if retry > config.max_retries:
                    raise
    runtime = time() - start_time
    return ClosedLoopOutput(
        Xs=Xs,
        Ys=Ys,
        Ycovs=Ycovs,
        best=best,
        best_model_objective=best_model_objective,
        best_model_feasibility=best_model_feasibility,
        costs=costs,
        runtime=runtime,
    )


def run_benchmark(
    func: Callable[[Tensor], List[Tensor]],
    gen_function: Callable[[int], Tensor],
    config: OptimizeConfig,
    objective: Callable[[Tensor], Tensor] = lambda Y: Y,
    constraints: Optional[List[Callable[[Tensor], Tensor]]] = None,
    lower_bounds: Optional[Tensor] = None,
    upper_bounds: Optional[Tensor] = None,
    verbose: bool = False,
    seed: Optional[int] = None,
    num_runs: Optional[int] = 1,
    true_func: Optional[Callable[[Tensor], List[Tensor]]] = None,
    global_optimum: Optional[float] = None,
) -> BenchmarkOutput:
    """
    Uses Bayesian Optimization to optimize func multiple times.

    Args:
        func: function to optimize (maximize by default)
        gen_function: A function n -> X_cand producing n (typically random)
            feasible candidates as a `n x d` tensor X_cand
        config: configuration for the optimization
        objective: A callable mapping a Tensor of size `b x q x (t)`
            to a Tensor of size `b x q`, where `t` is the number of
            outputs (tasks) of the model. Note: the callable must support broadcasting.
            If omitted, use the identity map (applicable to single-task models only).
            Assumed to be non-negative when the constraints are used!
        constraints: A list of callables, each mapping a Tensor of size
            `b x q x t` to a Tensor of size `b x q`, where negative values imply
            feasibility. Note: the callable must support broadcasting. Only
            relevant for multi-task models (`t` > 1).
        lower_bounds: minimum values for each column of initial_candidates
        upper_bounds: maximum values for each column of initial_candidates
        verbose: whether to provide verbose output
        seed: if seed is provided, do deterministic optimization where the function to
            optimize is fixed and not stochastic. Note: this seed is incremented
            each run.
        num_runs: number of runs of bayesian optimization
        true_func: true noiseless function being optimized
        global_optimum: the global optimum of func after applying the objective
            transformation. If provided, this is used to compute regret.
    Returns:
        BenchmarkOutput: outputs from optimization
    """
    outputs = BenchmarkOutput(
        Xs=[],
        Ys=[],
        Ycovs=[],
        best=[],
        best_model_objective=[],
        best_model_feasibility=[],
        costs=[],
        runtime=[],
        best_true_objective=[],
        best_true_feasibility=[],
        regrets=[],
        best_regrets=[],
        weights=[],
    )
    for run in range(num_runs):
        run_output = run_closed_loop(
            func=func,
            gen_function=gen_function,
            config=config,
            objective=objective,
            constraints=constraints,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            verbose=verbose,
            seed=seed + run if seed is not None else seed,  # increment seed each run
        )
        if verbose:
            print("---- Finished {} loops ----".format(run + 1))
        # compute true objective base on best point (greedy from model)
        best = torch.cat(run_output.best, dim=0)
        if true_func is not None:
            f_best = true_func(best)[0]
            best_true_objective = objective(f_best).view(-1)
            best_true_feasibility = torch.ones_like(best_true_objective)
            if constraints is not None:
                for constraint in constraints:
                    best_true_feasibility.mul_(  # pyre-ignore [16]
                        (constraint(f_best) < 0)  # pyre-ignore [16]
                        .type_as(best)
                        .view(-1)
                    )
            outputs.best_true_objective.append(best_true_objective)
            outputs.best_true_feasibility.append(best_true_feasibility)
            if global_optimum is not None:
                regrets = [
                    torch.abs(
                        -objective(true_func(X)[0]) + global_optimum  # pyre-ignore [16]
                    )
                    for X in run_output.Xs
                ]
                # compute regret on best point (greedy from model)
                best_regrets = torch.abs(global_optimum - best_true_objective)
                outputs.regrets.append(regrets)
                outputs.best_regrets.append(best_regrets)
        for f in run_output._fields:
            getattr(outputs, f).append(getattr(run_output, f))
    return outputs
