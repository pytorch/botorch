#!/usr/bin/env python3

from time import time
from typing import Callable, List, NamedTuple, Optional, Tuple

import gpytorch
import torch
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from torch import Tensor

from .. import fit_model
from ..acquisition.batch_modules import BatchAcquisitionFunction
from ..acquisition.utils import get_acquisition_function
from ..gen import gen_candidates_scipy, get_best_candidates
from ..models.model import Model
from ..optim.initializers import initialize_q_batch
from ..utils import draw_sobol_samples
from .output import BenchmarkOutput, ClosedLoopOutput


class OptimizeConfig(NamedTuple):
    """Config for closed loop optimization"""

    acquisition_function_name: str = "qEI"
    initial_points: int = 10
    q: int = 5
    n_batch: int = 10
    candidate_gen_maxiter: int = 25
    model_maxiter: int = 50
    num_starting_points: int = 1
    num_raw_samples: int = 500  # number of samples for random restart heuristic
    max_retries: int = 0  # number of retries, in the case of exceptions


def _get_fitted_model(
    train_X: Tensor, train_Y: Tensor, train_Y_se: Tensor, model: Model, maxiter: int
) -> Model:
    """
    Helper function that returns a model fitted to the provided data.

    Args:
        train_X: A `n x d` Tensor of points
        train_Y: A `n x (t)` Tensor of outcomes
        train_Y_se: A `n x (t)` Tensor of observed standard errors for each outcome
        model: an initialized Model. This model must have a likelihood attribute.
        maxiter: The maximum number of iterations
    Returns:
        Model: a fitted model
    """
    # TODO: copy over the state_dict from the existing model before
    # optimization begins
    model.reinitialize(train_X=train_X, train_Y=train_Y, train_Y_se=train_Y_se)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    mll.to(dtype=train_X.dtype, device=train_X.device)
    mll = fit_model(mll, options={"maxiter": maxiter})
    return model


def greedy(
    X: Tensor,
    model: Model,
    objective: Callable[[Tensor], Tensor] = lambda Y: Y,
    constraints: Optional[List[Callable[[Tensor], Tensor]]] = None,
    mc_samples: int = 10000,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Fetch the best point, best objective, and feasibility based on the joint
    posterior of the evaluated points.

    Args:
        X: `q x d` (or `b x q x d`)-dim (batch mode) tensor of points
        model: model: A fitted model.
        objective: A callable mapping a Tensor of size `b x q x (t)` to a Tensor
            of size `b x q`, where `t` is the number of outputs (tasks) of the
            model. Note: the callable must support broadcasting. If omitted, use
            the identity map (applicable to single-task models only).
            Assumed to be non-negative when the constraints are used!
        constraints: A list of callables, each mapping a Tensor of size
            `b x q x t` to a Tensor of size `b x q`, where negative values imply
            feasibility. Note: the callable must support broadcasting. Only
            relevant for multi-task models (`t` > 1).
        mc_samples: The number of Monte-Carlo samples to draw from the model
            posterior.
    Returns:
        Tensor: `d` (or `b x d`)-dim (batch mode) best point(s)
        Tensor: `0` (or `b`)-dim (batch mode) tensor of best objective(s)
        Tensor: `0` (or `b`)-dim (batch mode) tensor of feasibility of best point(s)

    """
    if X.dim() < 2 or X.dim() > 3:
        raise ValueError("X must have two or three dimensions")
    batch_mode = X.dim() == 3
    if not batch_mode:
        X = X.unsqueeze(0)  # internal logic always operates in batch mode
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        posterior = model.posterior(X)
        # mc_samples x b x q x (t)
        samples = posterior.rsample(sample_shape=torch.Size([mc_samples]))
    # TODO: handle non-positive definite objectives
    obj = objective(samples).clamp_min_(0)  # pyre-ignore [16]
    obj_raw = objective(samples)
    feas_raw = torch.ones_like(obj_raw)
    if constraints is not None:
        for constraint in constraints:
            feas_raw.mul_((constraint(samples) < 0).type_as(obj))  # pyre-ignore [16]
        obj.mul_(feas_raw)
    # get max index of weighted objective along the q-batch dimension
    _, best_idcs = torch.max(obj.mean(dim=0), dim=-1)
    # extract corresponding values of X, raw objective, and feasibility
    batch_idxr = torch.arange(best_idcs.numel(), device=best_idcs.device)
    X_best = X[batch_idxr, best_idcs, ...].detach()
    obj_best = obj_raw.mean(dim=0)[batch_idxr, best_idcs]
    feas_best = feas_raw.mean(dim=0)[batch_idxr, best_idcs]
    if not batch_mode:
        # squeeze dimensions back if not called in batch mode
        X_best = X_best.squeeze(0)
        obj_best = obj_best.squeeze(0)
        feas_best = feas_best.squeeze(0)
    return X_best, obj_best, feas_best


def run_closed_loop(
    func: Callable[[Tensor], List[Tensor]],
    gen_function: Callable[[int, int], Tensor],
    config: OptimizeConfig,
    model: Model,
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
        gen_function: A function `(b, q) -> X_cand` producing `b` (typically
            random) feasible q-batch candidates as a `b x q x d` tensor `X_cand`
        config: configuration for the optimization
        model: an initialized Model. This model must have a likelihood attribute.
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
    """
    # TODO: remove exception handling wrappers when model fitting is stabilized
    Xs = []
    Ys = []
    Ycovs = []
    best = []
    best_model_objective = []
    best_model_feasibility = []
    costs = []
    runtime = 0.0
    retry = 0
    refit_model = False

    best_point: Tensor
    obj: float
    feas: float
    train_X = None
    train_Y = None
    train_Y_se = None

    start_time = time()
    X = gen_function(1, config.initial_points).view(config.initial_points, -1)
    for iteration in range(config.n_batch):
        if verbose:
            print("Iteration:", iteration + 1)
        if iteration > 0:
            failed = True
            while retry <= config.max_retries and failed:
                try:
                    # If an exception occured during evaluation time, refit the model
                    if refit_model:
                        model = _get_fitted_model(
                            train_X=train_X,
                            train_Y=train_Y,
                            train_Y_se=train_Y_se,
                            model=model,
                            maxiter=config.model_maxiter,
                        )
                        best_point, obj, feas = greedy(
                            X=train_X,
                            model=model,
                            objective=objective,
                            constraints=constraints,
                        )
                        best[-1] = best_point
                        best_model_objective[-1] = obj
                        best_model_feasibility[-1] = feas
                        costs[-1] = 1.0
                        refit_model = False
                    # type check for pyre
                    assert isinstance(model, Model)
                    acq_func: BatchAcquisitionFunction = get_acquisition_function(
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

                    # Generate random points for determining intial conditions
                    X_rnd = draw_sobol_samples(
                        bounds=torch.stack([lower_bounds, upper_bounds]),
                        n=config.num_raw_samples,
                        q=config.q,
                    )
                    with torch.no_grad():
                        Y_rnd = acq_func(X_rnd)

                    # get a similarity measure for the random restart heuristic
                    covar_module = getattr(model, "covar_module", None)
                    if covar_module is None:
                        sim_measure = None
                    if covar_module is not None:

                        def sim_measure(X: Tensor, x: Tensor) -> Tensor:
                            return covar_module(X, x).evaluate()

                    batch_initial_conditions = initialize_q_batch(
                        X=X_rnd,
                        Y=Y_rnd,
                        n=config.num_starting_points,
                        sim_measure=sim_measure,
                    )

                    batch_candidates, batch_acq_values = gen_candidates_scipy(
                        initial_candidates=batch_initial_conditions,
                        acquisition_function=acq_func,
                        lower_bounds=lower_bounds,
                        upper_bounds=upper_bounds,
                    )
                    candidates = get_best_candidates(
                        batch_candidates=batch_candidates, batch_values=batch_acq_values
                    )

                    X = acq_func.extract_candidates(candidates).detach()
                    failed = False
                    refit_model = False
                except Exception:
                    retry += 1
                    refit_model = True
                    if verbose:
                        print("---- Failed {} times ----".format(retry))
                    if retry > config.max_retries:
                        raise
        if verbose:
            print("---- evaluate")
        Y, Ycov = func(X)
        Xs.append(X)
        Ys.append(Y)
        Ycovs.append(Ycov)
        train_X = torch.cat(Xs, dim=0)
        train_Y = torch.cat(Ys, dim=0)
        train_Y_se = torch.cat(Ycovs, dim=0).sqrt()
        failed = True
        # handle errors in model fitting and evaluation (when selecting best point)
        while retry <= config.max_retries and failed:
            try:
                if verbose:
                    print("---- train")
                model = _get_fitted_model(
                    train_X=train_X,
                    train_Y=train_Y,
                    train_Y_se=train_Y_se,
                    model=model,
                    maxiter=config.model_maxiter,
                )
                if verbose:
                    print("---- identify")
                best_point, obj, feas = greedy(
                    X=train_X, model=model, objective=objective, constraints=constraints
                )
                failed = False
            except Exception:
                retry += 1
                if verbose:
                    print(f"---- Failed {retry} times ----")
                if retry > config.max_retries:
                    raise
        best.append(best_point)
        best_model_objective.append(obj)
        best_model_feasibility.append(feas)
        costs.append(1.0)
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
    gen_function: Callable[[int, int], Tensor],
    config: OptimizeConfig,
    model: Model,
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
        model: an initialized Model. This model must have a likelihood attribute.
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
        cumulative_regrets=[],
        weights=[],
    )
    for run in range(num_runs):
        run_output = run_closed_loop(
            func=func,
            gen_function=gen_function,
            config=config,
            model=model,
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
                regrets = torch.tensor(
                    [
                        (global_optimum - objective(true_func(X)[0]))  # pyre-ignore [6]
                        .sum()
                        .item()
                        for X in run_output.Xs
                    ]
                ).type_as(best)
                # check that objective is never > than global_optimum
                assert torch.all(regrets >= 0)
                # compute regret on best point (greedy from model)
                cumulative_regrets = torch.cumsum(regrets, dim=0)
                outputs.regrets.append(regrets)
                outputs.cumulative_regrets.append(cumulative_regrets)
        for f in run_output._fields:
            getattr(outputs, f).append(getattr(run_output, f))
    return outputs
