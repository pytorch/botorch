#!/usr/bin/env python3

from typing import Callable, List, NamedTuple, Optional, Tuple

import torch
from botorch import fit_model
from gpytorch import fast_pred_var
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from torch import Tensor

from ..acquisition.utils import get_acquisition_function
from ..models.gp_regression import SingleTaskGP
from ..models.model import Model
from ..optim import random_restarts
from ..utils import standardize
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
        Tensor: best point
        float: best objective
        float: feasibility of best point

    """
    posterior = model.posterior(X)
    with fast_pred_var():
        # mc_samples x q x (t)
        samples = posterior.rsample(torch.Size([mc_samples]))
    obj = objective(samples).clamp_min_(0)
    obj_raw = objective(samples)
    feas_raw = torch.ones_like(obj_raw)
    if constraints is not None:
        for constraint in constraints:
            feas_raw.mul_((constraint(samples) < 0).type_as(obj))
        obj.mul_(feas_raw)
    _, best_idx = torch.max(obj.mean(dim=0), dim=-1)
    return (
        X[best_idx].detach(),
        obj_raw.mean(dim=0)[best_idx].item(),
        feas_raw.mean(dim=0)[best_idx].item(),
    )


def optimize(
    func: Callable[[Tensor], List[Tensor]],
    gen_function: Callable[[int], Tensor],
    config: OptimizeConfig,
    objective: Callable[[Tensor], Tensor] = lambda Y: Y,
    constraints: Optional[List[Callable[[Tensor], Tensor]]] = None,
    lower_bounds: Optional[Tensor] = None,
    upper_bounds: Optional[Tensor] = None,
    verbose: bool = False,
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
    Returns:
        ClosedLoopOutput: outputs from optimization

    # TODO: Add support for multi-task models.
    # TODO: Add support for known observation noise.

    """
    # TODO: remove exception handling wrapper when model fitting is stabilized
    failed_count = 0
    failed = True
    while failed:
        failed = False
        try:
            output = ClosedLoopOutput(
                Xs=[],
                Ys=[],
                Ycovs=[],
                best=[],
                best_model_objective=[],
                best_model_feasibility=[],
                costs=[],
                runtime=0,
            )
            X = gen_function(config.initial_points)
            model = None
            train_X = None
            for iteration in range(config.n_batch):
                if verbose:
                    print("Iteration:", iteration + 1)
                if iteration > 0:
                    acquisition_function = get_acquisition_function(
                        acquisition_function_name=config.acquisition_function_name,
                        model=model,
                        X_observed=train_X,
                        objective=objective,
                        constraints=constraints,
                        X_pending=None,
                    )
                    if verbose:
                        print("---- acquisition optimization")
                    candidates = random_restarts(
                        gen_function=gen_function,
                        acq_function=acquisition_function,
                        q=config.q,
                        num_starting_points=config.num_starting_points,
                        multiplier=100,
                        max_iter=config.candidate_gen_max_iter,
                        verbose=False,
                    )
                    X = candidates.detach()
                if verbose:
                    print("---- evaluate")

                Y, Ycov = func(X)
                output.Xs.append(X.detach())
                output.Ys.append(Y.detach())
                output.Ycovs.append(Ycov.detach())

                train_X = torch.cat([X for X in output.Xs], dim=0).data
                train_Y = standardize(torch.cat([Y for Y in output.Ys], dim=0).data)
                if verbose:
                    print("---- train")
                # TODO: copy over the state_dict from the existing model before
                # optimization begins
                likelihood = GaussianLikelihood()
                model = SingleTaskGP(train_X, train_Y, likelihood)
                mll = ExactMarginalLogLikelihood(likelihood, model)
                if train_X.is_cuda:
                    mll = mll.cuda()
                if train_X.dtype == torch.double:
                    mll = mll.double()
                mll = fit_model(mll)
                if verbose:
                    print("---- identify")
                best, obj, feas = greedy(train_X, model, objective, constraints)
                output.best.append(best)
                output.best_model_objective.append(obj)
                output.best_model_feasibility.append(feas)
                output.costs.append(1.0)
            return output
        except (RuntimeError, TypeError):
            failed_count += 1
            failed = True
            if verbose:
                print("---- Failed {} times ----".format(failed_count))


def optimize_multiple_runs(
    func: Callable[[Tensor], List[Tensor]],
    gen_function: Callable[[int], Tensor],
    config: OptimizeConfig,
    objective: Callable[[Tensor], Tensor] = lambda Y: Y,
    constraints: Optional[List[Callable[[Tensor], Tensor]]] = None,
    lower_bounds: Optional[Tensor] = None,
    upper_bounds: Optional[Tensor] = None,
    verbose: bool = False,
    num_runs: Optional[int] = 1,
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
        num_runs: number of runs of bayesian optimization
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
        run_output = optimize(
            func=func,
            gen_function=gen_function,
            config=config,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            verbose=verbose,
        )
        print("---- Finished {} loops ----".format(run + 1))
        for f in run_output._fields:
            getattr(outputs, f).append(getattr(run_output, f))
    return outputs
