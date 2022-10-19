#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Some methods for sampling the Pareto optimal points. This relies on the pymoo
package for solving multi-objective optimization problems using genetic algorithms.

TODO: Implement a multi-objective solver which takes advantage of derivatives. This
    could potentially also be parallelized.
"""

from __future__ import annotations
from typing import Optional, Tuple
import torch
from botorch.models.deterministic import GenericDeterministicModel
from botorch.models.model import Model
from torch import Tensor
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import (
    get_crossover,
    get_mutation,
    get_sampling,
    get_termination,
)
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from math import ceil
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning
)
from botorch.utils.gp_sampling import get_gp_samples


def pareto_solver(
    model: GenericDeterministicModel,
    bounds: Tensor,
    num_objectives: int,
    num_generations: int = 100,
    pop_size: int = 100,
    num_offsprings: int = 10,
    maximize: bool = True,
) -> Tuple[Tensor, Tensor]:
    r"""Runs pymoo genetic algorithm NSGA2 to compute the Pareto set and front.
        https://pymoo.org/algorithms/moo/nsga2.html

    Args:
        model: The random Fourier feature GP sample.
        bounds: A `2 x d`-dim Tensor containing the input bounds.
        num_objectives: The number of objectives.
        num_generations: The number of generations of NSGA2.
        pop_size: The population size maintained at each step of NSGA2.
        num_offsprings: The number of offsprings used in NSGA2.
        maximize: If true we solve for the Pareto maximum.

    Returns:
        The `num_pareto_points` pareto optimal set and front, where
        `num_pareto_points <= pop_size` depends randomly on the model and
        parameter choices.

        pareto_set: A `num_pareto_points x d`-dim Tensor.
        pareto_front: A `num_pareto_points x num_objectives`-dim Tensor.
    """
    d = bounds.shape[-1]
    weight = -1.0 if maximize else 1.0

    class PymooProblem(Problem):
        def __init__(self):
            super().__init__(
                n_var=d,
                n_obj=num_objectives,
                n_constr=0,
                xl=bounds[0].detach().numpy(),
                xu=bounds[1].detach().numpy(),
            )

        def _evaluate(self, x, out, *args, **kwargs):
            xt = torch.tensor(x, dtype=torch.float)
            out["F"] = weight * model.posterior(xt).mean.detach().numpy()
            return out

    # Use NSGA2 to generate a number of Pareto optimal points.
    results = minimize(
        problem=PymooProblem(),
        algorithm=NSGA2(
            pop_size=pop_size,
            n_offsprings=num_offsprings,
            sampling=get_sampling("real_random"),
            crossover=get_crossover("real_sbx", prob=0.9, eta=15),
            mutation=get_mutation("real_pm", eta=20),
            eliminate_duplicates=True,
        ),
        termination=get_termination("n_gen", num_generations),
    )
    pareto_set = torch.tensor(results.X, dtype=torch.float)
    pareto_front = weight * torch.tensor(results.F, dtype=torch.float)

    if pareto_set.ndim == 1:
        return pareto_set.unsqueeze(0), pareto_front.unsqueeze(0)
    else:
        return pareto_set, pareto_front


def sample_pareto_sets_and_fronts(
    model: Model,
    bounds: Tensor,
    num_pareto_samples: int,
    num_pareto_points: int,
    maximize: bool = True,
    num_generations: int = 100,
    pop_size: int = 100,
    num_offsprings: int = 10,
    num_rff_features: int = 500,
    max_tries: Optional[int] = 3,
    num_greedy: Optional[int] = 0,
    X_baseline: Optional[Tensor] = None,
    ref_point: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    r"""Computes the Pareto optimal set and front from samples of the GP.

    (i) Samples are generated using random Fourier features.
    (ii) Samples are optimized using NSGA2 (a genetic algorithm).
    (iii) The genetic algorithm does not guarantee exactly `num_pareto_points` are
        generated. Therefore, we apply a truncation strategy which selects
        `num_greedy` points based on the sample hypervolume improvement, and then
        selects the remaining points randomly.

    Args:
        model: The model. This does not support models which include fantasy
            observations.
        bounds: A `2 x d`-dim Tensor containing the input bounds.
        num_pareto_samples: The number of GP samples.
        num_pareto_points: The number of Pareto optimal points to be outputted.
        maximize: If true we solve for the Pareto maximum.
        num_generations: The number of generations of NSGA2.
        pop_size: The population size maintained at each step of NSGA2.
        num_offsprings: The number of offsprings used in NSGA2.
        num_rff_features: The number of random Fourier features used for GP
            sampling. Defaults to `500`.
        max_tries: The maximum number of runs of NSGA2 to find num_pareto_points.
        num_greedy: The number of points to select via the hypervolume improvement
            truncation.
        X_baseline: If `num_greedy > 0`, then we need to specify a `N x d`-dim Tensor
            containing the training inputs used to perform the greedy selection.
        ref_point: If `num_greedy > 0`, then we need to specify a `M`-dim Tensor
            containing the reference point.

    Returns:
        pareto_sets: A `num_samples x num_pareto_points x d`-dim Tensor
        pareto_fronts: A `num_samples x num_pareto_points x M`-dim Tensor.
    """
    M = model.num_outputs
    d = bounds.shape[-1]
    if M == 1:
        num_greedy = 0

    pareto_sets = torch.zeros((num_pareto_samples, num_pareto_points, d)).to(bounds)
    pareto_fronts = torch.zeros((num_pareto_samples, num_pareto_points, M)).to(bounds)
    for i in range(num_pareto_samples):
        model_sample_i = get_gp_samples(
            model=model, num_outputs=M, n_samples=1,
            num_rff_features=num_rff_features
        )

        ratio = 2
        num_tries = 0

        # Run solver until we find at least `num_pareto_samples` optimal points
        # or if the maximum number of retries is exceeded.
        while (ratio > 1) and (num_tries < max_tries):
            pareto_set_i, pareto_front_i = pareto_solver(
                model=model_sample_i,
                bounds=bounds,
                num_objectives=M,
                num_generations=num_generations,
                pop_size=pop_size,
                num_offsprings=num_offsprings,
                maximize=maximize,
            )
            num_pareto_generated = pareto_set_i.shape[0]
            ratio = ceil(num_pareto_points / num_pareto_generated)
            num_tries = num_tries + 1

        # If maximum number of retries exceeded throw out a runtime error.
        if ratio > 1:
            error_text = "Only found " + str(num_pareto_generated) + \
                " Pareto efficient points instead of " + str(num_pareto_points) + "."
            raise RuntimeError(error_text)

        # Randomly truncate the Pareto set and front
        if num_greedy == 0:
            indices = torch.randperm(num_pareto_generated)[:num_pareto_points]

        # Truncate Pareto set and front based on the sample hypervolume improvement
        # or else select the points randomly.
        else:
            # get `num_pareto_points - num_greedy` indices randomly
            num_remaining = max(0, num_pareto_points - num_greedy)
            indices = torch.randperm(num_pareto_generated)[:num_remaining].tolist()

            pending_pareto_front_i = model_sample_i.posterior(X_baseline).mean

            for k in range(num_pareto_points - num_remaining):
                partitioning = FastNondominatedPartitioning(
                    ref_point=ref_point, Y=pending_pareto_front_i
                )
                hypercell_bounds = partitioning.hypercell_bounds

                # `1 x num_boxes x M`
                lo = hypercell_bounds[0].unsqueeze(0)
                up = hypercell_bounds[1].unsqueeze(0)

                # Compute the sample hypervolume improvement.
                hvi = torch.max(
                    torch.min(pareto_front_i.unsqueeze(-2), up) - lo,
                    torch.zeros(lo.shape).to(bounds)
                ).prod(dim=-1).sum(dim=-1)

                # Zero out the pending points.
                hvi[indices] = 0

                # Store best index.
                best_index = torch.argmax(hvi).tolist()
                indices = indices + [best_index]

                # Update the pareto front.
                pending_pareto_front_i = torch.cat([
                    pending_pareto_front_i,
                    pareto_front_i[best_index:best_index+1, :]
                ], dim=0)

            indices = torch.tensor(indices)

        pareto_sets[i, :, :] = pareto_set_i[indices, :]
        pareto_fronts[i, :, :] = pareto_front_i[indices, :]

    return pareto_sets, pareto_fronts
