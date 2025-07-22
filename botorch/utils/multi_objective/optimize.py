#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Callable

import numpy as np

import torch
from botorch.acquisition.multi_acquisition import MultiOutputAcquisitionFunction
from botorch.acquisition.multi_objective.objective import (
    IdentityMCMultiOutputObjective,
    MCMultiOutputObjective,
)
from botorch.utils.multi_objective.hypervolume import get_hypervolume_maximizing_subset
from botorch.utils.multi_objective.pareto import is_non_dominated
from torch import Tensor

try:
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import Problem
    from pymoo.optimize import minimize
    from pymoo.termination.max_gen import MaximumGenerationTermination

    class BotorchPymooProblem(Problem):
        def __init__(
            self,
            n_var: int,
            n_obj: int,
            xl: np.ndarray,
            xu: np.ndarray,
            acqf: MultiOutputAcquisitionFunction,
            dtype: torch.dtype,
            device: torch.device,
            ref_point: Tensor | None = None,
            objective: MCMultiOutputObjective | None = None,
            constraints: list[Callable[[Tensor], Tensor]] | None = None,
        ) -> None:
            """PyMOO problem for optimizing the model posterior mean using NSGA-II.

            This is instantiated and used within `optimize_with_nsgaii` to define
            the optimization problem to interface with pymoo.

            This assumes maximization of all objectives.

            Args:
                n_var: The number of tunable parameters (`d`).
                n_obj: The number of objectives.
                xl: A `d`-dim np.ndarray of lower bounds for each tunable parameter.
                xu: A `d`-dim np.ndarray of upper bounds for each tunable parameter.
                acqf: A MultiOutputAcquisition.
                dtype: The torch dtype.
                device: The torch device.
                ref_point: A list or tensor with `m` elements representing the reference
                    point (in the outcome space), which is treated as a lower bound
                    on the objectives, after applying `objective` to the samples.
                objective: The MCMultiOutputObjective under which the samples are
                    evaluated. Defaults to `IdentityMultiOutputObjective()`.
                constraints: A list of callables, each mapping a Tensor of dimension
                    `sample_shape x batch-shape x q x m` to a Tensor of dimension
                    `sample_shape x batch-shape x q`, where negative values imply
                    feasibility. The acquisition function will compute expected feasible
                    hypervolume.
            """
            num_constraints = 0 if constraints is None else len(constraints)
            if ref_point is not None:
                num_constraints += ref_point.shape[0]
            super().__init__(
                n_var=n_var,
                n_obj=n_obj,
                n_ieq_constr=num_constraints,
                xl=xl,
                xu=xu,
                type_var=np.double,
            )
            self.botorch_acqf = acqf
            self.botorch_ref_point = ref_point
            self.botorch_objective = (
                IdentityMCMultiOutputObjective() if objective is None else objective
            )
            self.botorch_constraints = constraints
            self.torch_dtype = dtype
            self.torch_device = device

        def _evaluate(self, x: np.ndarray, out: dict[str, np.ndarray]) -> None:
            """Evaluate x with respect to the objective/constraints."""
            X = torch.from_numpy(x).to(dtype=self.torch_dtype, device=self.torch_device)
            with torch.no_grad():
                # eval in batch mode, since all we need is the mean and this helps
                # avoid ill-conditioning
                y = self.botorch_acqf(X=X.unsqueeze(-2))
            obj = self.botorch_objective(y)
            # negate the objectives, since we want to maximize this function
            out["F"] = -obj.cpu().numpy()
            constraint_vals = None
            if self.botorch_constraints is not None:
                constraint_vals = torch.stack(
                    [c(y) for c in self.botorch_constraints], dim=-1
                )
            if self.botorch_ref_point is not None:
                # add constraints for the ref point
                ref_constraints = self.botorch_ref_point - obj
                if constraint_vals is not None:
                    constraint_vals = torch.cat(
                        [constraint_vals, ref_constraints], dim=-1
                    )
                else:
                    constraint_vals = ref_constraints
            if constraint_vals is not None:
                out["G"] = constraint_vals.cpu().numpy()

    def optimize_with_nsgaii(
        acq_function: MultiOutputAcquisitionFunction,
        bounds: Tensor,
        num_objectives: int,
        q: int | None = None,
        ref_point: list[float] | Tensor | None = None,
        objective: MCMultiOutputObjective | None = None,
        constraints: list[Callable[[Tensor], Tensor]] | None = None,
        population_size: int = 250,
        max_gen: int | None = None,
        seed: int | None = None,
        fixed_features: dict[int, float] | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Optimize the posterior mean via NSGA-II, returning the Pareto set and front.

        This assumes maximization of all objectives.

        TODO: Add support for discrete parameters.

        Args:
            model: A fitted model.
            bounds: A `2 x d` tensor of lower and upper bounds for each column of `X`.
            num_objectives: The number of objectives.
            ref_point: A list or tensor with `m` elements representing the reference
                point (in the outcome space), which is treated as a lower bound
                on the objectives, after applying `objective` to the samples.
            objective: The MCMultiOutputObjective under which the samples are
                evaluated. Defaults to `IdentityMultiOutputObjective()`.
            constraints: A list of callables, each mapping a Tensor of dimension
                `sample_shape x batch-shape x q x m` to a Tensor of dimension
                `sample_shape x batch-shape x q`, where negative values imply
                feasibility. The acquisition function will compute expected feasible
                hypervolume.
            population_size: the population size for NSGA-II.
            max_gen: The number of iterations for NSGA-II. If None, this uses the
                default termination condition in pymoo for NSGA-II.
            seed: The random seed for NSGA-II.
            fixed_features: A map `{feature_index: value}` for features that
                should be fixed to a particular value during generation. All indices
                should be non-negative.

        Returns:
            A two-element tuple containing the pareto set X and pareto frontier Y.
        """
        tkwargs = {"dtype": bounds.dtype, "device": bounds.device}
        if ref_point is not None:
            ref_point = torch.as_tensor(ref_point, **tkwargs)
        if fixed_features is not None:
            # set lower and upper bounds to the fixed value
            for i, val in fixed_features.items():
                bounds[:, i] = val
        pymoo_problem = BotorchPymooProblem(
            n_var=bounds.shape[-1],
            n_obj=num_objectives,
            xl=bounds[0].cpu().numpy(),
            xu=bounds[1].cpu().numpy(),
            acqf=acq_function,
            ref_point=ref_point,
            objective=objective,
            constraints=constraints,
            **tkwargs,
        )
        algorithm = NSGA2(pop_size=population_size, eliminate_duplicates=True)
        res = minimize(
            problem=pymoo_problem,
            algorithm=algorithm,
            termination=(
                None
                if max_gen is None
                else MaximumGenerationTermination(n_max_gen=max_gen)
            ),
            seed=seed,
            verbose=False,
        )
        X = torch.tensor(res.X, **tkwargs)
        # multiply by negative one to return the correct sign for maximization
        Y = -torch.tensor(res.F, **tkwargs)
        pareto_mask = is_non_dominated(Y, deduplicate=True)
        X_pareto = X[pareto_mask]
        Y_pareto = Y[pareto_mask]
        if Y_pareto.shape[0] > q:
            Y_pareto, indices = get_hypervolume_maximizing_subset(
                # use nadir as reference point since we likely don't care about the
                # extrema as much as the interior
                n=q,
                Y=Y_pareto,
                ref_point=Y_pareto.min(dim=0).values,
            )
            X_pareto = X_pareto[indices]
        return X_pareto, Y_pareto

except ImportError:  # pragma: no cover
    pass
