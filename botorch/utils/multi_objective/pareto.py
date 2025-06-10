#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Callable

import numpy as np

import torch
from botorch.acquisition.multi_objective.objective import (
    IdentityMCMultiOutputObjective,
    MCMultiOutputObjective,
)
from botorch.models.model import Model
from torch import Tensor

# maximum tensor size for simple pareto computation
MAX_BYTES = 5e6


def is_non_dominated(
    Y: Tensor,
    maximize: bool = True,
    deduplicate: bool = True,
) -> Tensor:
    r"""Computes the non-dominated front.

    Note: this assumes maximization.

    For small `n`, this method uses a highly parallel methodology
    that compares all pairs of points in Y. However, this is memory
    intensive and slow for large `n`. For large `n` (or if Y is larger
    than 5MB), this method will dispatch to a loop-based approach
    that is faster and has a lower memory footprint.

    Args:
        Y: A `(batch_shape) x n x m`-dim tensor of outcomes.
            If any element of `Y` is NaN, the corresponding point
            will be treated as a dominated point (returning False).
        maximize: If True, assume maximization (default).
        deduplicate: A boolean indicating whether to only return
            unique points on the pareto frontier.

    Returns:
        A `(batch_shape) x n`-dim boolean tensor indicating whether
        each point is non-dominated.
    """
    n = Y.shape[-2]
    if n == 0:
        return torch.zeros(Y.shape[:-1], dtype=torch.bool, device=Y.device)
    el_size = 64 if Y.dtype == torch.double else 32
    if n > 1000 or n**2 * Y.shape[:-2].numel() * el_size / 8 > MAX_BYTES:
        return _is_non_dominated_loop(Y, maximize=maximize, deduplicate=deduplicate)

    Y1 = Y.unsqueeze(-3)
    Y2 = Y.unsqueeze(-2)
    if maximize:
        dominates = (Y1 >= Y2).all(dim=-1) & (Y1 > Y2).any(dim=-1)
    else:
        dominates = (Y1 <= Y2).all(dim=-1) & (Y1 < Y2).any(dim=-1)
    nd_mask = ~(dominates.any(dim=-1))
    if deduplicate:
        # remove duplicates
        # find index of first occurrence  of each unique element
        indices = (Y1 == Y2).all(dim=-1).long().argmax(dim=-1)
        keep = torch.zeros_like(nd_mask)
        keep.scatter_(dim=-1, index=indices, value=1.0)
        return nd_mask & keep
    return nd_mask


def _is_non_dominated_loop(
    Y: Tensor,
    maximize: bool = True,
    deduplicate: bool = True,
) -> Tensor:
    r"""Determine which points are non-dominated.

    Compared to `is_non_dominated`, this method is significantly
    faster for large `n` on a CPU and will significant reduce memory
    overhead. However, `is_non_dominated` is faster for smaller problems.

    Args:
        Y: A `(batch_shape) x n x m` Tensor of outcomes.
        maximize: If True, assume maximization (default).
        deduplicate: A boolean indicating whether to only return unique points on
            the pareto frontier.

    Returns:
        A `(batch_shape) x n`-dim Tensor of booleans indicating whether each point is
            non-dominated.
    """
    is_efficient = torch.ones(*Y.shape[:-1], dtype=bool, device=Y.device)
    for i in range(Y.shape[-2]):
        i_is_efficient = is_efficient[..., i]
        if i_is_efficient.any():
            vals = Y[..., i : i + 1, :]
            if maximize:
                update = (Y > vals).any(dim=-1)
            else:
                update = (Y < vals).any(dim=-1)
            # If an element in Y[..., i, :] is efficient, mark it as efficient
            update[..., i] = i_is_efficient.clone()
            # Only include batches where  Y[..., i, :] is efficient
            # Create a copy
            is_efficient2 = is_efficient.clone()
            if Y.ndim > 2:
                # Set all elements in all batches where Y[..., i, :] is not
                # efficient to False
                is_efficient2[~i_is_efficient] = False
            # Only include elements from is_efficient from the batches
            # where Y[..., i, :] is efficient
            is_efficient[is_efficient2] = update[is_efficient2]

    if not deduplicate:
        # Doing another pass over the data to remove duplicates. There may be a
        # more efficient way to do this. One could broadcast this as in
        # `is_non_dominated`, but we loop here to avoid high memory usage.
        is_efficient_dedup = is_efficient.clone()
        for i in range(Y.shape[-2]):
            i_is_efficient = is_efficient[..., i]
            if i_is_efficient.any():
                vals = Y[..., i : i + 1, :]
                duplicate = (vals == Y).all(dim=-1) & i_is_efficient.unsqueeze(-1)
                if duplicate.any():
                    is_efficient_dedup[duplicate] = True
        return is_efficient_dedup

    return is_efficient


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
            model: Model,
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
                model: A fitted model.
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
            self.botorch_model = model
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
                y = self.botorch_model.posterior(X=X.unsqueeze(-2)).mean.squeeze(-2)
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
        model: Model,
        bounds: Tensor,
        num_objectives: int,
        ref_point: list[float] | Tensor | None = None,
        objective: MCMultiOutputObjective | None = None,
        constraints: list[Callable[[Tensor], Tensor]] | None = None,
        population_size: int = 250,
        max_gen: int | None = None,
        seed: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Optimize the posterior mean via NSGA-II, returning the Pareto set and front.

        This assumes maximization of all objectives.

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

        Returns:
            A two-element tuple containing the pareto set X and pareto frontier Y.
        """
        tkwargs = {"dtype": bounds.dtype, "device": bounds.device}
        if ref_point is not None:
            ref_point = torch.as_tensor(ref_point, **tkwargs)
        pymoo_problem = BotorchPymooProblem(
            n_var=bounds.shape[-1],
            n_obj=num_objectives,
            xl=bounds[0].cpu().numpy(),
            xu=bounds[1].cpu().numpy(),
            model=model,
            ref_point=ref_point,
            objective=objective,
            constraints=constraints,
            **tkwargs,
        )
        algorithm = NSGA2(pop_size=population_size, eliminate_duplicates=True)
        res = minimize(
            problem=pymoo_problem,
            algorithm=algorithm,
            termination=None
            if max_gen is None
            else MaximumGenerationTermination(n_max_gen=max_gen),
            seed=seed,
            verbose=False,
        )
        X = torch.tensor(res.X, **tkwargs)
        # multiply by negative one to return the correct sign for maximization
        Y = -torch.tensor(res.F, **tkwargs)
        pareto_mask = is_non_dominated(Y, deduplicate=True)
        return X[pareto_mask], Y[pareto_mask]
except ImportError:  # pragma: no cover
    pass
