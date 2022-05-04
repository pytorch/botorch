#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Monte-Carlo Acquisition Functions for Multi-objective Bayesian optimization.

References

.. [Daulton2020qehvi]
    S. Daulton, M. Balandat, and E. Bakshy. Differentiable Expected Hypervolume
    Improvement for Parallel Multi-Objective Bayesian Optimization. Advances in Neural
    Information Processing Systems 33, 2020.

.. [Daulton2021nehvi]
    S. Daulton, M. Balandat, and E. Bakshy. Parallel Bayesian Optimization of
    Multiple Noisy Objectives with Expected Hypervolume Improvement. Advances
    in Neural Information Processing Systems 34, 2021.

"""

from __future__ import annotations

import warnings
from abc import abstractmethod
from copy import deepcopy
from itertools import combinations
from typing import Any, Callable, List, Optional, Union

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.cached_cholesky import CachedCholeskyMCAcquisitionFunction
from botorch.acquisition.multi_objective.objective import (
    IdentityMCMultiOutputObjective,
    MCMultiOutputObjective,
)
from botorch.acquisition.multi_objective.utils import (
    prune_inferior_points_multi_objective,
)
from botorch.exceptions.errors import UnsupportedError
from botorch.exceptions.warnings import BotorchWarning
from botorch.models.model import Model
from botorch.models.transforms.input import InputPerturbation
from botorch.posteriors import DeterministicPosterior
from botorch.posteriors.posterior import Posterior
from botorch.sampling.samplers import MCSampler, SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.box_decomposition_list import (
    BoxDecompositionList,
)
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
    NondominatedPartitioning,
)
from botorch.utils.multi_objective.box_decompositions.utils import (
    _pad_batch_pareto_frontier,
)
from botorch.utils.objective import apply_constraints_nonnegative_soft
from botorch.utils.torch import BufferDict
from botorch.utils.transforms import (
    concatenate_pending_points,
    is_fully_bayesian,
    match_batch_shape,
    t_batch_mode_transform,
)
from torch import Tensor


class MultiObjectiveMCAcquisitionFunction(AcquisitionFunction):
    r"""Abstract base class for Multi-Objective batch acquisition functions."""

    def __init__(
        self,
        model: Model,
        sampler: Optional[MCSampler] = None,
        objective: Optional[MCMultiOutputObjective] = None,
        constraints: Optional[List[Callable[[Tensor], Tensor]]] = None,
        X_pending: Optional[Tensor] = None,
    ) -> None:
        r"""Constructor for the MCAcquisitionFunction base class.

        Args:
            model: A fitted model.
            sampler: The sampler used to draw base samples. Defaults to
                `SobolQMCNormalSampler(num_samples=128, collapse_batch_dims=True)`.
            objective: The MCMultiOutputObjective under which the samples are
                evaluated. Defaults to `IdentityMultiOutputObjective()`.
            constraints: A list of callables, each mapping a Tensor of dimension
                `sample_shape x batch-shape x q x m` to a Tensor of dimension
                `sample_shape x batch-shape x q`, where negative values imply
                feasibility.
            X_pending:  A `m x d`-dim Tensor of `m` design points that have
                points that have been submitted for function evaluation
                but have not yet been evaluated.
        """
        super().__init__(model=model)
        if sampler is None:
            sampler = SobolQMCNormalSampler(num_samples=128, collapse_batch_dims=True)
        self.add_module("sampler", sampler)
        if objective is None:
            objective = IdentityMCMultiOutputObjective()
        elif not isinstance(objective, MCMultiOutputObjective):
            raise UnsupportedError(
                "Only objectives of type MCMultiOutputObjective are supported for "
                "Multi-Objective MC acquisition functions."
            )
        if (
            hasattr(model, "input_transform")
            and isinstance(model.input_transform, InputPerturbation)
            and constraints is not None
        ):
            raise UnsupportedError(
                "Constraints are not supported with input perturbations, due to"
                "sample q-batch shape being different than that of the inputs."
                "Use a composite objective that applies feasibility weighting to"
                "samples before calculating the risk measure."
            )
        self.add_module("objective", objective)
        self.constraints = constraints
        self.X_pending = None
        if X_pending is not None:
            self.set_X_pending(X_pending)

    @abstractmethod
    def forward(self, X: Tensor) -> Tensor:
        r"""Takes in a `batch_shape x q x d` X Tensor of t-batches with `q` `d`-dim
        design points each, and returns a Tensor with shape `batch_shape'`, where
        `batch_shape'` is the broadcasted batch shape of model and input `X`. Should
        utilize the result of `set_X_pending` as needed to account for pending function
        evaluations.
        """
        pass  # pragma: no cover


class qExpectedHypervolumeImprovement(MultiObjectiveMCAcquisitionFunction):
    def __init__(
        self,
        model: Model,
        ref_point: Union[List[float], Tensor],
        partitioning: NondominatedPartitioning,
        sampler: Optional[MCSampler] = None,
        objective: Optional[MCMultiOutputObjective] = None,
        constraints: Optional[List[Callable[[Tensor], Tensor]]] = None,
        X_pending: Optional[Tensor] = None,
        eta: float = 1e-3,
    ) -> None:
        r"""q-Expected Hypervolume Improvement supporting m>=2 outcomes.

        See [Daulton2020qehvi]_ for details.

        Example:
            >>> model = SingleTaskGP(train_X, train_Y)
            >>> ref_point = [0.0, 0.0]
            >>> qEHVI = qExpectedHypervolumeImprovement(model, ref_point, partitioning)
            >>> qehvi = qEHVI(test_X)

        Args:
            model: A fitted model.
            ref_point: A list or tensor with `m` elements representing the reference
                point (in the outcome space) w.r.t. to which compute the hypervolume.
                This is a reference point for the objective values (i.e. after
                applying`objective` to the samples).
            partitioning: A `NondominatedPartitioning` module that provides the non-
                dominated front and a partitioning of the non-dominated space in hyper-
                rectangles. If constraints are present, this partitioning must only
                include feasible points.
            sampler: The sampler used to draw base samples. Defaults to
                `SobolQMCNormalSampler(num_samples=128, collapse_batch_dims=True)`.
            objective: The MCMultiOutputObjective under which the samples are evaluated.
                Defaults to `IdentityMultiOutputObjective()`.
            constraints: A list of callables, each mapping a Tensor of dimension
                `sample_shape x batch-shape x q x m` to a Tensor of dimension
                `sample_shape x batch-shape x q`, where negative values imply
                feasibility. The acqusition function will compute expected feasible
                hypervolume.
            X_pending: A `batch_shape x m x d`-dim Tensor of `m` design points that have
                points that have been submitted for function evaluation but have not yet
                been evaluated. Concatenated into `X` upon forward call. Copied and set
                to have no gradient.
            eta: The temperature parameter for the sigmoid function used for the
                differentiable approximation of the constraints.
        """
        if len(ref_point) != partitioning.num_outcomes:
            raise ValueError(
                "The length of the reference point must match the number of outcomes. "
                f"Got ref_point with {len(ref_point)} elements, but expected "
                f"{partitioning.num_outcomes}."
            )
        ref_point = torch.as_tensor(
            ref_point,
            dtype=partitioning.pareto_Y.dtype,
            device=partitioning.pareto_Y.device,
        )
        super().__init__(
            model=model,
            sampler=sampler,
            objective=objective,
            constraints=constraints,
            X_pending=X_pending,
        )
        self.eta = eta
        self.register_buffer("ref_point", ref_point)
        cell_bounds = partitioning.get_hypercell_bounds()
        self.register_buffer("cell_lower_bounds", cell_bounds[0])
        self.register_buffer("cell_upper_bounds", cell_bounds[1])
        self.q_out = -1
        self.q_subset_indices = BufferDict()

    def _cache_q_subset_indices(self, q_out: int) -> None:
        r"""Cache indices corresponding to all subsets of `q_out`.

        This means that consecutive calls to `forward` with the same
        `q_out` will not recompute the indices for all (2^q_out - 1) subsets.

        Note: this will use more memory than regenerating the indices
        for each i and then deleting them, but it will be faster for
        repeated evaluations (e.g. during optimization).

        Args:
            q_out: The batch size of the objectives. This is typically equal
                to the q-batch size of `X`. However, if using a set valued
                objective (e.g., MVaR) that produces `s` objective values for
                each point on the q-batch of `X`, we need to properly account
                for each objective while calculating the hypervolume contributions
                by using `q_out = q * s`.
        """
        if q_out != self.q_out:
            indices = list(range(q_out))
            tkwargs = {"dtype": torch.long, "device": self.ref_point.device}
            self.q_subset_indices = BufferDict(
                {
                    f"q_choose_{i}": torch.tensor(
                        list(combinations(indices, i)), **tkwargs
                    )
                    for i in range(1, q_out + 1)
                }
            )
            self.q_out = q_out

    def _compute_qehvi(self, samples: Tensor, X: Optional[Tensor] = None) -> Tensor:
        r"""Compute the expected (feasible) hypervolume improvement given MC samples.

        Args:
            samples: A `n_samples x batch_shape x q' x m`-dim tensor of samples.
            X: A `batch_shape x q x d`-dim tensor of inputs.

        Returns:
            A `batch_shape x (model_batch_shape)`-dim tensor of expected hypervolume
            improvement for each batch.
        """
        # Note that the objective may subset the outcomes (e.g. this will usually happen
        # if there are constraints present).
        obj = self.objective(samples, X=X)
        q = obj.shape[-2]
        if self.constraints is not None:
            feas_weights = torch.ones(
                obj.shape[:-1], device=obj.device, dtype=obj.dtype
            )
            feas_weights = apply_constraints_nonnegative_soft(
                obj=feas_weights,
                constraints=self.constraints,
                samples=samples,
                eta=self.eta,
            )
        self._cache_q_subset_indices(q_out=q)
        batch_shape = obj.shape[:-2]
        # this is n_samples x input_batch_shape x
        areas_per_segment = torch.zeros(
            *batch_shape,
            self.cell_lower_bounds.shape[-2],
            dtype=obj.dtype,
            device=obj.device,
        )
        cell_batch_ndim = self.cell_lower_bounds.ndim - 2
        sample_batch_view_shape = torch.Size(
            [
                batch_shape[0] if cell_batch_ndim > 0 else 1,
                *[1 for _ in range(len(batch_shape) - max(cell_batch_ndim, 1))],
                *self.cell_lower_bounds.shape[1:-2],
            ]
        )
        view_shape = (
            *sample_batch_view_shape,
            self.cell_upper_bounds.shape[-2],
            1,
            self.cell_upper_bounds.shape[-1],
        )
        for i in range(1, self.q_out + 1):
            # TODO: we could use batches to compute (q choose i) and (q choose q-i)
            # simultaneously since subsets of size i and q-i have the same number of
            # elements. This would decrease the number of iterations, but increase
            # memory usage.
            q_choose_i = self.q_subset_indices[f"q_choose_{i}"]
            # this tensor is mc_samples x batch_shape x i x q_choose_i x m
            obj_subsets = obj.index_select(dim=-2, index=q_choose_i.view(-1))
            obj_subsets = obj_subsets.view(
                obj.shape[:-2] + q_choose_i.shape + obj.shape[-1:]
            )
            # since all hyperrectangles share one vertex, the opposite vertex of the
            # overlap is given by the component-wise minimum.
            # take the minimum in each subset
            overlap_vertices = obj_subsets.min(dim=-2).values
            # add batch-dim to compute area for each segment (pseudo-pareto-vertex)
            # this tensor is mc_samples x batch_shape x num_cells x q_choose_i x m
            overlap_vertices = torch.min(
                overlap_vertices.unsqueeze(-3), self.cell_upper_bounds.view(view_shape)
            )
            # substract cell lower bounds, clamp min at zero
            lengths_i = (
                overlap_vertices - self.cell_lower_bounds.view(view_shape)
            ).clamp_min(0.0)
            # take product over hyperrectangle side lengths to compute area
            # sum over all subsets of size i
            areas_i = lengths_i.prod(dim=-1)
            # if constraints are present, apply a differentiable approximation of
            # the indicator function
            if self.constraints is not None:
                feas_subsets = feas_weights.index_select(
                    dim=-1, index=q_choose_i.view(-1)
                ).view(feas_weights.shape[:-1] + q_choose_i.shape)
                areas_i = areas_i * feas_subsets.unsqueeze(-3).prod(dim=-1)
            areas_i = areas_i.sum(dim=-1)
            # Using the inclusion-exclusion principle, set the sign to be positive
            # for subsets of odd sizes and negative for subsets of even size
            areas_per_segment += (-1) ** (i + 1) * areas_i
        # sum over segments and average over MC samples
        return areas_per_segment.sum(dim=-1).mean(dim=0)

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        posterior = self.model.posterior(X)
        samples = self.sampler(posterior)
        return self._compute_qehvi(samples=samples, X=X)


class qNoisyExpectedHypervolumeImprovement(
    qExpectedHypervolumeImprovement, CachedCholeskyMCAcquisitionFunction
):
    def __init__(
        self,
        model: Model,
        ref_point: Union[List[float], Tensor],
        X_baseline: Tensor,
        sampler: Optional[MCSampler] = None,
        objective: Optional[MCMultiOutputObjective] = None,
        constraints: Optional[List[Callable[[Tensor], Tensor]]] = None,
        X_pending: Optional[Tensor] = None,
        eta: float = 1e-3,
        prune_baseline: bool = False,
        alpha: float = 0.0,
        cache_pending: bool = True,
        max_iep: int = 0,
        incremental_nehvi: bool = True,
        cache_root: bool = True,
        **kwargs: Any,
    ) -> None:
        r"""q-Noisy Expected Hypervolume Improvement supporting m>=2 outcomes.

        See [Daulton2021nehvi]_ for details.

        Example:
            >>> model = SingleTaskGP(train_X, train_Y)
            >>> ref_point = [0.0, 0.0]
            >>> qNEHVI = qNoisyExpectedHypervolumeImprovement(model, ref_point, train_X)
            >>> qnehvi = qNEHVI(test_X)

        Args:
            model: A fitted model.
            ref_point: A list or tensor with `m` elements representing the reference
                point (in the outcome space) w.r.t. to which compute the hypervolume.
                This is a reference point for the objective values (i.e. after
                applying `objective` to the samples).
            X_baseline: A `r x d`-dim Tensor of `r` design points that have already
                been observed. These points are considered as potential approximate
                pareto-optimal design points.
            sampler: The sampler used to draw base samples. Defaults to
                `SobolQMCNormalSampler(num_samples=128, collapse_batch_dims=True)`.
                Note: a pareto front is created for each mc sample, which can be
                computationally intensive for `m` > 2.
            objective: The MCMultiOutputObjective under which the samples are
                evaluated. Defaults to `IdentityMultiOutputObjective()`.
            constraints: A list of callables, each mapping a Tensor of dimension
                `sample_shape x batch-shape x q x m` to a Tensor of dimension
                `sample_shape x batch-shape x q`, where negative values imply
                feasibility. The acqusition function will compute expected feasible
                hypervolume.
            X_pending: A `batch_shape x m x d`-dim Tensor of `m` design points that
                have points that have been submitted for function evaluation, but
                have not yet been evaluated.
            eta: The temperature parameter for the sigmoid function used for the
                differentiable approximation of the constraints.
            prune_baseline: If True, remove points in `X_baseline` that are
                highly unlikely to be the pareto optimal and better than the
                reference point. This can significantly improve computation time and
                is generally recommended. In order to customize pruning parameters,
                instead manually call `prune_inferior_points_multi_objective` on
                `X_baseline` before instantiating the acquisition function.
            alpha: The hyperparameter controlling the approximate non-dominated
                partitioning. The default value of 0.0 means an exact partitioning
                is used. As the number of objectives `m` increases, consider increasing
                this parameter in order to limit computational complexity.
            cache_pending: A boolean indicating whether to use cached box
                decompositions (CBD) for handling pending points. This is
                generally recommended.
            max_iep: The maximum number of pending points before the box
                decompositions will be recomputed.
            incremental_nehvi: A boolean indicating whether to compute the
                incremental NEHVI from the `i`th point where `i=1, ..., q`
                under sequential greedy optimization, or the full qNEHVI over
                `q` points.
            cache_root: A boolean indicating whether to cache the root
                decomposition over `X_baseline` and use low-rank updates.
        """
        ref_point = torch.as_tensor(
            ref_point, dtype=X_baseline.dtype, device=X_baseline.device
        )
        super(qExpectedHypervolumeImprovement, self).__init__(
            model=model,
            sampler=sampler,
            objective=objective,
            constraints=constraints,
        )
        self._setup(
            model=model, sampler=self.sampler, cache_root=cache_root, check_sampler=True
        )

        if X_baseline.ndim > 2:
            raise UnsupportedError(
                "qNoisyExpectedHypervolumeImprovement does not support batched "
                f"X_baseline. Expected 2 dims, got {X_baseline.ndim}."
            )
        if prune_baseline:
            X_baseline = prune_inferior_points_multi_objective(
                model=model,
                X=X_baseline,
                objective=objective,
                constraints=constraints,
                ref_point=ref_point,
                marginalize_dim=kwargs.get("marginalize_dim"),
            )
        self.register_buffer("ref_point", ref_point)
        self.base_sampler = deepcopy(self.sampler)

        self.alpha = alpha
        self.eta = eta
        self.q_in = -1
        self.q_out = -1
        self.q_subset_indices = BufferDict()
        self.partitioning = None
        # set partitioning class and args
        self.p_kwargs = {}
        if self.alpha > 0:
            self.p_kwargs["alpha"] = self.alpha
            self.p_class = NondominatedPartitioning
        else:
            self.p_class = FastNondominatedPartitioning
        self.register_buffer("_X_baseline", X_baseline)
        self.register_buffer("_X_baseline_and_pending", X_baseline)
        self.register_buffer(
            "cache_pending",
            torch.tensor(cache_pending, dtype=bool),
        )
        self.register_buffer(
            "_prev_nehvi",
            torch.tensor(0.0, dtype=ref_point.dtype, device=ref_point.device),
        )
        self.register_buffer(
            "_max_iep",
            torch.tensor(max_iep, dtype=torch.long),
        )
        self.register_buffer(
            "incremental_nehvi",
            torch.tensor(incremental_nehvi, dtype=torch.bool),
        )

        if X_pending is not None:
            # This will call self._set_cell_bounds if the number of pending
            # points is greater than self._max_iep.
            self.set_X_pending(X_pending)
        # In the case that X_pending is not None, but there are fewer than
        # max_iep pending points, the box decompositions are not performed in
        # set_X_pending. Therefore, we need to perform a box decomposition over
        # f(X_baseline) here.
        if X_pending is None or X_pending.shape[-2] <= self._max_iep:
            self._set_cell_bounds(num_new_points=X_baseline.shape[0])

    @property
    def X_baseline(self) -> Tensor:
        r"""Return X_baseline augmented with pending points cached using CBD."""
        return self._X_baseline_and_pending

    def _compute_initial_hvs(self, obj: Tensor, feas: Optional[Tensor] = None) -> None:
        r"""Compute hypervolume dominated by f(X_baseline) under each sample.

        Args:
            obj: A `sample_shape x batch_shape x n x m`-dim tensor of samples
                of objectives.
            feas: `sample_shape x batch_shape x n`-dim tensor of samples
                of feasibility indicators.
        """
        initial_hvs = []
        for i, sample in enumerate(obj):
            if self.constraints is not None:
                sample = sample[feas[i]]
            dominated_partitioning = DominatedPartitioning(
                ref_point=self.ref_point,
                Y=sample,
            )
            hv = dominated_partitioning.compute_hypervolume()
            initial_hvs.append(hv)
        self.register_buffer(
            "_initial_hvs",
            torch.tensor(initial_hvs, dtype=obj.dtype, device=obj.device).view(
                self._batch_sample_shape, *obj.shape[-2:]
            ),
        )

    def _set_cell_bounds(self, num_new_points: int) -> None:
        r"""Compute the box decomposition under each posterior sample.

        Args:
            num_new_points: The number of new points (beyond the points
                in X_baseline) that were used in the previous box decomposition.
                In the first box decomposition, this should be the number of points
                in X_baseline.
        """
        feas = None
        if self.X_baseline.shape[0] > 0:
            with torch.no_grad():
                posterior = self.model.posterior(self.X_baseline)
            # Reset sampler, accounting for possible one-to-many transform.
            self.q_in = -1
            n_w = posterior.event_shape[-2] // self.X_baseline.shape[-2]
            self._set_sampler(q_in=num_new_points * n_w, posterior=posterior)
            # set base_sampler
            self.base_sampler.register_buffer(
                "base_samples", self.sampler.base_samples.detach().clone()
            )

            samples = self.base_sampler(posterior)
            # cache posterior
            if self._cache_root:
                self._cache_root_decomposition(posterior=posterior)
            obj = self.objective(samples, X=self.X_baseline)
            if self.constraints is not None:
                feas = torch.stack(
                    [c(samples) <= 0 for c in self.constraints], dim=0
                ).all(dim=0)
        else:
            obj = torch.empty(
                *self.sampler._sample_shape,
                0,
                self.ref_point.shape[-1],
                dtype=self.ref_point.dtype,
                device=self.ref_point.device,
            )
        self._batch_sample_shape = obj.shape[:-2]
        # collapse batch dimensions
        # use numel() rather than view(-1) to handle case of no baseline points
        new_batch_shape = self._batch_sample_shape.numel()
        obj = obj.view(new_batch_shape, *obj.shape[-2:])
        if self.constraints is not None and feas is not None:
            feas = feas.view(new_batch_shape, *feas.shape[-1:])

        if self.partitioning is None and not self.incremental_nehvi:
            self._compute_initial_hvs(obj=obj, feas=feas)
        if self.ref_point.shape[-1] > 2:
            # the partitioning algorithms run faster on the CPU
            # due to advanced indexing
            ref_point_cpu = self.ref_point.cpu()
            obj_cpu = obj.cpu()
            if self.constraints is not None and feas is not None:
                feas_cpu = feas.cpu()
                obj_cpu = [obj_cpu[i][feas_cpu[i]] for i in range(obj.shape[0])]
            partitionings = []
            for sample in obj_cpu:
                partitioning = self.p_class(
                    ref_point=ref_point_cpu, Y=sample, **self.p_kwargs
                )
                partitionings.append(partitioning)
            self.partitioning = BoxDecompositionList(*partitionings)
        else:
            # use batched partitioning
            obj = _pad_batch_pareto_frontier(
                Y=obj,
                ref_point=self.ref_point.unsqueeze(0).expand(
                    obj.shape[0], self.ref_point.shape[-1]
                ),
                feasibility_mask=feas,
            )
            self.partitioning = self.p_class(
                ref_point=self.ref_point, Y=obj, **self.p_kwargs
            )
        cell_bounds = self.partitioning.get_hypercell_bounds().to(self.ref_point)
        cell_bounds = cell_bounds.view(
            2, *self._batch_sample_shape, *cell_bounds.shape[-2:]
        )
        self.register_buffer("cell_lower_bounds", cell_bounds[0])
        self.register_buffer("cell_upper_bounds", cell_bounds[1])

    def set_X_pending(self, X_pending: Optional[Tensor] = None) -> None:
        r"""Informs the acquisition function about pending design points.

        Args:
            X_pending: `n x d` Tensor with `n` `d`-dim design points that have
                been submitted for evaluation but have not yet been evaluated.
        """
        if X_pending is None:
            self.X_pending = None
        else:
            if X_pending.requires_grad:
                warnings.warn(
                    "Pending points require a gradient but the acquisition function"
                    " will not provide a gradient to these points.",
                    BotorchWarning,
                )
            X_pending = X_pending.detach().clone()
            if self.cache_pending:
                X_baseline = torch.cat([self._X_baseline, X_pending], dim=-2)
                # Number of new points is the total number of points minus
                # (the number of previously cached pending points plus the
                # of number of baseline points).
                num_new_points = X_baseline.shape[0] - self.X_baseline.shape[0]
                if num_new_points > 0:
                    if num_new_points > self._max_iep:
                        # Set the new baseline points to include pending points.
                        self.register_buffer("_X_baseline_and_pending", X_baseline)
                        # Recompute box decompositions.
                        self._set_cell_bounds(num_new_points=num_new_points)
                        if not self.incremental_nehvi:
                            self._prev_nehvi = (
                                (self._hypervolumes - self._initial_hvs)
                                .clamp_min(0.0)
                                .mean()
                            )
                        # Set to None so that pending points are not concatenated in
                        # forward.
                        self.X_pending = None
                        # Set q_in=-1 to so that self.sampler is updated at the next
                        # forward call.
                        self.q_in = -1
                    else:
                        self.X_pending = X_pending[-num_new_points:]
            else:
                self.X_pending = X_pending

    @property
    def _hypervolumes(self) -> Tensor:
        r"""Compute hypervolume over X_baseline under each posterior sample.

        Returns:
            A `n_samples`-dim tensor of hypervolumes.
        """
        return (
            self.partitioning.compute_hypervolume()
            .to(self.ref_point)  # for m > 2, the partitioning is on the CPU
            .view(self._batch_sample_shape)
        )

    def _set_sampler(
        self,
        q_in: int,
        posterior: Posterior,
    ) -> None:
        r"""Update the sampler to use the original base samples for X_baseline.

        Args:
            q_in: The effective input batch size. This is typically equal to the
                q-batch size of `X`. However, if using a one-to-many input transform,
                e.g., `InputPerturbation` with `n_w` perturbations, the posterior will
                have `n_w` points on the q-batch for each point on the q-batch of `X`.
                In which case, `q_in = q * n_w` is used.
            posterior: The posterior.

        TODO: refactor some/all of this into the MCSampler.
        """
        if self.q_in != q_in:
            # create new base_samples
            base_sample_shape = self.sampler._get_base_sample_shape(posterior=posterior)
            self.sampler._construct_base_samples(
                posterior=posterior, shape=base_sample_shape
            )
            if (
                self.X_baseline.shape[0] > 0
                and self.base_sampler.base_samples is not None
                and not isinstance(posterior, DeterministicPosterior)
            ):
                current_base_samples = self.base_sampler.base_samples.detach().clone()
                # This is the # of non-`sample_shape` dimensions.
                base_ndims = current_base_samples.dim() - 1
                # Unsqueeze as many dimensions as needed to match base_sample_shape.
                view_shape = (
                    self.sampler.sample_shape
                    + torch.Size(
                        [1] * (len(base_sample_shape) - current_base_samples.dim())
                    )
                    + current_base_samples.shape[-base_ndims:]
                )
                expanded_shape = (
                    base_sample_shape[:-base_ndims]
                    + current_base_samples.shape[-base_ndims:]
                )
                # Use stored base samples:
                # Use all base_samples from the current sampler
                # this includes the base_samples from the base_sampler
                # and any base_samples for the new points in the sampler.
                # For example, when using sequential greedy candidate generation
                # then generate the new candidate point using last (-1) base_sample
                # in sampler. This copies that base sample.
                expanded_samples = current_base_samples.view(view_shape).expand(
                    expanded_shape
                )
                if self._uses_matheron:
                    n_train_samples = current_base_samples.shape[-1] // 2
                    # The train base samples.
                    self.sampler.base_samples[..., :n_train_samples] = expanded_samples[
                        ..., :n_train_samples
                    ]
                    # The train noise base samples.
                    self.sampler.base_samples[
                        ..., -n_train_samples:
                    ] = expanded_samples[..., -n_train_samples:]
                else:
                    end_idx = current_base_samples.shape[-2]
                    self.sampler.base_samples[..., :end_idx, :] = expanded_samples
                self.q_in = q_in

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        X_full = torch.cat([match_batch_shape(self.X_baseline, X), X], dim=-2)
        # Note: it is important to compute the full posterior over `(X_baseline, X)`
        # to ensure that we properly sample `f(X)` from the joint distribution `
        # `f(X_baseline, X) ~ P(f | D)` given that we can already fixed the sampled
        # function values for `f(X_baseline)`.
        # TODO: improve efficiency by not recomputing baseline-baseline
        # covariance matrix.
        posterior = self.model.posterior(X_full)
        # Account for possible one-to-many transform and the MCMC batch dimension in
        # `SaasFullyBayesianSingleTaskGP`
        event_shape_lag = 1 if is_fully_bayesian(self.model) else 2
        n_w = posterior.event_shape[X_full.dim() - event_shape_lag] // X_full.shape[-2]
        q_in = X.shape[-2] * n_w
        self._set_sampler(q_in=q_in, posterior=posterior)
        samples = self._get_f_X_samples(posterior=posterior, q_in=q_in)
        # Add previous nehvi from pending points.
        return self._compute_qehvi(samples=samples, X=X) + self._prev_nehvi
