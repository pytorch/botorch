#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Algorithms for partitioning the non-dominated space into rectangles.

References

.. [Couckuyt2012]
    I. Couckuyt, D. Deschrijver and T. Dhaene, "Towards Efficient
    Multiobjective Optimization: Multiobjective statistical criterions,"
    2012 IEEE Congress on Evolutionary Computation, Brisbane, QLD, 2012,
    pp. 1-8.

"""

from __future__ import annotations

from typing import Optional

import torch
from botorch.exceptions.errors import BotorchError, BotorchTensorDimensionError
from botorch.utils.multi_objective.pareto import is_non_dominated
from torch import Tensor
from torch.nn import Module


class NondominatedPartitioning(Module):
    r"""A class for partitioning the non-dominated space into hyper-cells.

    Note: this assumes maximization. Internally, it multiplies by -1 and performs
    the decomposition under minimization. TODO: use maximization internally as well.

    Note: it is only feasible to use this algorithm to compute an exact
    decomposition of the non-dominated space for `m<5` objectives (alpha=0.0).

    The alpha parameter can be increased to obtain an approximate partitioning
    faster. The `alpha` is a fraction of the total hypervolume encapsuling the
    entire pareto set. When a hypercell's volume divided by the total hypervolume
    is less than `alpha`, we discard the hypercell. See Figure 2 in
    [Couckuyt2012]_ for a visual representation.

    This PyTorch implementation of the binary partitioning algorithm ([Couckuyt2012]_)
    is adapted from numpy/tensorflow implementation at:
    https://github.com/GPflow/GPflowOpt/blob/master/gpflowopt/pareto.py.

    TODO: replace this with a more efficient decomposition. E.g.
    https://link.springer.com/content/pdf/10.1007/s10898-019-00798-7.pdf
    """

    def __init__(
        self,
        num_outcomes: int,
        Y: Optional[Tensor] = None,
        alpha: float = 0.0,
        eps: Optional[float] = None,
    ) -> None:
        """Initialize NondominatedPartitioning.

        Args:
            num_outcomes: The number of outcomes
            Y: A `n x m`-dim tensor
            alpha: a thresold fraction of total volume used in an approximate
                decomposition.
            eps: a small value for numerical stability
        """
        super().__init__()
        self.alpha = alpha
        self.num_outcomes = num_outcomes
        self._eps = eps
        if Y is not None:
            self.update(Y=Y)

    @property
    def eps(self) -> float:
        if self._eps is not None:
            return self._eps
        try:
            return 1e-6 if self._pareto_Y.dtype == torch.float else 1e-8
        except AttributeError:
            return 1e-6

    @property
    def pareto_Y(self) -> Tensor:
        r"""This returns the non-dominated set.

        Note: Internally, we store the negative pareto set (minimization).

        Returns:
            A `n_pareto x m`-dim tensor of outcomes.
        """
        if not hasattr(self, "_pareto_Y"):
            raise BotorchError("pareto_Y has not been initialized")
        return -self._pareto_Y

    def _update_pareto_Y(self) -> bool:
        r"""Update the non-dominated front."""
        # is_non_dominated assumes maximization
        non_dominated_mask = is_non_dominated(-self.Y)
        pf = self.Y[non_dominated_mask]
        # sort by first objective
        new_pareto_Y = pf[torch.argsort(pf[:, 0])]
        if not hasattr(self, "_pareto_Y") or not torch.equal(
            new_pareto_Y, self._pareto_Y
        ):
            self.register_buffer("_pareto_Y", new_pareto_Y)
            return True
        return False

    def update(self, Y: Tensor) -> None:
        r"""Update non-dominated front and decomposition.

        Args:
            Y: A `n x m`-dim tensor of outcomes.
        """
        # multiply by -1, since internally we minimize.
        self.Y = -Y
        is_new_pareto = self._update_pareto_Y()
        # Update decomposition if the pareto front changed
        if is_new_pareto:
            if self.num_outcomes > 2:
                self.binary_partition_non_dominated_space()
            else:
                self.partition_non_dominated_space_2d()

    def binary_partition_non_dominated_space(self):
        r"""Partition the non-dominated space into disjoint hypercells.

        This method works for an arbitrary number of outcomes, but is
        less efficient than `partition_non_dominated_space_2d` for the
        2-outcome case.
        """
        # Extend pareto front with the ideal and anti-ideal point
        ideal_point = self._pareto_Y.min(dim=0, keepdim=True).values - 1
        anti_ideal_point = self._pareto_Y.max(dim=0, keepdim=True).values + 1

        aug_pareto_Y = torch.cat([ideal_point, self._pareto_Y, anti_ideal_point], dim=0)
        # The binary parititoning algorithm uses indices the augmented pareto front.
        aug_pareto_Y_idcs = self._get_augmented_pareto_front_indices()

        # Initialize one cell over entire pareto front
        cell = torch.zeros(2, self.num_outcomes, dtype=torch.long, device=self.Y.device)
        cell[1] = aug_pareto_Y_idcs.shape[0] - 1
        stack = [cell]
        total_volume = (anti_ideal_point - ideal_point).prod()

        # hypercells contains the indices of the (augmented) pareto front
        # that specify that bounds of the each hypercell.
        # It is a `2 x num_cells x num_outcomes`-dim tensor
        self.register_buffer(
            "hypercells",
            torch.empty(
                2, 0, self.num_outcomes, dtype=torch.long, device=self.Y.device
            ),
        )
        outcome_idxr = torch.arange(
            self.num_outcomes, dtype=torch.long, device=self.Y.device
        )

        # Use binary partitioning
        while len(stack) > 0:
            cell = stack.pop()
            cell_bounds_pareto_idcs = aug_pareto_Y_idcs[cell, outcome_idxr]
            cell_bounds_pareto_values = aug_pareto_Y[
                cell_bounds_pareto_idcs, outcome_idxr
            ]
            # Check cell bounds
            # - if cell upper bound is better than pareto front on all outcomes:
            #   - accept the cell
            # - elif cell lower bound is better than pareto front on all outcomes:
            #   - this means the cell overlaps the pareto front. Divide the cell along
            #   - its longest edge.
            if (
                ((cell_bounds_pareto_values[1] - self.eps) < self._pareto_Y)
                .any(dim=1)
                .all()
            ):
                # Cell is entirely non-dominated
                self.hypercells = torch.cat(
                    [self.hypercells, cell_bounds_pareto_idcs.unsqueeze(1)], dim=1
                )
            elif (
                ((cell_bounds_pareto_values[0] + self.eps) < self._pareto_Y)
                .any(dim=1)
                .all()
            ):
                # The cell overlaps the pareto front
                # compute the distance (in integer indices)
                idx_dist = cell[1] - cell[0]
                any_not_adjacent = (idx_dist > 1).any()
                cell_volume = (
                    (cell_bounds_pareto_values[1] - cell_bounds_pareto_values[0])
                    .prod(dim=-1)
                    .item()
                )

                # Only divide a cell when it is not composed of adjacent indices
                # and the fraction of total volume is above the approximation
                # threshold fraction
                if (
                    any_not_adjacent
                    and ((cell_volume / total_volume) > self.alpha).all()
                ):
                    # Divide the test cell over its largest dimension
                    # largest (by index length)
                    length, longest_dim = torch.max(idx_dist, dim=0)
                    length = length.item()
                    longest_dim = longest_dim.item()

                    new_length1 = int(round(length / 2.0))
                    new_length2 = length - new_length1

                    # Store divided cells
                    # cell 1: subtract new_length1 from the upper bound of the cell
                    # cell 2: add new_length2 to the lower bound of the cell
                    for bound_idx, length_delta in (
                        (1, -new_length1),
                        (0, new_length2),
                    ):
                        new_cell = cell.clone()
                        new_cell[bound_idx, longest_dim] += length_delta
                        stack.append(new_cell)

    def partition_non_dominated_space_2d(self) -> None:
        r"""Partition the non-dominated space into disjoint hypercells.

        This direct method works for `m=2` outcomes.
        """
        if self.num_outcomes != 2:
            raise BotorchTensorDimensionError(
                "partition_non_dominated_space_2d requires 2 outputs, "
                f"but num_outcomes={self.num_outcomes}"
            )
        pf_ext_idx = self._get_augmented_pareto_front_indices()
        range_pf_plus1 = torch.arange(
            self._pareto_Y.shape[0] + 1, dtype=torch.long, device=self._pareto_Y.device
        )
        lower = torch.stack([range_pf_plus1, torch.zeros_like(range_pf_plus1)], dim=-1)
        upper = torch.stack(
            [range_pf_plus1 + 1, pf_ext_idx[-range_pf_plus1 - 1, -1]], dim=-1
        )
        self.register_buffer("hypercells", torch.stack([lower, upper], dim=0))

    def _get_augmented_pareto_front_indices(self) -> Tensor:
        r"""Get indices of augmented pareto front."""
        pf_idx = torch.argsort(self._pareto_Y, dim=0)
        return torch.cat(
            [
                torch.zeros(
                    1, self.num_outcomes, dtype=torch.long, device=self.Y.device
                ),
                # Add 1 because index zero is used for the ideal point
                pf_idx + 1,
                torch.full(
                    torch.Size([1, self.num_outcomes]),
                    self._pareto_Y.shape[0] + 1,
                    dtype=torch.long,
                    device=self.Y.device,
                ),
            ],
            dim=0,
        )

    def get_hypercell_bounds(self, ref_point: Tensor) -> Tensor:
        r"""Get the bounds of each hypercell in the decomposition.

        Args:
            ref_point: A `m`-dim tensor containing the reference point.

        Returns:
            A `2 x num_cells x num_outcomes`-dim tensor containing the
                lower and upper vertices bounding each hypercell.
        """
        aug_pareto_Y = torch.cat(
            [
                # -inf is the lower bound of the non-dominated space
                torch.full(
                    torch.Size([1, self.num_outcomes]),
                    float("-inf"),
                    dtype=self._pareto_Y.dtype,
                    device=self._pareto_Y.device,
                ),
                self._pareto_Y,
                # note: internally, this class minimizes, so use negative here
                -(ref_point.unsqueeze(0)),
            ],
            dim=0,
        )
        minimization_cell_bounds = self._get_hypercell_bounds(aug_pareto_Y=aug_pareto_Y)
        # swap upper and lower bounds and multiply by -1
        return torch.cat(
            [-minimization_cell_bounds[1:], -minimization_cell_bounds[:1]], dim=0
        )

    def _get_hypercell_bounds(self, aug_pareto_Y: Tensor) -> Tensor:
        r"""Get the bounds of each hypercell in the decomposition.

        Args:
            aug_pareto_Y: A `n_pareto + 2 x m`-dim tensor containing
            the augmented pareto front.

        Returns:
            A `2 x num_cells x num_outcomes`-dim tensor containing the
                lower and upper vertices bounding each hypercell.
        """
        num_cells = self.hypercells.shape[1]
        outcome_idxr = torch.arange(
            self.num_outcomes, dtype=torch.long, device=self.Y.device
        ).repeat(num_cells)
        # this tensor is 2 x (num_cells *num_outcomes) x 2
        # the batch dim corresponds to lower/upper bound
        cell_bounds_idxr = torch.stack(
            [self.hypercells.view(2, -1), outcome_idxr.unsqueeze(0).expand(2, -1)],
            dim=-1,
        )
        cell_bounds_values = aug_pareto_Y[
            cell_bounds_idxr.chunk(self.num_outcomes, dim=-1)
        ].view(2, -1, self.num_outcomes)
        return cell_bounds_values

    def compute_hypervolume(self, ref_point: Tensor) -> float:
        r"""Compute the hypervolume for the given reference point.

        Note: This assumes minimization.

        This method computes the hypervolume of the non-dominated space
        and computes the difference between the hypervolume between the
        ideal point and hypervolume of the non-dominated space.

        Note there are much more efficient alternatives for computing
        hypervolume when m > 2 (which do not require partitioning the
        non-dominated space). Given such a partitioning, this method
        is quite fast.

        Args:
            ref_point: A `m`-dim tensor containing the reference point.

        Returns:
            The dominated hypervolume.
        """
        # internally we minimize
        ref_point = -ref_point
        if (self._pareto_Y >= ref_point).any():
            raise ValueError(
                "The reference point must be greater than all pareto_Y values."
            )
        ideal_point = self._pareto_Y.min(dim=0, keepdim=True).values
        ref_point = ref_point.unsqueeze(0)
        aug_pareto_Y = torch.cat([ideal_point, self._pareto_Y, ref_point], dim=0)
        cell_bounds_values = self._get_hypercell_bounds(aug_pareto_Y=aug_pareto_Y)
        total_volume = (ref_point - ideal_point).prod()
        non_dom_volume = (
            (cell_bounds_values[1] - cell_bounds_values[0]).prod(dim=-1).sum()
        )
        return (total_volume - non_dom_volume).item()
