#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Box decomposition algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
from botorch.exceptions.errors import BotorchError, BotorchTensorDimensionError
from botorch.utils.multi_objective.box_decompositions.utils import (
    _expand_ref_point,
    _pad_batch_pareto_frontier,
)
from torch import Tensor
from torch.nn import Module


class BoxDecomposition(Module, ABC):
    r"""An abstract class for box decompositions.

    Note: Internally, we store the negative reference point (minimization).
    """

    def __init__(
        self, ref_point: Tensor, sort: bool, Y: Optional[Tensor] = None
    ) -> None:
        """Initialize BoxDecomposition.

        Args:
            ref_point: A `m`-dim tensor containing the reference point.
            sort: A boolean indicating whether to sort the Pareto frontier.
            Y: A `(batch_shape) x n x m`-dim tensor of outcomes.
        """
        super().__init__()
        self.register_buffer("_neg_ref_point", -ref_point)
        self.register_buffer("sort", torch.tensor(sort, dtype=torch.bool))
        self.num_outcomes = ref_point.shape[-1]
        if Y is not None:
            self.update(Y=Y)

    @property
    def pareto_Y(self) -> Tensor:
        r"""This returns the non-dominated set.

        Returns:
            A `n_pareto x m`-dim tensor of outcomes.
        """
        try:
            return -self._neg_pareto_Y
        except AttributeError:
            raise BotorchError("pareto_Y has not been initialized")

    @property
    def ref_point(self) -> Tensor:
        r"""Get the reference point.

        Returns:
            A `m`-dim tensor of outcomes.
        """
        return -self._neg_ref_point

    @property
    def Y(self) -> Tensor:
        r"""Get the raw outcomes.

        Returns:
            A `n x m`-dim tensor of outcomes.
        """
        return -self._neg_Y

    def _update_pareto_Y(self) -> bool:
        r"""Update the non-dominated front.

        Returns:
            A boolean indicating whether the Pareto frontier has changed.
        """
        # is_non_dominated assumes maximization
        if self._neg_Y.shape[-2] == 0:
            pareto_Y = self._neg_Y
        else:
            # assumes maximization
            pareto_Y = -_pad_batch_pareto_frontier(
                Y=self.Y,
                ref_point=_expand_ref_point(
                    ref_point=self.ref_point, batch_shape=self.batch_shape
                ),
            )
            if self.sort:
                # sort by first objective
                if len(self.batch_shape) > 0:
                    pareto_Y = pareto_Y.gather(
                        index=torch.argsort(pareto_Y[..., :1], dim=-2).expand(
                            pareto_Y.shape
                        ),
                        dim=-2,
                    )
                else:
                    pareto_Y = pareto_Y[torch.argsort(pareto_Y[:, 0])]

        if not hasattr(self, "_neg_pareto_Y") or not torch.equal(
            pareto_Y, self._neg_pareto_Y
        ):
            self.register_buffer("_neg_pareto_Y", pareto_Y)
            return True
        return False

    def partition_space(self) -> None:
        r"""Compute box decomposition."""
        try:
            self.partition_space_2d()
        except BotorchTensorDimensionError:
            self._partition_space()

    @abstractmethod
    def partition_space_2d(self) -> None:
        r"""Compute box decomposition for 2 objectives."""
        pass  # pragma: no cover

    @abstractmethod
    def get_hypercell_bounds(self) -> Tensor:
        r"""Get the bounds of each hypercell in the decomposition.

        Returns:
            A `2 x num_cells x num_outcomes`-dim tensor containing the
                lower and upper vertices bounding each hypercell.
        """
        pass  # pragma: no cover

    def update(self, Y: Tensor) -> None:
        r"""Update non-dominated front and decomposition.

        Args:
            Y: A `(batch_shape) x n x m`-dim tensor of outcomes.
        """
        self.batch_shape = Y.shape[:-2]
        if len(self.batch_shape) > 1:
            raise NotImplementedError(
                f"{type(self).__name__} only supports a single "
                f"batch dimension, but got {len(self.batch_shape)} "
                "batch dimensions."
            )
        elif len(self.batch_shape) > 0 and self.num_outcomes > 2:
            raise NotImplementedError(
                f"{type(self).__name__} only supports a batched box "
                f"decompositions in the 2-objective setting."
            )
        # multiply by -1, since internally we minimize.
        self._neg_Y = -Y
        is_new_pareto = self._update_pareto_Y()
        # Update decomposition if the Pareto front changed
        if is_new_pareto:
            self.partition_space()
