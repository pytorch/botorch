#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Hypervolume Utilities.

References

.. [Fonseca2006]
    C. M. Fonseca, L. Paquete, and M. Lopez-Ibanez. An improved dimension-sweep
    algorithm for the hypervolume indicator. In IEEE Congress on Evolutionary
    Computation, pages 1157-1163, Vancouver, Canada, July 2006.

.. [Ishibuchi2011]
    H. Ishibuchi, N. Akedo, and Y. Nojima. A many-objective test problem
    for visually examining diversity maintenance behavior in a decision
    space. Proc. 13th Annual Conf. Genetic Evol. Comput., 2011.

"""

from __future__ import annotations

from typing import List, Optional

import torch
from botorch.exceptions.errors import BotorchError, BotorchTensorDimensionError
from torch import Tensor

MIN_Y_RANGE = 1e-7


def infer_reference_point(
    pareto_Y: Tensor,
    max_ref_point: Optional[Tensor] = None,
    scale: float = 0.1,
    scale_max_ref_point: bool = False,
) -> Tensor:
    r"""Get reference point for hypervolume computations.

    This sets the reference point to be `ref_point = nadir - 0.1 * range`
    when there is no pareto_Y that is better than the reference point.

    [Ishibuchi2011]_ find 0.1 to be a robust multiplier for scaling the
    nadir point.

    Note: this assumes maximization of all objectives.

    Args:
        pareto_Y: A `n x m`-dim tensor of Pareto-optimal points.
        max_ref_point: A `m` dim tensor indicating the maximum reference point.
        scale: A multiplier used to scale back the reference point based on the
            range of each objective.
        scale_max_ref_point: A boolean indicating whether to apply scaling to
            the max_ref_point based on the range of each objective.

    Returns:
        A `m`-dim tensor containing the reference point.
    """

    if pareto_Y.shape[0] == 0:
        if max_ref_point is None:
            raise BotorchError("Empty pareto set and no max ref point provided")
        if scale_max_ref_point:
            return max_ref_point - scale * max_ref_point.abs()
        return max_ref_point
    if max_ref_point is not None:
        better_than_ref = (pareto_Y > max_ref_point).all(dim=-1)
    else:
        better_than_ref = torch.full(
            pareto_Y.shape[:1], 1, dtype=bool, device=pareto_Y.device
        )
    if max_ref_point is not None and better_than_ref.any():
        Y_range = pareto_Y[better_than_ref].max(dim=0).values - max_ref_point
        if scale_max_ref_point:
            return max_ref_point - scale * Y_range
        return max_ref_point
    elif pareto_Y.shape[0] == 1:
        # no points better than max_ref_point and only a single observation
        # subtract MIN_Y_RANGE to handle the case that pareto_Y is a singleton
        # with objective value of 0.
        return (pareto_Y - scale * pareto_Y.abs().clamp_min(MIN_Y_RANGE)).view(-1)
    # no points better than max_ref_point and multiple observations
    # make sure that each dimension of the nadir point is no greater than
    # the max_ref_point
    nadir = pareto_Y.min(dim=0).values
    if max_ref_point is not None:
        nadir = torch.min(nadir, max_ref_point)
    ideal = pareto_Y.max(dim=0).values
    # handle case where all values for one objective are the same
    Y_range = (ideal - nadir).clamp_min(MIN_Y_RANGE)
    return nadir - scale * Y_range


class Hypervolume:
    r"""Hypervolume computation dimension sweep algorithm from [Fonseca2006]_.

    Adapted from Simon Wessing's implementation of the algorithm
    (Variant 3, Version 1.2) in [Fonseca2006]_ in PyMOO:
    https://github.com/msu-coinlab/pymoo/blob/master/pymoo/vendor/hv.py

    Maximization is assumed.

    TODO: write this in C++ for faster looping.
    """

    def __init__(self, ref_point: Tensor) -> None:
        r"""Initialize hypervolume object.

        Args:
            ref_point: `m`-dim Tensor containing the reference point.

        """
        self.ref_point = ref_point

    @property
    def ref_point(self) -> Tensor:
        r"""Get reference point (for maximization).

        Returns:
            A `m`-dim tensor containing the reference point.
        """
        return -self._ref_point

    @ref_point.setter
    def ref_point(self, ref_point: Tensor) -> None:
        r"""Set the reference point for maximization

        Args:
            ref_point:  A `m`-dim tensor containing the reference point.
        """
        self._ref_point = -ref_point

    def compute(self, pareto_Y: Tensor) -> float:
        r"""Compute hypervolume.

        Args:
            pareto_Y: A `n x m`-dim tensor of pareto optimal outcomes

        Returns:
            The hypervolume.
        """
        if pareto_Y.shape[-1] != self._ref_point.shape[0]:
            raise BotorchTensorDimensionError(
                "pareto_Y must have the same number of objectives as ref_point. "
                f"Got {pareto_Y.shape[-1]}, expected {self._ref_point.shape[0]}."
            )
        elif pareto_Y.ndim != 2:
            raise BotorchTensorDimensionError(
                f"pareto_Y must have exactly two dimensions, got {pareto_Y.ndim}."
            )
        # This assumes maximization, but internally flips the sign of the pareto front
        # and the reference point and computes hypervolume for the minimization problem.
        pareto_Y = -pareto_Y
        better_than_ref = (pareto_Y <= self._ref_point).all(dim=-1)
        pareto_Y = pareto_Y[better_than_ref]
        # shift the pareto front so that reference point is all zeros
        pareto_Y = pareto_Y - self._ref_point
        self._initialize_multilist(pareto_Y)
        bounds = torch.full_like(self._ref_point, float("-inf"))
        return self._hv_recursive(
            i=self._ref_point.shape[0] - 1, n_pareto=pareto_Y.shape[0], bounds=bounds
        )

    def _hv_recursive(self, i: int, n_pareto: int, bounds: Tensor) -> float:
        r"""Recursive method for hypervolume calculation.

        This assumes minimization (internally).

        In contrast to the paper, this code assumes that the reference point
        is the origin. This enables pruning a few operations.

        Args:
            i: objective index
            n_pareto: number of pareto points
            bounds: objective bounds

        Returns:
            The hypervolume.
        """
        hvol = torch.tensor(0.0, dtype=bounds.dtype, device=bounds.device)
        sentinel = self.list.sentinel
        if n_pareto == 0:
            # base case: one dimension
            return hvol.item()
        elif i == 0:
            # base case: one dimension
            return -sentinel.next[0].data[0].item()
        elif i == 1:
            # two dimensions, end recursion
            q = sentinel.next[1]
            h = q.data[0]
            p = q.next[1]
            while p is not sentinel:
                hvol += h * (q.data[1] - p.data[1])
                if p.data[0] < h:
                    h = p.data[0]
                q = p
                p = q.next[1]
            hvol += h * q.data[1]
            return hvol.item()
        else:
            p = sentinel
            q = p.prev[i]
            while q.data is not None:
                if q.ignore < i:
                    q.ignore = 0
                q = q.prev[i]
            q = p.prev[i]
            while n_pareto > 1 and (
                q.data[i] > bounds[i] or q.prev[i].data[i] >= bounds[i]
            ):
                p = q
                self.list.remove(p, i, bounds)
                q = p.prev[i]
                n_pareto -= 1
            q_prev = q.prev[i]
            if n_pareto > 1:
                hvol = q_prev.volume[i] + q_prev.area[i] * (q.data[i] - q_prev.data[i])
            else:
                q.area[0] = 1
                q.area[1 : i + 1] = q.area[:i] * -(q.data[:i])
            q.volume[i] = hvol
            if q.ignore >= i:
                q.area[i] = q_prev.area[i]
            else:
                q.area[i] = self._hv_recursive(i - 1, n_pareto, bounds)
                if q.area[i] <= q_prev.area[i]:
                    q.ignore = i
            while p is not sentinel:
                p_data = p.data[i]
                hvol += q.area[i] * (p_data - q.data[i])
                bounds[i] = p_data
                self.list.reinsert(p, i, bounds)
                n_pareto += 1
                q = p
                p = p.next[i]
                q.volume[i] = hvol
                if q.ignore >= i:
                    q.area[i] = q.prev[i].area[i]
                else:
                    q.area[i] = self._hv_recursive(i - 1, n_pareto, bounds)
                    if q.area[i] <= q.prev[i].area[i]:
                        q.ignore = i
            hvol -= q.area[i] * q.data[i]
            return hvol.item()

    def _initialize_multilist(self, pareto_Y: Tensor) -> None:
        r"""Sets up the multilist data structure needed for calculation.

        Note: this assumes minimization.

        Args:
            pareto_Y: A `n x m`-dim tensor of pareto optimal objectives.

        """
        m = pareto_Y.shape[-1]
        nodes = [
            Node(m=m, dtype=pareto_Y.dtype, device=pareto_Y.device, data=point)
            for point in pareto_Y
        ]
        self.list = MultiList(m=m, dtype=pareto_Y.dtype, device=pareto_Y.device)
        for i in range(m):
            sort_by_dimension(nodes, i)
            self.list.extend(nodes, i)


def sort_by_dimension(nodes: List[Node], i: int) -> None:
    r"""Sorts the list of nodes in-place by the specified objective.

    Args:
        nodes: A list of Nodes
        i: The index of the objective to sort by

    """
    # build a list of tuples of (point[i], node)
    decorated = [(node.data[i], index, node) for index, node in enumerate(nodes)]
    # sort by this value
    decorated.sort()
    # write back to original list
    nodes[:] = [node for (_, _, node) in decorated]


class Node:
    r"""Node in the MultiList data structure."""

    def __init__(
        self,
        m: int,
        dtype: torch.dtype,
        device: torch.device,
        data: Optional[Tensor] = None,
    ) -> None:
        r"""Initialize MultiList.

        Args:
            m: The number of objectives
            dtype: The dtype
            device: The device
            data: The tensor data to be stored in this Node.
        """
        self.data = data
        self.next = [None] * m
        self.prev = [None] * m
        self.ignore = 0
        self.area = torch.zeros(m, dtype=dtype, device=device)
        self.volume = torch.zeros_like(self.area)


class MultiList:
    r"""A special data structure used in hypervolume computation.

    It consists of several doubly linked lists that share common nodes.
    Every node has multiple predecessors and successors, one in every list.
    """

    def __init__(self, m: int, dtype: torch.dtype, device: torch.device) -> None:
        r"""Initialize `m` doubly linked lists.

        Args:
            m: number of doubly linked lists
            dtype: the dtype
            device: the device

        """
        self.m = m
        self.sentinel = Node(m=m, dtype=dtype, device=device)
        self.sentinel.next = [self.sentinel] * m
        self.sentinel.prev = [self.sentinel] * m

    def append(self, node: Node, index: int) -> None:
        r"""Appends a node to the end of the list at the given index.

        Args:
            node: the new node
            index: the index where the node should be appended.
        """
        last = self.sentinel.prev[index]
        node.next[index] = self.sentinel
        node.prev[index] = last
        # set the last element as the new one
        self.sentinel.prev[index] = node
        last.next[index] = node

    def extend(self, nodes: List[Node], index: int) -> None:
        r"""Extends the list at the given index with the nodes.

        Args:
            nodes: list of nodes to append at the given index.
            index: the index where the nodes should be appended.

        """
        for node in nodes:
            self.append(node=node, index=index)

    def remove(self, node: Node, index: int, bounds: Tensor) -> Node:
        r"""Removes and returns 'node' from all lists in [0, 'index'].

        Args:
            node: The node to remove
            index: The upper bound on the range of indices
            bounds: A `2 x m`-dim tensor bounds on the objectives
        """
        for i in range(index):
            predecessor = node.prev[i]
            successor = node.next[i]
            predecessor.next[i] = successor
            successor.prev[i] = predecessor
        bounds.data = torch.min(bounds, node.data)
        return node

    def reinsert(self, node: Node, index: int, bounds: Tensor) -> None:
        r"""Re-inserts the node at its original position.

        Re-inserts the node at its original position in all lists in [0, 'index']
        before it was removed. This method assumes that the next and previous
        nodes of the node that is reinserted are in the list.

        Args:
            node: The node
            index: The upper bound on the range of indices
            bounds: A `2 x m`-dim tensor bounds on the objectives

        """
        for i in range(index):
            node.prev[i].next[i] = node
            node.next[i].prev[i] = node
        bounds.data = torch.min(bounds, node.data)
