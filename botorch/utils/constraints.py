#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Helpers for handling input or outcome constraints.
"""

from __future__ import annotations

from functools import partial
from typing import Callable, Optional

import torch
from torch import Tensor


def get_outcome_constraint_transforms(
    outcome_constraints: Optional[tuple[Tensor, Tensor]]
) -> Optional[list[Callable[[Tensor], Tensor]]]:
    r"""Create outcome constraint callables from outcome constraint tensors.

    Args:
        outcome_constraints: A tuple of `(A, b)`. For `k` outcome constraints
            and `m` outputs at `f(x)``, `A` is `k x m` and `b` is `k x 1` such
            that `A f(x) <= b`.

    Returns:
        A list of callables, each mapping a Tensor of size `b x q x m` to a
        tensor of size `b x q`, where `m` is the number of outputs of the model.
        Negative values imply feasibility. The callables support broadcasting
        (e.g. for calling on a tensor of shape `mc_samples x b x q x m`).

    Example:
        >>> # constrain `f(x)[0] <= 0`
        >>> A = torch.tensor([[1., 0.]])
        >>> b = torch.tensor([[0.]])
        >>> outcome_constraints = get_outcome_constraint_transforms((A, b))
    """
    if outcome_constraints is None:
        return None
    A, b = outcome_constraints

    def _oc(a: Tensor, rhs: Tensor, Y: Tensor) -> Tensor:
        r"""Evaluate constraints.

        Note: einsum multiples Y by a and sums over the `m`-dimension. Einsum
            is ~2x faster than using `(Y * a.view(1, 1, -1)).sum(dim-1)`.

        Args:
            a: `m`-dim tensor of weights for the outcomes
            rhs: Singleton tensor containing the outcome constraint value
            Y: `... x b x q x m` tensor of function values

        Returns:
            A `... x b x q`-dim tensor where negative values imply feasibility
        """
        lhs = torch.einsum("...m, m", [Y, a])
        return lhs - rhs

    return [partial(_oc, a, rhs) for a, rhs in zip(A, b)]


def get_monotonicity_constraints(
    d: int,
    descending: bool = False,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> tuple[Tensor, Tensor]:
    """Returns a system of linear inequalities `(A, b)` that generically encodes order
    constraints on the elements of a `d`-dimsensional space, i.e. `A @ x < b` implies
    `x[i] < x[i + 1]` for a `d`-dimensional vector `x`.

    Idea: Could encode `A` as sparse matrix, if it is supported well.

    Args:
        d: Dimensionality of the constraint space, i.e. number of monotonic parameters.
        descending: If True, forces the elements of a vector to be monotonically de-
            creasing and be monotonically increasing otherwise.
        dtype: The dtype of the returned Tensors.
        device: The device of the returned Tensors.

    Returns:
        A tuple of Tensors `(A, b)` representing the monotonicity constraint as a system
        of linear inequalities `A @ x < b`. `A` is `(d - 1) x d`-dimensional and `b` is
        `(d - 1) x 1`-dimensional.
    """
    A = torch.zeros(d - 1, d, dtype=dtype, device=device)
    idx = torch.arange(d - 1)
    A[idx, idx] = 1
    A[idx, idx + 1] = -1
    b = torch.zeros(d - 1, 1, dtype=dtype, device=device)
    if descending:
        A = -A
    return A, b
