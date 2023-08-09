#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Representations for different kinds of datasets."""

from __future__ import annotations

import warnings
from itertools import count, repeat
from typing import Any, Dict, Hashable, Iterable, Optional, TypeVar, Union

from botorch.utils.containers import BotorchContainer, DenseContainer, SliceContainer
from torch import long, ones, Tensor

T = TypeVar("T")
ContainerLike = Union[BotorchContainer, Tensor]
MaybeIterable = Union[T, Iterable[T]]


class SupervisedDataset:
    r"""Base class for datasets consisting of labelled pairs `(X, Y)`
    and an optional `Yvar` that stipulates observations variances so
    that `Y[i] ~ N(f(X[i]), Yvar[i])`.

    This class object's `__init__` method converts Tensors `src` to
    DenseContainers under the assumption that `event_shape=src.shape[-1:]`.

    Example:

    .. code-block:: python

        X = torch.rand(16, 2)
        Y = torch.rand(16, 1)
        A = SupervisedDataset(X, Y)
        B = SupervisedDataset(
            DenseContainer(X, event_shape=X.shape[-1:]),
            DenseContainer(Y, event_shape=Y.shape[-1:]),
        )
        assert A == B
    """

    X: BotorchContainer
    Y: BotorchContainer
    Yvar: Optional[BotorchContainer]

    def __init__(
        self,
        X: ContainerLike,
        Y: ContainerLike,
        Yvar: Optional[ContainerLike] = None,
        validate_init: bool = True,
    ) -> None:
        r"""Constructs a `SupervisedDataset`.

        Args:
            X: A `Tensor` or `BotorchContainer` representing the input features.
            Y: A `Tensor` or `BotorchContainer` representing the outcomes.
            Yvar: An optional `Tensor` or `BotorchContainer` representing
                the observation noise.
            validate_init: If `True`, validates the input shapes.
        """
        self.X = _containerize(X)
        self.Y = _containerize(Y)
        self.Yvar = None if Yvar is None else _containerize(Yvar)
        if validate_init:
            self._validate()

    def _validate(self) -> None:
        shape_X = self.X.shape
        shape_X = shape_X[: len(shape_X) - len(self.X.event_shape)]
        shape_Y = self.Y.shape
        shape_Y = shape_Y[: len(shape_Y) - len(self.Y.event_shape)]
        if shape_X != shape_Y:
            raise ValueError("Batch dimensions of `X` and `Y` are incompatible.")
        if self.Yvar is not None and self.Yvar.shape != self.Y.shape:
            raise ValueError("Shapes of `Y` and `Yvar` are incompatible.")

    @classmethod
    def dict_from_iter(
        cls,
        X: MaybeIterable[ContainerLike],
        Y: MaybeIterable[ContainerLike],
        Yvar: Optional[MaybeIterable[ContainerLike]] = None,
        *,
        keys: Optional[Iterable[Hashable]] = None,
    ) -> Dict[Hashable, SupervisedDataset]:
        r"""Returns a dictionary of `SupervisedDataset` from iterables."""
        single_X = isinstance(X, (Tensor, BotorchContainer))
        single_Y = isinstance(Y, (Tensor, BotorchContainer))
        if single_X:
            X = (X,) if single_Y else repeat(X)
        if single_Y:
            Y = (Y,) if single_X else repeat(Y)
        Yvar = repeat(Yvar) if isinstance(Yvar, (Tensor, BotorchContainer)) else Yvar

        # Pass in Yvar only if it is not None.
        iterables = (X, Y) if Yvar is None else (X, Y, Yvar)
        return {
            elements[0]: cls(*elements[1:])
            for elements in zip(keys or count(), *iterables)
        }

    def __eq__(self, other: Any) -> bool:
        return (
            type(other) is type(self)
            and self.X == other.X
            and self.Y == other.Y
            and self.Yvar == other.Yvar
        )


class FixedNoiseDataset(SupervisedDataset):
    r"""A SupervisedDataset with an additional field `Yvar` that stipulates
    observations variances so that `Y[i] ~ N(f(X[i]), Yvar[i])`.

    NOTE: This is deprecated. Use `SupervisedDataset` instead.
    """

    def __init__(
        self,
        X: ContainerLike,
        Y: ContainerLike,
        Yvar: ContainerLike,
        validate_init: bool = True,
    ) -> None:
        r"""Initialize a `FixedNoiseDataset` -- deprecated!"""
        warnings.warn(
            "`FixedNoiseDataset` is deprecated. Use `SupervisedDataset` instead.",
            DeprecationWarning,
        )
        super().__init__(X=X, Y=Y, Yvar=Yvar, validate_init=validate_init)


class RankingDataset(SupervisedDataset):
    r"""A SupervisedDataset whose labelled pairs `(x, y)` consist of m-ary combinations
    `x ∈ Z^{m}` of elements from a ground set `Z = (z_1, ...)` and ranking vectors
    `y {0, ..., m - 1}^{m}` with properties:

        a) Ranks start at zero, i.e. min(y) = 0.
        b) Sorted ranks are contiguous unless one or more ties are present.
        c) `k` ranks are skipped after a `k`-way tie.

    Example:

    .. code-block:: python

        X = SliceContainer(
            values=torch.rand(16, 2),
            indices=torch.stack([torch.randperm(16)[:3] for _ in range(8)]),
            event_shape=torch.Size([3 * 2]),
        )
        Y = DenseContainer(
            torch.stack([torch.randperm(3) for _ in range(8)]),
            event_shape=torch.Size([3])
        )
        dataset = RankingDataset(X, Y)
    """

    X: SliceContainer
    Y: BotorchContainer

    def __init__(
        self, X: SliceContainer, Y: ContainerLike, validate_init: bool = True
    ) -> None:
        r"""Construct a `RankingDataset`.

        Args:
            X: A `SliceContainer` representing the input features being ranked.
            Y: A `Tensor` or `BotorchContainer` representing the rankings.
            validate_init: If `True`, validates the input shapes.
        """
        super().__init__(X=X, Y=Y, Yvar=None, validate_init=validate_init)

    def _validate(self) -> None:
        super()._validate()

        Y = self.Y()
        arity = self.X.indices.shape[-1]
        if Y.min() < 0 or Y.max() >= arity:
            raise ValueError("Invalid ranking(s): out-of-bounds ranks detected.")

        # Ensure that rankings are well-defined
        Y_sort = Y.sort(descending=False, dim=-1).values
        y_incr = ones([], dtype=long)
        y_prev = None
        for i, y in enumerate(Y_sort.unbind(dim=-1)):
            if i == 0:
                if (y != 0).any():
                    raise ValueError("Invalid ranking(s): missing zero-th rank.")
                y_prev = y
                continue

            y_diff = y - y_prev
            y_prev = y

            # Either a tie or next ranking when accounting for previous ties
            if not ((y_diff == 0) | (y_diff == y_incr)).all():
                raise ValueError("Invalid ranking(s): ranks not skipped after ties.")

            # Same as: torch.where(y_diff == 0, y_incr + 1, 1)
            y_incr = y_incr - y_diff + 1


def _containerize(value: ContainerLike) -> BotorchContainer:
    r"""Converts Tensor-valued arguments to DenseContainer under the assumption
    that said arguments house collections of feature vectors.
    """
    if isinstance(value, Tensor):
        return DenseContainer(value, event_shape=value.shape[-1:])
    else:
        return value
