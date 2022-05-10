#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Representations for different kinds of datasets."""

from __future__ import annotations

from dataclasses import dataclass, fields, MISSING
from itertools import chain, count, repeat
from typing import Any, Dict, Hashable, Iterable, Optional, TypeVar, Union

from botorch.utils.containers import BotorchContainer, DenseContainer, SliceContainer
from torch import long, ones, Tensor
from typing_extensions import get_type_hints

T = TypeVar("T")
ContainerLike = Union[BotorchContainer, Tensor]
MaybeIterable = Union[T, Iterable[T]]


@dataclass
class BotorchDataset:
    # TODO: Once v3.10 becomes standard, expose `validate_init` as a kw_only InitVar
    def __post_init__(self, validate_init: bool = True) -> None:
        if validate_init:
            self._validate()

    def _validate(self) -> None:
        pass


class SupervisedDatasetMeta(type):
    def __call__(cls, *args: Any, **kwargs: Any):
        r"""Converts Tensor-valued fields to DenseContainer under the assumption
        that said fields house collections of feature vectors."""
        hints = get_type_hints(cls)
        f_iter = filter(
            lambda f: f.init and issubclass(hints[f.name], BotorchContainer),
            fields(cls),
        )
        f_dict = {}
        for obj, f in chain(
            zip(args, f_iter), ((kwargs.pop(f.name, MISSING), f) for f in f_iter)
        ):
            if obj is MISSING:
                if f.default is not MISSING:

                    obj = f.default
                elif f.default_factory is not MISSING:
                    obj = f.default_factory()
                else:
                    raise RuntimeError(f"Missing required field `{f.name}`.")

            if isinstance(obj, Tensor):
                obj = DenseContainer(obj, event_shape=obj.shape[-1:])
            elif not isinstance(obj, BotorchContainer):
                raise TypeError(
                    f"Expected <BotorchContainer | Tensor> for field `{f.name}` "
                    f"but was {type(obj)}."
                )
            f_dict[f.name] = obj

        return super().__call__(**f_dict, **kwargs)


@dataclass
class SupervisedDataset(BotorchDataset, metaclass=SupervisedDatasetMeta):
    r"""Base class for datasets consisting of labelled pairs `(x, y)`.

    This class object's `__call__` method converts Tensors `src` to
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

    def _validate(self) -> None:
        shape_X = self.X.shape
        shape_X = shape_X[: len(shape_X) - len(self.X.event_shape)]
        shape_Y = self.Y.shape
        shape_Y = shape_Y[: len(shape_Y) - len(self.Y.event_shape)]
        if shape_X != shape_Y:
            raise ValueError("Batch dimensions of `X` and `Y` are incompatible.")

    @classmethod
    def dict_from_iter(
        cls,
        X: MaybeIterable[ContainerLike],
        Y: MaybeIterable[ContainerLike],
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
        return {key: cls(x, y) for key, x, y in zip(keys or count(), X, Y)}


@dataclass
class FixedNoiseDataset(SupervisedDataset):
    r"""A SupervisedDataset with an additional field `Yvar` that stipulates
    observations variances so that `Y[i] ~ N(f(X[i]), Yvar[i])`."""

    X: BotorchContainer
    Y: BotorchContainer
    Yvar: BotorchContainer

    @classmethod
    def dict_from_iter(
        cls,
        X: MaybeIterable[ContainerLike],
        Y: MaybeIterable[ContainerLike],
        Yvar: Optional[MaybeIterable[ContainerLike]] = None,
        *,
        keys: Optional[Iterable[Hashable]] = None,
    ) -> Dict[Hashable, SupervisedDataset]:
        r"""Returns a dictionary of `FixedNoiseDataset` from iterables."""
        single_X = isinstance(X, (Tensor, BotorchContainer))
        single_Y = isinstance(Y, (Tensor, BotorchContainer))
        if single_X:
            X = (X,) if single_Y else repeat(X)
        if single_Y:
            Y = (Y,) if single_X else repeat(Y)

        Yvar = repeat(Yvar) if isinstance(Yvar, (Tensor, BotorchContainer)) else Yvar
        return {key: cls(x, y, c) for key, x, y, c in zip(keys or count(), X, Y, Yvar)}


@dataclass
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
