#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Representations for different kinds of datasets."""

from __future__ import annotations

import warnings
from typing import Any, Iterable, List, Optional, TypeVar, Union

import torch
from botorch.utils.containers import BotorchContainer, SliceContainer
from torch import long, ones, Tensor

T = TypeVar("T")
MaybeIterable = Union[T, Iterable[T]]


class SupervisedDataset:
    r"""Base class for datasets consisting of labelled pairs `(X, Y)`
    and an optional `Yvar` that stipulates observations variances so
    that `Y[i] ~ N(f(X[i]), Yvar[i])`.

    Example:

    .. code-block:: python

        X = torch.rand(16, 2)
        Y = torch.rand(16, 1)
        feature_names = ["learning_rate", "embedding_dim"]
        outcome_names = ["neg training loss"]
        A = SupervisedDataset(
            X=X,
            Y=Y,
            feature_names=feature_names,
            outcome_names=outcome_names,
        )
        B = SupervisedDataset(
            X=DenseContainer(X, event_shape=X.shape[-1:]),
            Y=DenseContainer(Y, event_shape=Y.shape[-1:]),
            feature_names=feature_names,
            outcome_names=outcome_names,
        )
        assert A == B
    """

    def __init__(
        self,
        X: Union[BotorchContainer, Tensor],
        Y: Union[BotorchContainer, Tensor],
        *,
        feature_names: List[str],
        outcome_names: List[str],
        Yvar: Union[BotorchContainer, Tensor, None] = None,
        validate_init: bool = True,
    ) -> None:
        r"""Constructs a `SupervisedDataset`.

        Args:
            X: A `Tensor` or `BotorchContainer` representing the input features.
            Y: A `Tensor` or `BotorchContainer` representing the outcomes.
            feature_names: A list of names of the features in `X`.
            outcome_names: A list of names of the outcomes in `Y`.
            Yvar: An optional `Tensor` or `BotorchContainer` representing
                the observation noise.
            validate_init: If `True`, validates the input shapes.
        """
        self._X = X
        self._Y = Y
        self._Yvar = Yvar
        self.feature_names = feature_names
        self.outcome_names = outcome_names
        if validate_init:
            self._validate()

    @property
    def X(self) -> Tensor:
        if isinstance(self._X, Tensor):
            return self._X
        return self._X()

    @property
    def Y(self) -> Tensor:
        if isinstance(self._Y, Tensor):
            return self._Y
        return self._Y()

    @property
    def Yvar(self) -> Optional[Tensor]:
        if self._Yvar is None or isinstance(self._Yvar, Tensor):
            return self._Yvar
        return self._Yvar()

    def _validate(
        self,
        validate_feature_names: bool = True,
        validate_outcome_names: bool = True,
    ) -> None:
        r"""Checks that the shapes of the inputs are compatible with each other.

        Args:
            validate_feature_names: By default, we validate that the length of
                `feature_names` matches the # of columns of `self.X`. If a
                particular dataset, e.g., `RankingDataset`, is known to violate
                this assumption, this can be set to `False`.
            validate_outcome_names: By default, we validate that the length of
                `outcomes_names` matches the # of columns of `self.Y`. If a
                particular dataset, e.g., `RankingDataset`, is known to violate
                this assumption, this can be set to `False`.
        """
        shape_X = self.X.shape
        if isinstance(self._X, BotorchContainer):
            shape_X = shape_X[: len(shape_X) - len(self._X.event_shape)]
        else:
            shape_X = shape_X[:-1]
        shape_Y = self.Y.shape
        if isinstance(self._Y, BotorchContainer):
            shape_Y = shape_Y[: len(shape_Y) - len(self._Y.event_shape)]
        else:
            shape_Y = shape_Y[:-1]
        if shape_X != shape_Y:
            raise ValueError("Batch dimensions of `X` and `Y` are incompatible.")
        if self.Yvar is not None and self.Yvar.shape != self.Y.shape:
            raise ValueError("Shapes of `Y` and `Yvar` are incompatible.")
        if validate_feature_names and len(self.feature_names) != self.X.shape[-1]:
            raise ValueError(
                "`X` must have the same number of columns as the number of "
                "features in `feature_names`."
            )
        if validate_outcome_names and len(self.outcome_names) != self.Y.shape[-1]:
            raise ValueError(
                "`Y` must have the same number of columns as the number of "
                "outcomes in `outcome_names`."
            )

    def __eq__(self, other: Any) -> bool:
        return (
            type(other) is type(self)
            and torch.equal(self.X, other.X)
            and torch.equal(self.Y, other.Y)
            and (
                other.Yvar is None
                if self.Yvar is None
                else torch.equal(self.Yvar, other.Yvar)
            )
            and self.feature_names == other.feature_names
            and self.outcome_names == other.outcome_names
        )


class FixedNoiseDataset(SupervisedDataset):
    r"""A SupervisedDataset with an additional field `Yvar` that stipulates
    observations variances so that `Y[i] ~ N(f(X[i]), Yvar[i])`.

    NOTE: This is deprecated. Use `SupervisedDataset` instead.
    """

    def __init__(
        self,
        X: Union[BotorchContainer, Tensor],
        Y: Union[BotorchContainer, Tensor],
        Yvar: Union[BotorchContainer, Tensor],
        feature_names: List[str],
        outcome_names: List[str],
        validate_init: bool = True,
    ) -> None:
        r"""Initialize a `FixedNoiseDataset` -- deprecated!"""
        warnings.warn(
            "`FixedNoiseDataset` is deprecated. Use `SupervisedDataset` instead.",
            DeprecationWarning,
        )
        super().__init__(
            X=X,
            Y=Y,
            feature_names=feature_names,
            outcome_names=outcome_names,
            Yvar=Yvar,
            validate_init=validate_init,
        )


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
        feature_names = ["item_0", "item_1"]
        outcome_names = ["ranking outcome"]
        dataset = RankingDataset(
            X=X,
            Y=Y,
            feature_names=feature_names,
            outcome_names=outcome_names,
        )
    """

    def __init__(
        self,
        X: SliceContainer,
        Y: Union[BotorchContainer, Tensor],
        feature_names: List[str],
        outcome_names: List[str],
        validate_init: bool = True,
    ) -> None:
        r"""Construct a `RankingDataset`.

        Args:
            X: A `SliceContainer` representing the input features being ranked.
            Y: A `Tensor` or `BotorchContainer` representing the rankings.
            feature_names: A list of names of the features in X.
            outcome_names: A list of names of the outcomes in Y.
            validate_init: If `True`, validates the input shapes.
        """
        super().__init__(
            X=X,
            Y=Y,
            feature_names=feature_names,
            outcome_names=outcome_names,
            Yvar=None,
            validate_init=validate_init,
        )

    def _validate(self) -> None:
        super()._validate(validate_feature_names=False, validate_outcome_names=False)
        if len(self.feature_names) != self._X.values.shape[-1]:
            raise ValueError(
                "The `values` field of `X` must have the same number of columns as "
                "the number of features in `feature_names`."
            )

        Y = self.Y
        arity = self._X.indices.shape[-1]
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
