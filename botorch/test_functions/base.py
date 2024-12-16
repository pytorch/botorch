#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Base class for test functions for optimization benchmarks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable, Iterator, Protocol

import torch
from botorch.exceptions.errors import InputDataError
from pyre_extensions import none_throws
from torch import Tensor
from torch.nn import Module


class BaseTestProblem(Module, ABC):
    r"""Base class for test functions."""

    dim: int
    _bounds: list[tuple[float, float]]
    _check_grad_at_opt: bool = True

    def __init__(
        self,
        noise_std: None | float | list[float] = None,
        negate: bool = False,
        dtype: torch.dtype = torch.double,
    ) -> None:
        r"""Base constructor for test functions.

        Args:
            noise_std: Standard deviation of the observation noise. If a list is
                provided, specifies separate noise standard deviations for each
                objective in a multiobjective problem.
            negate: If True, negate the function.
            dtype: The dtype that is used for the bounds of the function.
        """
        super().__init__()
        self.noise_std = noise_std
        self.negate = negate
        if len(self._bounds) != self.dim:
            raise InputDataError(
                "Expected the bounds to match the dimensionality of the domain. "
                f"Got {self.dim=} and {len(self._bounds)=}."
            )
        self.register_buffer(
            "bounds",
            torch.tensor(self._bounds, dtype=dtype).transpose(-1, -2),
        )

    def forward(self, X: Tensor, noise: bool = True) -> Tensor:
        r"""Evaluate the function on a set of points.

        Args:
            X: A `(batch_shape) x d`-dim tensor of point(s) at which to evaluate
                the function.
            noise: If `True`, add observation noise as specified by `noise_std`.

        Returns:
            A `batch_shape`-dim tensor ouf function evaluations.
        """
        f = self.evaluate_true(X=X)
        if noise and self.noise_std is not None:
            _noise = torch.tensor(self.noise_std, device=X.device, dtype=X.dtype)
            f += _noise * torch.randn_like(f)
        if self.negate:
            f = -f
        return f

    @abstractmethod
    def evaluate_true(self, X: Tensor) -> Tensor:
        r"""
        Evaluate the function (w/o observation noise) on a set of points.

        Args:
            X: A `(batch_shape) x d`-dim tensor of point(s) at which to
                evaluate.

        Returns:
            A `batch_shape`-dim tensor.
        """
        pass  # pragma: no cover


class ConstrainedBaseTestProblem(BaseTestProblem, ABC):
    r"""Base class for test functions with constraints.

    In addition to one or more objectives, a problem may have a number of outcome
    constraints of the form `c_i(x) >= 0` for `i=1, ..., n_c`.

    This base class provides common functionality for such problems.
    """

    num_constraints: int
    _check_grad_at_opt: bool = False
    constraint_noise_std: None | float | list[float] = None

    def evaluate_slack(self, X: Tensor, noise: bool = True) -> Tensor:
        r"""Evaluate the constraint slack on a set of points.

        Constraints `i` is assumed to be feasible at `x` if the associated slack
        `c_i(x)` is positive. Zero slack means that the constraint is active. Negative
        slack means that the constraint is violated.

        Args:
            X: A `batch_shape x d`-dim tensor of point(s) at which to evaluate the
                constraint slacks: `c_1(X), ...., c_{n_c}(X)`.
            noise: If `True`, add observation noise to the slack as specified by
                `noise_std`.

        Returns:
            A `batch_shape x n_c`-dim tensor of constraint slack (where positive slack
                corresponds to the constraint being feasible).
        """
        cons = self.evaluate_slack_true(X=X)
        if noise and self.constraint_noise_std is not None:
            _constraint_noise = torch.tensor(
                self.constraint_noise_std, device=X.device, dtype=X.dtype
            )
            cons += _constraint_noise * torch.randn_like(cons)
        return cons

    def is_feasible(self, X: Tensor, noise: bool = True) -> Tensor:
        r"""Evaluate whether the constraints are feasible on a set of points.

        Args:
            X: A `batch_shape x d`-dim tensor of point(s) at which to evaluate the
                constraints.
            noise: If `True`, add observation noise as specified by `noise_std`.

        Returns:
            A `batch_shape`-dim boolean tensor that is `True` iff all constraint
                slacks (potentially including observation noise) are positive.
        """
        return (self.evaluate_slack(X=X, noise=noise) >= 0.0).all(dim=-1)

    @abstractmethod
    def evaluate_slack_true(self, X: Tensor) -> Tensor:
        r"""Evaluate the constraint slack (w/o observation noise) on a set of points.

        Args:
            X: A `batch_shape x d`-dim tensor of point(s) at which to evaluate the
                constraint slacks: `c_1(X), ...., c_{n_c}(X)`.

        Returns:
            A `batch_shape x n_c`-dim tensor of constraint slack (where positive slack
                corresponds to the constraint being feasible).
        """
        pass  # pragma: no cover


class MultiObjectiveTestProblem(BaseTestProblem, ABC):
    r"""Base class for multi-objective test functions.

    TODO: add a pareto distance function that returns the distance
    between a provided point and the closest point on the true pareto front.
    """

    num_objectives: int
    _ref_point: list[float]
    _max_hv: float | None = None

    def __init__(
        self,
        noise_std: None | float | list[float] = None,
        negate: bool = False,
        dtype: torch.dtype = torch.double,
    ) -> None:
        r"""Base constructor for multi-objective test functions.

        Args:
            noise_std: Standard deviation of the observation noise. If a list is
                provided, specifies separate noise standard deviations for each
                objective.
            negate: If True, negate the objectives.
        """
        if isinstance(noise_std, list) and len(noise_std) != len(self._ref_point):
            raise InputDataError(
                f"If specified as a list, length of noise_std ({len(noise_std)}) "
                f"must match the number of objectives ({len(self._ref_point)})"
            )
        super().__init__(noise_std=noise_std, negate=negate, dtype=dtype)
        ref_point = torch.tensor(self._ref_point, dtype=dtype)
        if negate:
            ref_point *= -1
        self.register_buffer("ref_point", ref_point)

    @property
    def max_hv(self) -> float:
        if self._max_hv is not None:
            return self._max_hv
        else:
            raise NotImplementedError(
                f"Problem {self.__class__.__name__} does not specify maximal "
                "hypervolume."
            )

    def gen_pareto_front(self, n: int) -> Tensor:
        r"""Generate `n` pareto optimal points."""
        raise NotImplementedError


class SeedingMixin(ABC):
    _seeds: Iterator[int] | None
    _current_seed: int | None

    @property
    def has_seeds(self) -> bool:
        return self._seeds is not None

    def increment_seed(self) -> int:
        self._current_seed = next(none_throws(self._seeds))
        return none_throws(self._current_seed)

    @property
    def seed(self) -> int | None:
        return self._current_seed


# Outlier problems
class OutlierGenerator(Protocol):
    def __call__(self, problem: BaseTestProblem, X: Tensor, bounds: Tensor) -> Tensor:
        """Call signature for outlier generators for single-objective problems.

        Args:
            problem: The test problem.
            X: The input tensor.
            bounds: The bounds of the test problem.

        Returns:
            A tensor of outliers with shape X.shape[:-1] (1d if unbatched).
        """
        pass  # pragma: no cover


def constant_outlier_generator(
    problem: Any, X: Tensor, bounds: Any, constant: float
) -> Tensor:
    """
    Generates outliers that are all the same constant. To be used in conjunction with
    `partial` to fix the constant value and conform to the `OutlierGenerator` protocol.

    Example:
        >>> generator = partial(constant_outlier_generator, constant=1.0)

    Args:
        problem: Not used.
        X: The `batch_shape x n x d`-dim inputs. Also determines the number, dtype,
            and device of the returned tensor.
        bounds: Not used.
        constant: The constant value of the outliers.

    Returns:
        Tensor of shape `batch_shape x n` (1d if unbatched).
    """
    return torch.full(X.shape[:-1], constant, dtype=X.dtype, device=X.device)


class CorruptedTestProblem(BaseTestProblem, SeedingMixin):
    def __init__(
        self,
        base_test_problem: BaseTestProblem,
        outlier_generator: OutlierGenerator,
        outlier_fraction: float,
        bounds: list[tuple[float, float]] | None = None,
        seeds: Iterable[int] | None = None,
    ) -> None:
        """A problem with outliers.

        NOTE: Both noise_std and negate will be taken from the base test problem.

        Args:
            base_test_problem: The base function to be corrupted.
            outlier_generator: A function that generates outliers. It will be called
                with arguments `f`, `X` and `bounds`, where `f` is the
                `base_test_problem`, `X` is the
                argument passed to the `forward` method, and `bounds`
                are as here, and it returns the values of outliers.
            outlier_fraction: The fraction of outliers.
            bounds: The bounds of the function.
            seeds: The seeds to use for the outlier generator. If seeds are provided,
                the problem will iterate through the list of seeds, changing the seed
                with a call to `next(seeds)` with every `forward` call. If a list is
                provided, it will first be converted to an iterator.
        """
        self.dim: int = base_test_problem.dim
        self._bounds: list[tuple[float, float]] = (
            bounds if bounds is not None else base_test_problem._bounds
        )
        super().__init__(
            noise_std=base_test_problem.noise_std,
            negate=base_test_problem.negate,
        )
        self.base_test_problem = base_test_problem
        self.outlier_generator = outlier_generator
        self.outlier_fraction = outlier_fraction
        self._current_seed: int | None = None
        self._seeds: Iterator[int] | None = None if seeds is None else iter(seeds)

    def evaluate_true(self, X: Tensor) -> Tensor:
        return self.base_test_problem.evaluate_true(X)

    def forward(self, X: Tensor, noise: bool = True) -> Tensor:
        """
        Generate data at X and corrupt it, if noise is True.

        Args:
            X: The `batch_shape x n x d`-dim inputs.
            noise: Whether to corrupt the data.

        Returns:
            A `batch_shape x n`-dim tensor.
        """
        Y = super().forward(X, noise=noise)
        if noise:
            if self.has_seeds:
                self.increment_seed()
                torch.manual_seed(self.seed)
            corrupt = torch.rand(X.shape[:-1]) < self.outlier_fraction
            outliers = self.outlier_generator(
                problem=self.base_test_problem, X=X, bounds=self.bounds
            )
            Y = torch.where(corrupt, outliers, Y)
        return Y
