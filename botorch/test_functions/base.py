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
from typing import Optional, Union

import torch
from botorch.exceptions.errors import InputDataError
from torch import Tensor
from torch.nn import Module


class BaseTestProblem(Module, ABC):
    r"""Base class for test functions."""

    dim: int
    _bounds: list[tuple[float, float]]
    _check_grad_at_opt: bool = True

    def __init__(
        self,
        noise_std: Union[None, float, list[float]] = None,
        negate: bool = False,
    ) -> None:
        r"""Base constructor for test functions.

        Args:
            noise_std: Standard deviation of the observation noise. If a list is
                provided, specifies separate noise standard deviations for each
                objective in a multiobjective problem.
            negate: If True, negate the function.
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
            "bounds", torch.tensor(self._bounds, dtype=torch.double).transpose(-1, -2)
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
    constraint_noise_std: Union[None, float, list[float]] = None

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
    _max_hv: Optional[float] = None

    def __init__(
        self,
        noise_std: Union[None, float, list[float]] = None,
        negate: bool = False,
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
        super().__init__(noise_std=noise_std, negate=negate)
        ref_point = torch.tensor(self._ref_point, dtype=torch.get_default_dtype())
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
