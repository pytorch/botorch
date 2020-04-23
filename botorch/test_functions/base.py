#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Base class for test functions for optimization benchmarks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Module


class BaseTestProblem(Module, ABC):
    r"""Base class for test functions."""

    dim: int
    _bounds: List[Tuple[float, float]]
    _check_grad_at_opt: bool = True

    def __init__(self, noise_std: Optional[float] = None, negate: bool = False) -> None:
        r"""Base constructor for test functions.

        Arguments:
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the function.
        """
        super().__init__()
        self.noise_std = noise_std
        self.negate = negate
        self.register_buffer(
            "bounds", torch.tensor(self._bounds, dtype=torch.float).transpose(-1, -2)
        )

    def forward(self, X: Tensor, noise: bool = True) -> Tensor:
        r"""Evaluate the function on a set of points.

        Args:
            X: The point(s) at which to evaluate the function.
            noise: If `True`, add observation noise as specified by `noise_std`.
        """
        batch = X.ndimension() > 1
        X = X if batch else X.unsqueeze(0)
        f = self.evaluate_true(X=X)
        if noise and self.noise_std is not None:
            f += self.noise_std * torch.randn_like(f)
        if self.negate:
            f = -f
        return f if batch else f.squeeze(0)

    @abstractmethod
    def evaluate_true(self, X: Tensor) -> Tensor:
        r"""Evaluate the function (w/o observation noise) on a set of points."""
        pass  # pragma: no cover
