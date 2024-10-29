#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Synthetic functions for multi-fidelity optimization benchmarks.
"""

from __future__ import annotations

import math

import torch
from botorch.test_functions.synthetic import SyntheticTestFunction
from torch import Tensor


class AugmentedBranin(SyntheticTestFunction):
    r"""Augmented Branin test function for multi-fidelity optimization.

    3-dimensional function with domain `[-5, 10] x [0, 15] * [0,1]`, where
    the last dimension of is the fidelity parameter:

        B(x) = (x_2 - (b - 0.1 * (1 - x_3))x_1^2 + c x_1 - r)^2 +
            10 (1-t) cos(x_1) + 10

    Here `b`, `c`, `r` and `t` are constants where `b = 5.1 / (4 * math.pi ** 2)`
    `c = 5 / math.pi`, `r = 6`, `t = 1 / (8 * math.pi)`.
    B has infinitely many minimizers with `x_1 = -pi, pi, 3pi`
    and `B_min = 0.397887`
    """

    dim = 3
    _bounds = [(-5.0, 10.0), (0.0, 15.0), (0.0, 1.0)]
    _optimal_value = 0.397887
    _optimizers = [  # this is a subset, ther are infinitely many optimizers
        (-math.pi, 12.275, 1),
        (math.pi, 1.3867356039019576, 0.1),
        (math.pi, 1.781519779945532, 0.5),
        (math.pi, 2.1763039559891064, 0.9),
    ]

    def evaluate_true(self, X: Tensor) -> Tensor:
        t1 = (
            X[..., 1]
            - (5.1 / (4 * math.pi**2) - 0.1 * (1 - X[..., 2])) * X[..., 0].pow(2)
            + 5 / math.pi * X[..., 0]
            - 6
        )
        t2 = 10 * (1 - 1 / (8 * math.pi)) * torch.cos(X[..., 0])
        return t1.pow(2) + t2 + 10


class AugmentedHartmann(SyntheticTestFunction):
    r"""Augmented Hartmann synthetic test function.

    7-dimensional function (typically evaluated on `[0, 1]^7`), where the last
    dimension is the fidelity parameter.

        H(x) = -(ALPHA_1 - 0.1 * (1-x_7)) * exp(- sum_{j=1}^6 A_1j (x_j - P_1j) ** 2) -
            sum_{i=2}^4 ALPHA_i exp( - sum_{j=1}^6 A_ij (x_j - P_ij) ** 2)

    H has a unique global minimizer
    `x = [0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573, 1.0]`

    with `H_min = -3.32237`
    """

    dim = 7
    _bounds = [(0.0, 1.0) for _ in range(7)]
    _optimal_value = -3.32237
    _optimizers = [(0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573, 1.0)]
    _check_grad_at_opt = False

    def __init__(
        self,
        noise_std: float | None = None,
        negate: bool = False,
        dtype: torch.dtype = torch.double,
    ) -> None:
        r"""
        Args:
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the function.
            dtype: The dtype that is used for the bounds of the function.
        """
        super().__init__(noise_std=noise_std, negate=negate, dtype=dtype)
        self.register_buffer("ALPHA", torch.tensor([1.0, 1.2, 3.0, 3.2]))
        A = [
            [10, 3, 17, 3.5, 1.7, 8],
            [0.05, 10, 17, 0.1, 8, 14],
            [3, 3.5, 1.7, 10, 17, 8],
            [17, 8, 0.05, 10, 0.1, 14],
        ]
        P = [
            [1312, 1696, 5569, 124, 8283, 5886],
            [2329, 4135, 8307, 3736, 1004, 9991],
            [2348, 1451, 3522, 2883, 3047, 6650],
            [4047, 8828, 8732, 5743, 1091, 381.0],
        ]
        self.register_buffer("A", torch.tensor(A))
        self.register_buffer("P", torch.tensor(P))

    def evaluate_true(self, X: Tensor) -> Tensor:
        self.to(device=X.device, dtype=X.dtype)
        inner_sum = torch.sum(
            self.A * (X[..., :6].unsqueeze(-2) - 0.0001 * self.P).pow(2), dim=-1
        )
        alpha1 = self.ALPHA[0] - 0.1 * (1 - X[..., 6])
        H = (
            -(torch.sum(self.ALPHA[1:] * torch.exp(-inner_sum)[..., 1:], dim=-1))
            - alpha1 * torch.exp(-inner_sum)[..., 0]
        )
        return H


class AugmentedRosenbrock(SyntheticTestFunction):
    r"""Augmented Rosenbrock synthetic test function for multi-fidelity optimization.

    d-dimensional function (usually evaluated on `[-5, 10]^(d-2) * [0, 1]^2`),
    where the last two dimensions are the fidelity parameters:

        f(x) = sum_{i=1}^{d-1} (100 (x_{i+1} - x_i^2 + 0.1 * (1-x_{d-1}))^2 +
            (x_i - 1 + 0.1 * (1 - x_d)^2)^2)

    f has one minimizer for its global minimum at `z_1 = (1, 1, ..., 1)` with
    `f(z_i) = 0.0`.
    """

    _optimal_value = 0.0

    def __init__(
        self,
        dim=3,
        noise_std: float | None = None,
        negate: bool = False,
        dtype: torch.dtype = torch.double,
    ) -> None:
        r"""
        Args:
            dim: The (input) dimension. Must be at least 3.
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the function.
            dtype: The dtype that is used for the bounds of the function.
        """
        if dim < 3:
            raise ValueError(
                "AugmentedRosenbrock must be defined it at least 3 dimensions"
            )
        self.dim = dim
        self._bounds = [(-5.0, 10.0) for _ in range(self.dim)]
        self._optimizers = [tuple(1.0 for _ in range(self.dim))]
        super().__init__(noise_std=noise_std, negate=negate, dtype=dtype)

    def evaluate_true(self, X: Tensor) -> Tensor:
        X_curr = X[..., :-3]
        X_next = X[..., 1:-2]
        t1 = 100 * (X_next - X_curr.pow(2) + 0.1 * (1 - X[..., -2:-1])).pow(2)
        t2 = (X_curr - 1 + 0.1 * (1 - X[..., -1:]).pow(2)).pow(2)
        return -((t1 + t2).sum(dim=-1))
