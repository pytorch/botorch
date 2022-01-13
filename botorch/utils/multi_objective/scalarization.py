#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Helper utilities for constructing scalarizations.

References

.. [Knowles2005]
    J. Knowles, "ParEGO: a hybrid algorithm with on-line landscape approximation
    for expensive multiobjective optimization problems," in IEEE Transactions
    on Evolutionary Computation, vol. 10, no. 1, pp. 50-66, Feb. 2006.
"""
from __future__ import annotations

from typing import Callable, Optional

import torch
from botorch.exceptions.errors import BotorchTensorDimensionError
from botorch.utils.transforms import normalize
from torch import Tensor


def get_chebyshev_scalarization(
    weights: Tensor, Y: Tensor, alpha: float = 0.05
) -> Callable[[Tensor, Optional[Tensor]], Tensor]:
    r"""Construct an augmented Chebyshev scalarization.

    Augmented Chebyshev scalarization:
        objective(y) = min(w * y) + alpha * sum(w * y)

    Outcomes are first normalized to [0,1] for maximization (or [-1,0] for minimization)
    and then an augmented Chebyshev scalarization is applied.

    Note: this assumes maximization of the augmented Chebyshev scalarization.
    Minimizing/Maximizing an objective is supported by passing a negative/positive
    weight for that objective. To make all w * y's have positive sign
    such that they are comparable when computing min(w * y), outcomes of minimization
    objectives are shifted from [0,1] to [-1,0].

    See [Knowles2005]_ for details.

    This scalarization can be used with qExpectedImprovement to implement q-ParEGO
    as proposed in [Daulton2020qehvi]_.

    Args:
        weights: A `m`-dim tensor of weights.
            Positive for maximization and negative for minimization.
        Y: A `n x m`-dim tensor of observed outcomes, which are used for
            scaling the outcomes to [0,1] or [-1,0].
        alpha: Parameter governing the influence of the weighted sum term. The
            default value comes from [Knowles2005]_.

    Returns:
        Transform function using the objective weights.

    Example:
        >>> weights = torch.tensor([0.75, -0.25])
        >>> transform = get_aug_chebyshev_scalarization(weights, Y)
    """
    if weights.shape != Y.shape[-1:]:
        raise BotorchTensorDimensionError(
            "weights must be an `m`-dim tensor where Y is `... x m`."
            f"Got shapes {weights.shape} and {Y.shape}."
        )
    elif Y.ndim > 2:
        raise NotImplementedError("Batched Y is not currently supported.")

    def chebyshev_obj(Y: Tensor, X: Optional[Tensor] = None) -> Tensor:
        product = weights * Y
        return product.min(dim=-1).values + alpha * product.sum(dim=-1)

    if Y.shape[-2] == 0:
        # If there are no observations, we do not need to normalize the objectives
        return chebyshev_obj
    if Y.shape[-2] == 1:
        # If there is only one observation, set the bounds to be
        # [min(Y_m), min(Y_m) + 1] for each objective m. This ensures we do not
        # divide by zero
        Y_bounds = torch.cat([Y, Y + 1], dim=0)
    else:
        # Set the bounds to be [min(Y_m), max(Y_m)], for each objective m
        Y_bounds = torch.stack([Y.min(dim=-2).values, Y.max(dim=-2).values])

    # A boolean mask indicating if minimizing an objective
    minimize = weights < 0

    def obj(Y: Tensor, X: Optional[Tensor] = None) -> Tensor:
        # scale to [0,1]
        Y_normalized = normalize(Y, bounds=Y_bounds)
        # If minimizing an objective, convert Y_normalized values to [-1,0],
        # such that min(w*y) makes sense, we want all w*y's to be positive
        Y_normalized[..., minimize] = Y_normalized[..., minimize] - 1
        return chebyshev_obj(Y=Y_normalized)

    return obj
