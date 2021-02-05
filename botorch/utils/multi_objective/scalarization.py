#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
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

    Outcomes are first normalized to [0,1] and then an augmented
    Chebyshev scalarization is applied.

    Augmented Chebyshev scalarization:
        objective(y) = min(w * y) + alpha * sum(w * y)

    Note: this assumes maximization.

    See [Knowles2005]_ for details.

    This scalarization can be used with qExpectedImprovement to implement q-ParEGO
    as proposed in [Daulton2020qehvi]_.

    Args:
        weights: A `m`-dim tensor of weights.
        Y: A `n x m`-dim tensor of observed outcomes, which are used for
            scaling the outcomes to [0,1].
        alpha: Parameter governing the influence of the weighted sum term. The
            default value comes from [Knowles2005]_.

    Returns:
        Transform function using the objective weights.

    Example:
        >>> weights = torch.tensor([0.75, 0.25])
        >>> transform = get_aug_chebyshev_scalarization(weights, Y)
    """
    if weights.shape != Y.shape[-1:]:
        raise BotorchTensorDimensionError(
            "weights must be an `m`-dim tensor where Y is `... x m`."
            f"Got shapes {weights.shape} and {Y.shape}."
        )
    elif Y.ndim > 2:
        raise NotImplementedError("Batched Y is not currently supported.")
    Y_bounds = torch.stack([Y.min(dim=-2).values, Y.max(dim=-2).values])

    def obj(Y: Tensor, X: Optional[Tensor] = None) -> Tensor:
        # scale to [0,1]
        Y_normalized = normalize(Y, bounds=Y_bounds)
        product = weights * Y_normalized
        return product.min(dim=-1).values + alpha * product.sum(dim=-1)

    return obj
