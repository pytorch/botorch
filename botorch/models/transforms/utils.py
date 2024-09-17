#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from functools import wraps

import torch
from torch import Tensor


def lognorm_to_norm(mu: Tensor, Cov: Tensor) -> tuple[Tensor, Tensor]:
    """Compute mean and covariance of a MVN from those of the associated log-MVN

    If `Y` is log-normal with mean mu_ln and covariance Cov_ln, then
    `X ~ N(mu_n, Cov_n)` with

        Cov_n_{ij} = log(1 + Cov_ln_{ij} / (mu_ln_{i} * mu_n_{j}))
        mu_n_{i} = log(mu_ln_{i}) - 0.5 * log(1 + Cov_ln_{ii} / mu_ln_{i}**2)

    Args:
        mu: A `batch_shape x n` mean vector of the log-Normal distribution.
        Cov: A `batch_shape x n x n` covariance matrix of the log-Normal
            distribution.

    Returns:
        A two-tuple containing:

        - The `batch_shape x n` mean vector of the Normal distribution
        - The `batch_shape x n x n` covariance matrix of the Normal distribution
    """
    Cov_n = torch.log1p(Cov / (mu.unsqueeze(-1) * mu.unsqueeze(-2)))
    mu_n = torch.log(mu) - 0.5 * torch.diagonal(Cov_n, dim1=-1, dim2=-2)
    return mu_n, Cov_n


def norm_to_lognorm(mu: Tensor, Cov: Tensor) -> tuple[Tensor, Tensor]:
    """Compute mean and covariance of a log-MVN from its MVN sufficient statistics

    If `X ~ N(mu, Cov)` and `Y = exp(X)`, then `Y` is log-normal with

        mu_ln_{i} = exp(mu_{i} + 0.5 * Cov_{ii})
        Cov_ln_{ij} = exp(mu_{i} + mu_{j} + 0.5 * (Cov_{ii} + Cov_{jj})) *
        (exp(Cov_{ij}) - 1)

    Args:
        mu: A `batch_shape x n` mean vector of the Normal distribution.
        Cov: A `batch_shape x n x n` covariance matrix of the Normal distribution.

    Returns:
        A two-tuple containing:

        - The `batch_shape x n` mean vector of the log-Normal distribution.
        - The `batch_shape x n x n` covariance matrix of the log-Normal
            distribution.
    """
    diag = torch.diagonal(Cov, dim1=-1, dim2=-2)
    b = mu + 0.5 * diag
    mu_ln = torch.exp(b)
    Cov_ln = torch.special.expm1(Cov) * torch.exp(b.unsqueeze(-1) + b.unsqueeze(-2))
    return mu_ln, Cov_ln


def norm_to_lognorm_mean(mu: Tensor, var: Tensor) -> Tensor:
    """Compute mean of a log-MVN from its MVN marginals

    Args:
        mu: A `batch_shape x n` mean vector of the Normal distribution.
        var: A `batch_shape x n` variance vectorof the Normal distribution.

    Returns:
        The `batch_shape x n` mean vector of the log-Normal distribution.
    """
    return torch.exp(mu + 0.5 * var)


def norm_to_lognorm_variance(mu: Tensor, var: Tensor) -> Tensor:
    """Compute variance of a log-MVN from its MVN marginals

    Args:
        mu: A `batch_shape x n` mean vector of the Normal distribution.
        var: A `batch_shape x n` variance vectorof the Normal distribution.

    Returns:
        The `batch_shape x n` variance vector of the log-Normal distribution.
    """
    b = mu + 0.5 * var
    return torch.special.expm1(var) * torch.exp(2 * b)


def expand_and_copy_tensor(X: Tensor, batch_shape: torch.Size) -> Tensor:
    r"""Expand and copy X according to batch_shape.

    Args:
        X: A `input_batch_shape x n x d`-dim tensor of inputs.
        batch_shape: The new batch shape.

    Returns:
        A `new_batch_shape x n x d`-dim tensor of inputs, where `new_batch_shape`
        is `input_batch_shape` against `batch_shape`.
    """
    try:
        batch_shape = torch.broadcast_shapes(X.shape[:-2], batch_shape)
    except RuntimeError:
        raise RuntimeError(
            f"Provided batch shape ({batch_shape}) and input batch shape "
            f"({X.shape[:-2]}) are not broadcastable."
        )
    expand_shape = batch_shape + X.shape[-2:]
    return X.expand(expand_shape).clone()


def subset_transform(transform):
    r"""Decorator of an input transform function to separate out indexing logic."""

    @wraps(transform)
    def f(self, X: Tensor) -> Tensor:
        if not hasattr(self, "indices") or self.indices is None:
            return transform(self, X)
        has_shape = hasattr(self, "batch_shape")
        Y = expand_and_copy_tensor(X, self.batch_shape) if has_shape else X.clone()
        Y[..., self.indices] = transform(self, X[..., self.indices])
        return Y

    return f
