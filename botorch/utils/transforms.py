#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Some basic data transformation helpers.
"""

from __future__ import annotations

import warnings
from functools import wraps
from typing import Any, Callable, List, Optional

import torch
from torch import Tensor


def squeeze_last_dim(Y: Tensor) -> Tensor:
    r"""Squeeze the last dimension of a Tensor.

    Args:
        Y: A `... x d`-dim Tensor.

    Returns:
        The input tensor with last dimension squeezed.

    Example:
        >>> Y = torch.rand(4, 3)
        >>> Y_squeezed = squeeze_last_dim(Y)
    """
    warnings.warn(
        "`botorch.utils.transforms.squeeze_last_dim` is deprecated. "
        "Simply use `.squeeze(-1)1 instead.",
        DeprecationWarning,
    )
    return Y.squeeze(-1)


def standardize(Y: Tensor) -> Tensor:
    r"""Standardizes (zero mean, unit variance) a tensor by dim=-2.

    If the tensor is single-dimensional, simply standardizes the tensor.
    If for some batch index all elements are equal (or if there is only a single
    data point), this function will return 0 for that batch index.

    Args:
        Y: A `batch_shape x n x m`-dim tensor.

    Returns:
        The standardized `Y`.

    Example:
        >>> Y = torch.rand(4, 3)
        >>> Y_standardized = standardize(Y)
    """
    stddim = -1 if Y.dim() < 2 else -2
    Y_std = Y.std(dim=stddim, keepdim=True)
    Y_std = Y_std.where(Y_std >= 1e-9, torch.full_like(Y_std, 1.0))
    return (Y - Y.mean(dim=stddim, keepdim=True)) / Y_std


def normalize(X: Tensor, bounds: Tensor) -> Tensor:
    r"""Min-max normalize X w.r.t. the provided bounds.

    Args:
        X: `... x d` tensor of data
        bounds: `2 x d` tensor of lower and upper bounds for each of the X's d
            columns.

    Returns:
        A `... x d`-dim tensor of normalized data, given by
            `(X - bounds[0]) / (bounds[1] - bounds[0])`. If all elements of `X`
            are contained within `bounds`, the normalized values will be
            contained within `[0, 1]^d`.

    Example:
        >>> X = torch.rand(4, 3)
        >>> bounds = torch.stack([torch.zeros(3), 0.5 * torch.ones(3)])
        >>> X_normalized = normalize(X, bounds)
    """
    return (X - bounds[0]) / (bounds[1] - bounds[0])


def unnormalize(X: Tensor, bounds: Tensor) -> Tensor:
    r"""Un-normalizes X w.r.t. the provided bounds.

    Args:
        X: `... x d` tensor of data
        bounds: `2 x d` tensor of lower and upper bounds for each of the X's d
            columns.

    Returns:
        A `... x d`-dim tensor of unnormalized data, given by
            `X * (bounds[1] - bounds[0]) + bounds[0]`. If all elements of `X`
            are contained in `[0, 1]^d`, the un-normalized values will be
            contained within `bounds`.

    Example:
        >>> X_normalized = torch.rand(4, 3)
        >>> bounds = torch.stack([torch.zeros(3), 0.5 * torch.ones(3)])
        >>> X = unnormalize(X_normalized, bounds)
    """
    return X * (bounds[1] - bounds[0]) + bounds[0]


def normalize_indices(indices: Optional[List[int]], d: int) -> Optional[List[int]]:
    r"""Normalize a list of indices to ensure that they are positive.

    Args:
        indices: A list of indices (may contain negative indices for indexing
            "from the back").
        d: The dimension of the tensor to index.

    Returns:
        A normalized list of indices such that each index is between `0` and
        `d-1`, or None if indices is None.
    """
    if indices is None:
        return indices
    normalized_indices = []
    for i in indices:
        if i < 0:
            i = i + d
        if i < 0 or i > d - 1:
            raise ValueError(f"Index {i} out of bounds for tensor or length {d}.")
        normalized_indices.append(i)
    return normalized_indices


def t_batch_mode_transform(
    expected_q: Optional[int] = None,
) -> Callable[[Callable[[Any, Tensor], Any]], Callable[[Any, Tensor], Any]]:
    r"""Factory for decorators taking a t-batched `X` tensor.

    This method creates decorators for instance methods to transform an input tensor
    `X` to t-batch mode (i.e. with at least 3 dimensions). This assumes the tensor
    has a q-batch dimension. The decorator also checks the q-batch size if `expected_q`
    is provided.

    Args:
        expected_q: The expected q-batch size of X. If specified, this will raise an
            AssertitionError if X's q-batch size does not equal expected_q.

    Returns:
        The decorated instance method.

    Example:
        >>> class ExampleClass:
        >>>     @t_batch_mode_transform(expected_q=1)
        >>>     def single_q_method(self, X):
        >>>         ...
        >>>
        >>>     @t_batch_mode_transform()
        >>>     def arbitrary_q_method(self, X):
        >>>         ...
    """

    def decorator(method: Callable[[Any, Tensor], Any]) -> Callable[[Any, Tensor], Any]:
        @wraps(method)
        def decorated(cls: Any, X: Tensor, **kwargs: Any) -> Any:
            if X.dim() < 2:
                raise ValueError(
                    f"{type(cls).__name__} requires X to have at least 2 dimensions,"
                    f" but received X with only {X.dim()} dimensions."
                )
            elif expected_q is not None and X.shape[-2] != expected_q:
                raise AssertionError(
                    f"Expected X to be `batch_shape x q={expected_q} x d`, but"
                    f" got X with shape {X.shape}."
                )
            X = X if X.dim() > 2 else X.unsqueeze(0)
            return method(cls, X, **kwargs)

        return decorated

    return decorator


def concatenate_pending_points(
    method: Callable[[Any, Tensor], Any]
) -> Callable[[Any, Tensor], Any]:
    r"""Decorator concatenating X_pending into an acquisition function's argument.

    This decorator works on the `forward` method of acquisition functions taking
    a tensor `X` as the argument. If the acquisition function has an `X_pending`
    attribute (that is not `None`), this is concatenated into the input `X`,
    appropriately expanding the pending points to match the batch shape of `X`.

    Example:
        >>> class ExampleAcquisitionFunction:
        >>>     @concatenate_pending_points
        >>>     @t_batch_mode_transform()
        >>>     def forward(self, X):
        >>>         ...
    """

    @wraps(method)
    def decorated(cls: Any, X: Tensor, **kwargs: Any) -> Any:
        if cls.X_pending is not None:
            X = torch.cat([X, match_batch_shape(cls.X_pending, X)], dim=-2)
        return method(cls, X, **kwargs)

    return decorated


def match_batch_shape(X: Tensor, Y: Tensor) -> Tensor:
    r"""Matches the batch dimension of a tensor to that of another tensor.

    Args:
        X: A `batch_shape_X x q x d` tensor, whose batch dimensions that
            correspond to batch dimensions of `Y` are to be matched to those
            (if compatible).
        Y: A `batch_shape_Y x q' x d` tensor.

    Returns:
        A `batch_shape_Y x q x d` tensor containing the data of `X` expanded to
        the batch dimensions of `Y` (if compatible). For instance, if `X` is
        `b'' x b' x q x d` and `Y` is `b x q x d`, then the returned tensor is
        `b'' x b x q x d`.

    Example:
        >>> X = torch.rand(2, 1, 5, 3)
        >>> Y = torch.rand(2, 6, 4, 3)
        >>> X_matched = match_batch_shape(X, Y)
        >>> X_matched.shape
        torch.Size([2, 6, 5, 3])

    """
    return X.expand(X.shape[: -(Y.dim())] + Y.shape[:-2] + X.shape[-2:])


def convert_to_target_pre_hook(module, *args):
    r"""Pre-hook for automatically calling `.to(X)` on module prior to `forward`"""
    module.to(args[0][0])
