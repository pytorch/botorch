#!/usr/bin/env python3

r"""
Some basic data transformation helpers.
"""

from functools import wraps
from typing import Any, Callable

import torch
from torch import Tensor


def squeeze_last_dim(Y: Tensor) -> Tensor:
    """Squeeze the last dimension of a Tensor."""
    return Y.squeeze(-1)


def standardize(X: Tensor) -> Tensor:
    r"""Standardize a tensor by dim=0.

    Args:
        X: tensor `n x (d)`

    Returns:
        The standardized `X`.
    """
    X_std = X.std(dim=0)
    X_std = X_std.where(X_std >= 1e-9, torch.full_like(X_std, 1.0))
    return (X - X.mean(dim=0)) / X_std


def normalize(X: Tensor, bounds: Tensor) -> Tensor:
    r"""
    Min-max normalize X to [0,1] using the provided bounds.

    Args:
        X: `... x d` tensor of data
        bounds: `2 x d` tensor of lower and upper bounds for each of the X's d
            columns.
    Returns:
        A `... x d`-dim tensor of normalized data.
    """
    return (X - bounds[0]) / (bounds[1] - bounds[0])


def unnormalize(X: Tensor, bounds: Tensor) -> Tensor:
    r"""
    Unscale X from [0,1] to the original scale.

    Args:
        X: `... x d` tensor of data
        bounds: `2 x d` tensor of lower and upper bounds for each of the X's d
            columns.
    Returns:
        A `... x d`-dim tensor of unnormalized data.
    """
    return X * (bounds[1] - bounds[0]) + bounds[0]


def t_batch_mode_transform(
    method: Callable[[Any, Tensor], Any]
) -> Callable[[Any, Tensor], Any]:
    r"""Decorates instance functions to always receive a t-batched tensor.

    Decorator for instance methods that transforms an input tensor `X` to
    t-batch mode (i.e. with at least 3 dimensions). This assumes the tensor
    has a q-batch dimension.

    Args:
        method: The (instance) method to be wrapped in the batch mode transform.

    Returns:
        The decorated instance method.
    """

    @wraps(method)
    def decorated(cls: Any, X: Tensor) -> Any:
        X = X if X.dim() > 2 else X.unsqueeze(0)
        return method(cls, X)

    return decorated


def q_batch_mode_transform(
    method: Callable[[Any, Tensor], Any]
) -> Callable[[Any, Tensor], Any]:
    r"""Decorates instance functions to always receive a q-batched tensor.

    Decorator for instance methods that transforms an input tensor `X` to
    q-batch mode. Assumes that the tensor does not have a q-batch dimension.

    Args:
        method: The (instance) method to be wrapped in the batch mode transform.

    Returns:
        The decorated instance method.
    """

    @wraps(method)
    def decorated(cls: Any, X: Tensor) -> Any:
        return method(cls, X.unsqueeze(-2))

    return decorated


def match_batch_shape(X: Tensor, Y: Tensor) -> Tensor:
    r"""Matches the batch dimension of a tensor to that of anther tensor.

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
    """
    return X.expand(X.shape[: -Y.dim()] + Y.shape[:-2] + X.shape[-2:])


def convert_to_target_pre_hook(module, *args):
    r"""Pre-hook for automatically calling `.to(X)` on module prior to `forward`"""
    module.to(args[0][0])
