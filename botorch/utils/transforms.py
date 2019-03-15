#!/usr/bin/env python3

from functools import wraps
from typing import Any, Callable

import torch
from torch import Tensor


def squeeze_last_dim(Y: Tensor) -> Tensor:
    """Squeeze the last dimension of a Tensor."""
    return Y.squeeze(-1)


def standardize(X: Tensor) -> Tensor:
    """Standardize a tensor by dim=0.

    Args:
        X: tensor `n x (d)`

    Returns:
        Tensor: standardized X
    """
    X_std = X.std(dim=0)
    X_std = X_std.where(X_std >= 1e-9, torch.full_like(X_std, 1.0))
    return (X - X.mean(dim=0)) / X_std


def batch_mode_transform(
    method: Callable[[Any, Tensor], Any]
) -> Callable[[Any, Tensor], Any]:
    """Decorates instance functions to always receive a t-batched tensor.

    Decorator for instance methods that transforms an an input tensor `X` to
    t-batch mode (i.e. with at least 3 dimensions).

    Args:
        method: The (instance) method to be wrapped in the batch mode transform.

    Returns:
        Callable[..., Any]: The decorated instance method.
    """

    @wraps(method)
    def decorated(cls: Any, X: Tensor) -> Any:
        X = X if X.dim() > 2 else X.unsqueeze(0)
        return method(cls, X)

    return decorated


def match_batch_shape(X: Tensor, Y: Tensor) -> Tensor:
    """Matches the batch dimension of a tensor to that of anther tensor.

    Args:
        X: A `batch_shape_X x q x d` tensor, whose batch dimensions that
            correspond to batch dimensions of `Y` are to be matched to those
            (if compatible).
        Y: A `batch_shape_Y x q' x d` tensor.

    Returns:
        Tensor: A `batch_shape_Y x q x d` tensor containing the data of `X`
            expanded to the batch dimensions of `Y` (if compatible). For
            instance, if `X` is `b'' x b' x q x d` and `Y` is `b x q x d`,
            then the returned tensor is `b'' x b x q x d`.
    """
    return X.expand(X.shape[: -Y.dim()] + Y.shape[:-2] + X.shape[-2:])
