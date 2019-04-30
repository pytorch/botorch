#!/usr/bin/env python3

r"""
Some basic data transformation helpers.
"""

from functools import wraps
from typing import Any, Callable, Optional

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
    return Y.squeeze(-1)


def standardize(X: Tensor) -> Tensor:
    r"""Standardize a tensor by dim=0.

    Args:
        X: A `n` or `n x d`-dim tensor

    Returns:
        The standardized `X`.

    Example:
        >>> X = torch.rand(4, 3)
        >>> X_standardized = standardize(X)
    """
    X_std = X.std(dim=0)
    X_std = X_std.where(X_std >= 1e-9, torch.full_like(X_std, 1.0))
    return (X - X.mean(dim=0)) / X_std


def normalize(X: Tensor, bounds: Tensor) -> Tensor:
    r"""Min-max normalize X to [0, 1] using the provided bounds.

    Args:
        X: `... x d` tensor of data
        bounds: `2 x d` tensor of lower and upper bounds for each of the X's d
            columns.
    Returns:
        A `... x d`-dim tensor of normalized data.

    Example:
        >>> X = torch.rand(4, 3)
        >>> bounds = torch.stack([torch.zeros(3), 0.5 * torch.ones(3)])
        >>> X_normalized = unnormalize(X, bounds)
    """
    return (X - bounds[0]) / (bounds[1] - bounds[0])


def unnormalize(X: Tensor, bounds: Tensor) -> Tensor:
    r"""Unscale X from [0, 1] to the original scale.

    Args:
        X: `... x d` tensor of data
        bounds: `2 x d` tensor of lower and upper bounds for each of the X's d
            columns.
    Returns:
        A `... x d`-dim tensor of unnormalized data.

    Example:
        >>> X_normalized = torch.rand(4, 3)
        >>> bounds = torch.stack([torch.zeros(3), 0.5 * torch.ones(3)])
        >>> X = unnormalize(X_normalized, bounds)
    """
    return X * (bounds[1] - bounds[0]) + bounds[0]


def t_batch_mode_transform(
    expected_q: Optional[int] = None,
) -> Callable[[Callable[[Any, Tensor], Any]], Callable[[Any, Tensor], Any]]:
    r"""Factory for decorators that make instance methods receive a t-batched `X` tensor.

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
        def decorated(cls: Any, X: Tensor) -> Any:
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
            return method(cls, X)

        return decorated

    return decorator


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
    return X.expand(X.shape[: -Y.dim()] + Y.shape[:-2] + X.shape[-2:])


def convert_to_target_pre_hook(module, *args):
    r"""Pre-hook for automatically calling `.to(X)` on module prior to `forward`"""
    module.to(args[0][0])
