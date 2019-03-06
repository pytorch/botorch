#!/usr/bin/env python3

"""
Utilities for batch acquisition functions.
"""

from functools import wraps
from typing import Any, Callable

import torch
from torch import Tensor


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


def batch_mode_transform(
    batch_acquisition_function: Callable[..., Tensor]
) -> Callable[..., Tensor]:
    """Decorates acquisition functions to always receive t-batch-mode arguments.

    Decorator for batch acquisition functions that transforms an input tensor
    `X` (first argument is always assumed to be a Tensor) and all other Tensor
    arguments to t-batch mode (at least 3 dimensions). It also ensures that
    tensors have the same t-batch shape.
    The decorator calls the batch acquisition function and untransforms the
    output if `X` was not originally in batch mode.

    Args:
        batch_acquisition_function: the batch acquisition function to decorate

    Returns:
        Callable[..., Tensor]: the decorated batch acquisition function
    """

    @wraps(batch_acquisition_function)
    def decorated(X, *args, **kwargs) -> Tensor:
        # if X is `b x q x d`, then X is in batch mode
        is_batch_mode = X.dim() > 2
        if not is_batch_mode:
            X = X.unsqueeze(0)
        # transform other tensor arguments to batch mode
        transf_args = [
            match_batch_shape(arg, X) if torch.is_tensor(arg) else arg for arg in args
        ]
        transf_kwargs = {
            key: match_batch_shape(arg, X) if torch.is_tensor(arg) else arg
            for key, arg in kwargs.items()
        }
        val = batch_acquisition_function(X, *transf_args, **transf_kwargs)
        if not is_batch_mode:
            val = val.squeeze(0)
        return val

    return decorated


def batch_mode_instance_method(
    instance_method: Callable[[Any, Tensor], Any]
) -> Callable[[Any, Tensor], Any]:
    """Decorates instance functions to always receive a t-batched tensor.

    Decorator for instance methods that transforms an an input tensor `X` to
    t-batch mode (i.e. with at least 3 dimensions).

    Args:
        instance_method: The instance method

    Returns:
        Callable[..., Any]: the decorated instance method
    """

    @wraps(instance_method)
    def decorated(cls: Any, X: Tensor) -> Any:
        X = X if X.dim() > 2 else X.unsqueeze(0)
        return instance_method(cls, X)

    return decorated
