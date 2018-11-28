#!/usr/bin/env python3
from functools import wraps
from typing import Any, Callable

from torch import Tensor


TFunc = Callable[..., Any]


def tranform_arg_to_batch_mode(x: Any, b: int) -> Any:
    """
    Helper function to unsqueeze non-batch mode tensor arguments and expand
    the batch dimension to match b (the number of t-batches).
    """
    if isinstance(x, Tensor):
        if x.ndimension() == 2:
            x = x.unsqueeze(0)
        if x.shape[0] < b:
            repeat_vals = [b] + [-1] * (x.ndimension() - 1)
            x = x.expand(*repeat_vals)
    return x


def match_batch_size(X: Tensor, X_to_match: Tensor) -> Tensor:
    """
    Helper function to match the batch dimension of X_to_match to the batch dimension
    of X.
    Args:
        X: A `(b) x q x d` tensor
        X_to_match: A `q' x d` tensor
    Returns:
        A `(b) x q' x d` with the same batch dimension (no batch dimension) as X
    """
    if X.ndimension() > 2:
        b = X.shape[0]
        X_to_match = tranform_arg_to_batch_mode(X_to_match, b)
    return X_to_match


def batch_mode_transform(batch_acquisition_function: TFunc) -> TFunc:
    """
    Decorator for batch acquisition functions that transforms X (`(b) x q x d`)
    and all other tensor arguments to batch mode (`b x ...`). Then the decorator calls
    the batch acquisition function and untransforms the output if X was not originally
    in batch mode.

    Args:
        batch_acquisition_function: the batch acquisition function to decorate
    Returns:
        Tfunc: the decorated batch acquisition function
    """

    @wraps(batch_acquisition_function)
    def decorated(X, *args, **kwargs) -> Tensor:
        # if X is b x q x d, then X is in batch mode
        is_batch_mode = X.ndimension() > 2
        if not is_batch_mode:
            X = X.unsqueeze(0)
        b = X.shape[0]
        # transform other tensor arguments to batch mode
        for i in range(len(args)):
            args[i] = tranform_arg_to_batch_mode(x=args[i], b=b)
        for k in kwargs:
            kwargs[k] = tranform_arg_to_batch_mode(x=kwargs[k], b=b)
        val = batch_acquisition_function(X, *args, **kwargs)
        if not is_batch_mode:
            val = val.squeeze(0)
        return val

    return decorated
