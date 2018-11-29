#!/usr/bin/env python3

from functools import wraps
from typing import Any, Callable

from torch import Tensor


TFunc = Callable[..., Any]


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

    def tranform_arg_to_batch_mode(x: Any) -> Any:
        """
        Helper function to unsqueeze non-batch mode tensor arguments.
        """
        if isinstance(x, Tensor):
            if x.ndimension() == 2:
                return x.unsqueeze(0)
        return x

    @wraps(batch_acquisition_function)
    def decorated(X, *args, **kwargs) -> Tensor:
        # if X is b x q x d, then X is in batch mode
        is_batch_mode = X.ndimension() > 2
        if not is_batch_mode:
            X = X.unsqueeze(0)
        # transform other tensor arguments to batch mode
        for i in range(len(args)):
            args[i] = tranform_arg_to_batch_mode(args[i])
        for k in kwargs:
            kwargs[k] = tranform_arg_to_batch_mode(kwargs[k])
        val = batch_acquisition_function(X, *args, **kwargs)
        if not is_batch_mode:
            val = val.squeeze(0)
        return val

    return decorated
