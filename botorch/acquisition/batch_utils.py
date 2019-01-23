#!/usr/bin/env python3

from functools import wraps
from typing import Any, Callable, Optional, Tuple

from torch import Size, Tensor

from ..posteriors.posterior import Posterior
from ..qmc.sobol import SobolEngine
from ..utils import draw_sobol_normal_samples, manual_seed


TFunc = Callable[..., Any]


def transform_arg_to_batch_mode(x: Any, b: int) -> Any:
    """
    Helper function to unsqueeze non-batch mode tensor arguments and expand
    the batch dimension to match b (the number of t-batches).

    A tuple (Tensor, int) can be used to specify where the batch dimension is to
    be inserted.  In this circumstance, the dimension is always inserted
    regardless of the shape of the Tensor.
    """
    if isinstance(x, Tensor):
        if x.ndimension() == 2:
            x = x.unsqueeze(0)
        if x.shape[0] < b:
            repeat_vals = [b] + [-1] * (x.ndimension() - 1)
            x = x.expand(*repeat_vals)
        return x
    elif isinstance(x, tuple):
        if isinstance(x[0], Tensor) and len(x) == 2 and isinstance(x[1], int):
            batch_dim = x[1]
            t = x[0].unsqueeze(batch_dim)
            if t.shape[batch_dim] < b:
                repeat_vals = [-1] * batch_dim + [b] + [-1] * (t.dim() - batch_dim - 1)
                t = t.expand(*repeat_vals)
            return t
        return x
    else:
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
        X_to_match = transform_arg_to_batch_mode(X_to_match, b)
    return X_to_match


def batch_mode_transform(batch_acquisition_function: TFunc) -> TFunc:
    """
    Decorator for batch acquisition functions that transforms X (`(b) x q x d`)
    and all other tensor arguments to batch mode (`b x ...`). Then the decorator calls
    the batch acquisition function and untransforms the output if X was not originally
    in batch mode.

    By default the batch dimension is inserted for all Tensor arguments at dimension
    zero.  If the batch dimension should be inserted at a different position, provide
    Tuple(Tensor, int) for an argument where the batch_acquisition expects a Tensor
    with batch dimension at index int.  Note the batch_acquisition function will receive
    Tensor, not Tuple(Tensor, int).

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
        args = list(args)
        for i in range(len(args)):
            args[i] = transform_arg_to_batch_mode(x=args[i], b=b)
        for k in kwargs:
            kwargs[k] = transform_arg_to_batch_mode(x=kwargs[k], b=b)
        val = batch_acquisition_function(X, *args, **kwargs)
        if not is_batch_mode:
            val = val.squeeze(0)
        return val

    return decorated


def construct_base_samples_from_posterior(
    posterior: Posterior, num_samples: int, qmc: bool, seed: Optional[int] = None
) -> Tuple[Tensor, int]:
    d = posterior.event_shape.numel()
    if qmc and d < SobolEngine.MAXDIM:
        base_samples = draw_sobol_normal_samples(
            d=d,
            n=num_samples,
            device=posterior.device,
            dtype=posterior.dtype,
            seed=seed,
        ).view(num_samples, *posterior.event_shape)
    else:
        with manual_seed(seed=seed):
            base_samples = posterior.get_base_samples(sample_shape=Size([num_samples]))
    # Inform batch_mode_eval decorator that batch_dim should be inserted at
    # dimension 1 if needed
    return (base_samples, 1)
