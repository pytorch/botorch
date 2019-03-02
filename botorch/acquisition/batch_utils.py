#!/usr/bin/env python3

"""
Utilities for batch acquisition functions
"""

from functools import wraps
from typing import Any, Callable, Optional

import torch
from torch import Size, Tensor

from ..posteriors.posterior import Posterior
from ..qmc.sobol import SobolEngine
from ..utils import draw_sobol_normal_samples, manual_seed


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


def construct_base_samples_from_posterior(
    posterior: Posterior, num_samples: int, qmc: bool, seed: Optional[int] = None
) -> Tensor:
    """Construct a tensor of normally distributed base samples.

    Args:
        posterior: A Posterior object.
        num_samples: The number of base_samples to draw.
        qmc: If True, use quasi-MC sampling (instead of iid draws).
        seed: If provided, use as a seed for the RNG.

    Returns:
        base_samples: A `num_samples x 1 x q x t` dimensional Tensor of base
            samples, drawn from a N(0, I_qt) distribution (using QMC if
            `qmc=True`). Here `q` and `t` are the same as in the posterior's
            `event_shape` `b x q x t`. Importantly, this only obtain a single
            t-batch of samples, so as to not introduce any sampling variance
            across t-batches.
    """
    output_shape = posterior.event_shape[-2:]  # shape of joint output: q x t
    base_sample_shape = torch.Size([1] * len(posterior.batch_shape)) + output_shape
    output_dim = output_shape.numel()
    if qmc and output_dim < SobolEngine.MAXDIM:
        base_samples = draw_sobol_normal_samples(
            d=output_dim,
            n=num_samples,
            device=posterior.device,
            dtype=posterior.dtype,
            seed=seed,
        )
        base_samples = base_samples.view(num_samples, *base_sample_shape)
    else:
        with manual_seed(seed=seed):
            base_samples = posterior.get_base_samples(
                sample_shape=Size([num_samples]), collapse_batch_dims=True
            )
    return base_samples
