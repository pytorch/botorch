#!/usr/bin/env python3

from contextlib import contextmanager
from typing import Dict, Generator, List, Optional, Union

import torch
from torch import Tensor


def check_convergence(
    loss_trajectory: List[float],
    param_trajectory: Dict[str, List[Tensor]],
    options: Dict[str, Union[float, str]],
    max_iter: int = 50,
) -> bool:
    """Check convergence of optimization."""
    # TODO: Be A LOT smarter about this
    # TODO: Make this work in batch mode (see parallel L-BFGS-P)
    if len(loss_trajectory) >= max_iter:
        return True
    else:
        return False


def columnwise_clamp(
    X: Tensor,
    lower: Optional[Union[float, Tensor]] = None,
    upper: Optional[Union[float, Tensor]] = None,
) -> Tensor:
    """Clamp values of a Tensor in column-wise fashion.

    This function is useful in conjunction with optimizers from the torch.optim
    package, which don't natively handle constraints. If you apply this after
    a gradient step you can be fancy and call it "projected gradient descent".

    Args:
        X: The `n x d` input tensor.
        lower: The column-wise lower bounds. If scalar, apply bound to all columns.
        upper: The column-wise upper bounds. If scalar, apply bound to all columns.

    Returns:
        The clamped tensor.

    """
    min_bounds = _expand_bounds(lower, X)
    max_bounds = _expand_bounds(upper, X)
    if min_bounds is not None and max_bounds is not None:
        if torch.any(min_bounds > max_bounds):
            raise ValueError("Minimum values must be <= maximum values")
    Xout = X
    if min_bounds is not None:
        Xout = Xout.max(min_bounds)
    if max_bounds is not None:
        Xout = Xout.min(max_bounds)
    return Xout


@contextmanager
def manual_seed(seed: Optional[int] = None) -> Generator:
    """Contextmanager for manual setting the torch.random seed"""
    old_state = torch.random.get_rng_state()
    try:
        if seed is not None:
            torch.random.manual_seed(seed)
        yield
    finally:
        if seed is not None:
            torch.random.set_rng_state(old_state)


def _expand_bounds(
    bounds: Optional[Union[float, Tensor]], X: Tensor
) -> Optional[Tensor]:
    # TODO: Make work properly with batch mode
    if bounds is not None:
        target = {"dtype": X.dtype, "device": X.device}
        if not torch.is_tensor(bounds):
            bounds = torch.tensor(bounds)
        if len(bounds.shape) == 0:
            ebounds = bounds.expand(1, X.shape[1])
        elif len(bounds.shape) == 1:
            ebounds = bounds.view(1, -1)
        else:
            ebounds = bounds
        if ebounds.shape[1] != X.shape[1]:
            raise RuntimeError(
                "Bounds must either be a single value or the same dimension as X"
            )
        return ebounds.to(**target)
    else:
        return None
