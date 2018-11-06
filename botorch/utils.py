#!/usr/bin/env python3

from contextlib import contextmanager
from typing import Callable, Dict, Generator, List, Optional, Union

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


def _fix_feature(Z: Tensor, value: Optional[float]) -> Tensor:
    Z_detached = Z.detach().requires_grad_(False)
    if value is None:
        return Z_detached
    else:
        return Z_detached.fill_(value)


def fix_features(
    X: Tensor, fixed_features: Optional[Dict[int, Optional[float]]] = None
) -> Tensor:
    """Fix feature values in a Tensor.  These fixed features
        will have zero gradient in downstream calculations.

    Args:
        X: input Tensor with shape (..., p) where p is the number of features
        fixed_features:  A dictionary with keys as column
            indices and values equal to what the feature should be set to
            in X.  If the value is None, that column is just
            considered fixed.  Keys should be in the range [0, p - 1].

    Returns:
        Tensor X with fixed features.
    """
    if fixed_features is None:
        return X
    else:
        return torch.cat(
            [
                X[..., i].unsqueeze(-1)
                if i not in fixed_features
                else _fix_feature(X[..., i].unsqueeze(-1), fixed_features[i])
                for i in range(X.shape[-1])
            ],
            dim=-1,
        )


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


def gen_x_uniform(n: int, bounds: Tensor) -> Tensor:
    """
    Generate random n points within the bounds specified for each column.
    Args:
        n: The number of points to sample.
        bounds: A (2 x d) tensor where bounds[0] is contains the lower bounds
        for each column and bounds[1] contains the upper bounds.
    Returns:
        torch.Tensor: a (n x d) tensor of points
    """
    x_ranges = torch.sum(
        bounds
        * torch.tensor([-1.0, 1.0], dtype=bounds.dtype, device=bounds.device).view(
            2, -1
        ),
        dim=0,
    )
    return (
        torch.rand((n, bounds.shape[1]), dtype=bounds.dtype, device=bounds.device)
        * x_ranges
        + bounds[0]
    )


def get_objective_weights_transform(
    objective_weights: Optional[Tensor]
) -> Callable[[Tensor], Tensor]:
    """
    Create a callable mapping a Tensor of size `b x q x t x mc_samples`
        to a Tensor of size `b x q x mc_samples`, where `t` is the number of
        outputs (tasks) of the model using the objective weights. For t=1, the
        objective weight determines the optimization direction.
    Args:
        objective_weights: a 1-dimensional Tensor containing a weight for each task.
        If not provided, the identity mapping is used.
    Returns:
        Callable[Tensor, Tensor]: transform function using the objective weights
    """

    if objective_weights is None:
        return lambda Y: Y
    weights = objective_weights.view(-1)
    if weights.shape[0] == 0:
        return lambda Y: Y
    elif weights.shape[0] == 1:
        return lambda Y: Y * weights[0]
    # TODO: replace with einsum once performance issues are resolved upstream.
    return lambda Y: torch.sum(Y * weights.view(1, 1, -1, 1), dim=2)
