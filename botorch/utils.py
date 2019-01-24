#!/usr/bin/env python3

from contextlib import contextmanager
from typing import Callable, Dict, Generator, List, Optional, Union

import torch
from botorch.qmc.normal import NormalQMCEngine

# TODO: Use torch Sobol engine: https://github.com/pytorch/pytorch/pull/10505
from botorch.qmc.sobol import SobolEngine
from torch import Tensor


def check_convergence(
    loss_trajectory: List[float],
    param_trajectory: Dict[str, List[Tensor]],
    options: Dict[str, Union[float, str]],
) -> bool:
    """Check convergence of optimization for pytorch optimizers.

    Right now this is just a dummy function and only checks for maxiter.

    """
    maxiter: int = options.get("maxiter", 50)
    # TODO: Be A LOT smarter about this
    # TODO: Make this work in batch mode (see parallel L-BFGS-P)
    if len(loss_trajectory) >= maxiter:
        return True
    else:
        return False


def _fix_feature(Z: Tensor, value: Optional[float]) -> Tensor:
    if value is None:
        return Z.detach()
    return torch.full_like(Z, value)


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
    """Clamp values of a Tensor in column-wise fashion (with support for t-batches).

    This function is useful in conjunction with optimizers from the torch.optim
    package, which don't natively handle constraints. If you apply this after
    a gradient step you can be fancy and call it "projected gradient descent".

    Args:
        X: The `b x n x d` input tensor. If 2-dimensional, b is assumed to be 1.
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
    """
    Expand the dimension of bounds if necessary such that the 1st dimension of
        bounds is the same as the last dimension of `X`.

    Args:
        bounds: a bound (either upper or lower) of each column (last dimension)
            of X. If this is a single float, then all columns have the same bound.
        X: `... x d` tensor

    Returns:
        A tensor of bounds expanded to be compatible with the size of `X` if
            bounds is not None, and None if bounds is None

    """
    if bounds is not None:
        if not torch.is_tensor(bounds):
            bounds = torch.tensor(bounds)
        if len(bounds.shape) == 0:
            ebounds = bounds.expand(1, X.shape[-1])
        elif len(bounds.shape) == 1:
            ebounds = bounds.view(1, -1)
        else:
            ebounds = bounds
        if ebounds.shape[1] != X.shape[-1]:
            raise RuntimeError(
                "Bounds must either be a single value or the same dimension as X"
            )
        return ebounds.to(dtype=X.dtype, device=X.device)
    else:
        return None


def gen_x_uniform(b: int, q: int, bounds: Tensor) -> Tensor:
    """Generate `b` random `q`-batches with elements within the specified bounds.

    Args:
        n: The number of `q`-batches to sample.
        q: The size of the `q`-batches.
        bounds: A `2 x d` tensor where bounds[0] (bounds[1]) contains the lower
            (upper) bounds for each column.

    Returns:
        A `b x q x d` tensor with elements uniformly sampled from the box
            specified by bounds.

    """
    x_ranges = torch.sum(bounds * torch.tensor([[-1.0], [1.0]]).type_as(bounds), dim=0)
    return bounds[0] + torch.rand((b, q, bounds.shape[1])).type_as(bounds) * x_ranges


def get_objective_weights_transform(
    objective_weights: Optional[Tensor]
) -> Callable[[Tensor], Tensor]:
    """
    Create a callable mapping a Tensor of size `b x q x t` to a Tensor of size
        `b x q`, where `t` is the number of outputs (tasks) of the model using
        the objective weights. This callable supports broadcasting (e.g.
        for calling on a tensor of shape `mc_samples x b x q x t`. For t=1, the
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
    return lambda Y: torch.sum(Y * weights.view(1, 1, -1), dim=-1)


def draw_sobol_samples(
    bounds: Tensor, n: int, q: int, seed: Optional[int] = None
) -> Tensor:
    """Draw qMC samples from the box defined by bounds

    NOTE: This currently uses botorch's own cython SobolEngine. In the future
        (once perf issues are resolved), this will instead use the native torch
        implementation from https://github.com/pytorch/pytorch/pull/10505

    Args:
        bounds: A `2 x d` dimensional tensor specifying box constraints on a
            `d`-dimensional space, where bounds[0, :] and bounds[1, :] correspond
            to lower and upper bounds, respectively.
        n: The number of (q-batch) samples.
        q: The size of each q-batch.
        seed: The seed used for initializing Owen scrambling. If None (default),
            use a random seed.

    Returns:
        An `n x q x d` tensor `X` of qMC samples from the box defined by bounds

    """
    d = bounds.shape[-1]
    lower = bounds[0]
    rng = bounds[1] - bounds[0]
    sobol_engine = SobolEngine(d, scramble=True, seed=seed)
    samples_np = sobol_engine.draw(n * q).reshape(n, q, d)
    samples_raw = torch.from_numpy(samples_np).type_as(lower)
    return lower + rng * samples_raw


def draw_sobol_normal_samples(
    d: int,
    n: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    seed: Optional[int] = None,
) -> Tensor:
    """Draw qMC samples from a multi-variate standard normal N(0, I_d)

    A primary use-case for this functionality is to compute an QMC average
    of f(X) over X where each element of X is drawn N(0, 1).

    NOTE: This currently uses botorch's own cython SobolEngine. In the future
        (once perf issues are resolved), this will instead use the native torch
        implementation from https://github.com/pytorch/pytorch/pull/10505

    Args:
        d: The dimension of the normal distribution
        n: The number of samples to return
        device: The torch device
        dtype:  The torch dtype
        seed: The seed used for initializing Owen scrambling. If None (default),
            use a random seed.

    Returns:
        An tensor of qMC standard normal samples with dimension `n x d` with device
        and dtype specified by the input.

    """
    normal_qmc_engine = NormalQMCEngine(d=d, seed=seed, inv_transform=True)
    samples_np = normal_qmc_engine.draw(n)
    return torch.from_numpy(samples_np).to(
        dtype=torch.float if dtype is None else dtype,
        device=device,  # None here will leave it on the cpu
    )


def standardize(X: Tensor) -> Tensor:
    """
    Standardize a tensor by dim=0
    Args:
        X: tensor `n x (d)`
    Returns:
        Tensor: standardized X
    """
    X_std = X.std(dim=0)
    X_std = X_std.where(X_std >= 1e-9, torch.full_like(X_std, 1.0))
    return (X - X.mean(dim=0)) / X_std
