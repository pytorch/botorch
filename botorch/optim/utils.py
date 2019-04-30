#!/usr/bin/env python3

r"""
Utilities for optimization.
"""

from inspect import signature
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.mlls.marginal_log_likelihood import MarginalLogLikelihood
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from gpytorch.mlls.variational_elbo import VariationalELBO
from torch import Tensor


def check_convergence(
    loss_trajectory: List[float],
    param_trajectory: Dict[str, List[Tensor]],
    options: Dict[str, Union[float, str]],
) -> bool:
    r"""Check convergence of optimization for pytorch optimizers.

    Right now this is just a dummy function and only checks for maxiter.

    Args:
        loss_trajectory: A list containing the loss value at each iteration.
        param_trajectory: A dictionary mapping each parameter name to a list of Tensors
            where the `i`th Tensor is the parameter value at iteration `i`.
        options: dictionary of options. Currently only "maxiter" is supported.

    Returns:
        A boolean indicating whether optimization has converged.
    """
    maxiter: int = options.get("maxiter", 50)
    # TODO: Be A LOT smarter about this
    # TODO: Make this work in batch mode (see parallel L-BFGS-P)
    if len(loss_trajectory) >= maxiter:
        return True
    else:
        return False


def columnwise_clamp(
    X: Tensor,
    lower: Optional[Union[float, Tensor]] = None,
    upper: Optional[Union[float, Tensor]] = None,
) -> Tensor:
    r"""Clamp values of a Tensor in column-wise fashion (with support for t-batches).

    This function is useful in conjunction with optimizers from the torch.optim
    package, which don't natively handle constraints. If you apply this after
    a gradient step you can be fancy and call it "projected gradient descent".

    Args:
        X: The `b x n x d` input tensor. If 2-dimensional, `b` is assumed to be 1.
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


def fix_features(
    X: Tensor, fixed_features: Optional[Dict[int, Optional[float]]] = None
) -> Tensor:
    r"""Fix feature values in a Tensor.

    The fixed features will have zero gradient in downstream calculations.

    Args:
        X: input Tensor with shape `... x p`, where `p` is the number of features
        fixed_features: A dictionary with keys as column indices and values
            equal to what the feature should be set to in `X`. If the value is
            None, that column is just considered fixed. Keys should be in the
            range `[0, p - 1]`.

    Returns:
        The tensor X with fixed features.
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


def _fix_feature(Z: Tensor, value: Optional[float]) -> Tensor:
    r"""Helper function returns a Tensor like `Z` filled with `value` if provided."""
    if value is None:
        return Z.detach()
    return torch.full_like(Z, value)


def _expand_bounds(
    bounds: Optional[Union[float, Tensor]], X: Tensor
) -> Optional[Tensor]:
    r"""Expands a tensor representing bounds.

    Expand the dimension of bounds if necessary such that the last dimension of
    bounds is the same as the last dimension of `X`.

    Args:
        bounds: a bound (either upper or lower) of each column (last dimension)
            of `X`. If this is a single float, then all columns have the same bound.
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


def _get_extra_mll_args(
    mll: MarginalLogLikelihood
) -> Union[List[Tensor], List[List[Tensor]]]:
    r"""Obtain extra arguments for MarginalLogLikelihood objects.

    Get extra arguments (beyond the model output and training targets) required
    for the particular type of MarginalLogLikelihood for a forward pass.

    Args:
        mll: The MarginalLogLikelihood module.

    Returns:
        Extra arguments for the MarginalLogLikelihood.
    """
    if isinstance(mll, ExactMarginalLogLikelihood):
        return list(mll.model.train_inputs)
    elif isinstance(mll, SumMarginalLogLikelihood):
        return [list(x) for x in mll.model.train_inputs]
    elif isinstance(mll, VariationalELBO):
        return []
    else:
        raise ValueError("Do not know how to optimize MLL type.")


def _filter_kwargs(function: Callable, **kwargs: Any) -> Any:
    r"""Filter out kwargs that are not applicable for a given function.
    Return a copy of given kwargs dict with only the required kwargs."""
    return {k: v for k, v in kwargs.items() if k in signature(function).parameters}
