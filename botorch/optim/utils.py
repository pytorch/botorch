#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

r"""
Utilities for optimization.
"""

import warnings
from inspect import signature
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.mlls.marginal_log_likelihood import MarginalLogLikelihood
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from gpytorch.mlls.variational_elbo import VariationalELBO
from torch import Tensor

from ..exceptions.errors import BotorchError
from ..exceptions.warnings import BotorchWarning
from ..models.gpytorch import GPyTorchModel


def sample_all_priors(model: GPyTorchModel) -> None:
    r"""Sample from hyperparameter priors (in-place).

    Args:
        model: A GPyTorchModel.
    """
    for _, prior, closure, setting_closure in model.named_priors():
        if setting_closure is None:
            raise RuntimeError(
                "Must provide inverse transform to be able to sample from prior."
            )
        try:
            setting_closure(prior.sample(closure().shape))
        except NotImplementedError:
            warnings.warn(
                f"`rsample` not implemented for {type(prior)}. Skipping.",
                BotorchWarning,
            )


class ConvergenceCriterion:
    r"""Basic class for evaluating optimization convergence.
    """

    def __init__(
        self,
        maxiter: int = 15000,
        ftol: float = 2.220446049250313e-09,
        minimize: bool = True,
    ) -> None:
        r"""Constructor for ConvergenceCriterion.

        Args:
            maxiter: maximum number of iterations.
            ftol: Function value relative tolerance for termination.
            minimize: boolean indicating the optimization direction.
        """
        # by default, use the scipy defaults
        self.maxiter = maxiter
        self.ftol = ftol
        self.minimize = minimize
        self.prev_fvals = None
        self.iter = 0

    def evaluate(self, fvals: Tensor) -> bool:
        r"""Evaluate convergence criterion.

        Args:
            fvals: tensor containing function values for the current iteration.
                If `fvals` contains more than one element, then the relative
                tolerance criterion is evaluated element-wise and True is returned
                if all elements have converged.

        TODO: add support for utilizing gradient information

        Returns:
            bool: convergence indicator
        """
        self.iter += 1
        if self.iter == self.maxiter:
            return True
        elif self.prev_fvals is not None:
            fmax = torch.stack(
                [self.prev_fvals.abs(), fvals.abs(), torch.ones_like(fvals)], dim=0
            ).max(dim=0)[0]
            f_delta = (self.prev_fvals - fvals).div(fmax)
            if not self.minimize:
                f_delta *= -1
            if torch.all(f_delta <= self.ftol):
                return True
        self.prev_fvals = fvals
        return False


def columnwise_clamp(
    X: Tensor,
    lower: Optional[Union[float, Tensor]] = None,
    upper: Optional[Union[float, Tensor]] = None,
    raise_on_violation: bool = False,
) -> Tensor:
    r"""Clamp values of a Tensor in column-wise fashion (with support for t-batches).

    This function is useful in conjunction with optimizers from the torch.optim
    package, which don't natively handle constraints. If you apply this after
    a gradient step you can be fancy and call it "projected gradient descent".
    This funtion is also useful for post-processing candidates generated by the
    scipy optimizer that satisfy bounds only up to numerical accuracy.

    Args:
        X: The `b x n x d` input tensor. If 2-dimensional, `b` is assumed to be 1.
        lower: The column-wise lower bounds. If scalar, apply bound to all columns.
        upper: The column-wise upper bounds. If scalar, apply bound to all columns.
        raise_on_violation: If `True`, raise an exception when the elments in `X`
            are out of the specified bounds (up to numerical accuracy). This is
            useful for post-processing candidates generated by optimizers that
            satisfy imposed bounds only up to numerical accuracy.

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
    if raise_on_violation and not torch.allclose(Xout, X):
        raise BotorchError("Original value(s) are out of bounds.")
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
        bounds is not None, and None if bounds is None.
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
