#!/usr/bin/env python3

r"""
Utilities for model fitting.
"""

from typing import Any, Callable

from gpytorch.mlls.marginal_log_likelihood import MarginalLogLikelihood

from .optim.fit import fit_gpytorch_scipy


def fit_gpytorch_model(
    mll: MarginalLogLikelihood, optimizer: Callable = fit_gpytorch_scipy, **kwargs: Any
) -> MarginalLogLikelihood:
    r"""Fit hyperparameters of a gpytorch model.

    Optimizer functions are in botorch.optim.fit.

    Args:
        mll: MarginalLogLikelihood to be maximized.
        optimizer: The optimizer function.
        kwargs: Arguments passed along to the optimizer function.

    Returns:
        MarginalLogLikelihood with optimized parameters.

    Example:
        >>> gp = SingleTaskGP(train_X, train_Y)
        >>> mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        >>> fit_gpytorch_model(mll)
    """
    mll.train()
    mll, _ = optimizer(mll, track_iterations=False, **kwargs)
    mll.eval()
    return mll
