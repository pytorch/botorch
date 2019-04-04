#!/usr/bin/env python3

from typing import Any, Callable

from gpytorch.mlls.marginal_log_likelihood import MarginalLogLikelihood

from .optim.fit import fit_gpytorch_scipy


def fit_gpytorch_model(
    mll: MarginalLogLikelihood, optimizer: Callable = fit_gpytorch_scipy, **kwargs: Any
) -> MarginalLogLikelihood:
    """Fit hyperparameters of a gpytorch model.

    Optimizer functions are in botorch.optim.fit.

    Args:
        mll: MarginalLogLikelihood to be maximized.
        optimizer: Optimizer function.
        kwargs: Passed along to optimizer function.

    Returns:
        mll with optimized parameters.
    """
    mll.train()
    mll, _ = optimizer(mll, track_iterations=False, **kwargs)
    mll.eval()
    return mll
