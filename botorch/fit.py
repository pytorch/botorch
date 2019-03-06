#!/usr/bin/env python3

from typing import Callable

from gpytorch.mlls.marginal_log_likelihood import MarginalLogLikelihood

from .optim.fit import fit_scipy


def fit_model(
    mll: MarginalLogLikelihood, optimizer: Callable = fit_scipy, **kwargs
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
    mll.model.train()
    mll.likelihood.train()

    mll, _ = optimizer(mll, track_iterations=False, **kwargs)

    mll.model.eval()
    mll.likelihood.eval()
    return mll
