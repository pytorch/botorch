#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

r"""
Utilities for model fitting.
"""

import logging
import warnings
from copy import deepcopy
from typing import Any, Callable

from gpytorch.mlls.marginal_log_likelihood import MarginalLogLikelihood
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood

from .exceptions.warnings import OptimizationWarning
from .optim.fit import fit_gpytorch_scipy
from .optim.utils import sample_all_priors


def fit_gpytorch_model(
    mll: MarginalLogLikelihood, optimizer: Callable = fit_gpytorch_scipy, **kwargs: Any
) -> MarginalLogLikelihood:
    r"""Fit hyperparameters of a gpytorch model. On optimizer failures, a new
    initial condition is sampled from the hyperparameter priors and optimization
    is retried. The maximum number of retries can be passed in as a `max_retries`
    kwarg (default is 5).

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
    sequential = kwargs.pop("sequential", True)
    if isinstance(mll, SumMarginalLogLikelihood) and sequential:
        for mll_ in mll.mlls:
            fit_gpytorch_model(mll=mll_, optimizer=optimizer, **kwargs)
        return mll
    max_retries = kwargs.pop("max_retries", 5)
    original_state_dict = deepcopy(mll.model.state_dict())
    mll.train()
    retry = 0
    while retry < max_retries:
        with warnings.catch_warnings(record=True) as ws:
            if retry > 0:  # use normal initial conditions on first try
                mll.model.load_state_dict(original_state_dict)
                sample_all_priors(mll.model)
            mll, _ = optimizer(mll, track_iterations=False, **kwargs)
            if not any(issubclass(w.category, OptimizationWarning) for w in ws):
                mll.eval()
                return mll
            retry += 1
            logging.warning(f"Fitting failed on try {retry}.")

    warnings.warn("Fitting failed on all retries.", OptimizationWarning)
    mll.eval()
    return mll
