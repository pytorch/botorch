#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Cross-validation utilities using batch evaluation mode.
"""

from __future__ import annotations

from typing import Any, Dict, NamedTuple, Optional, Type

import torch
from botorch.fit import fit_gpytorch_model
from botorch.models.gpytorch import GPyTorchModel
from botorch.optim.utils import _filter_kwargs
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.mlls.marginal_log_likelihood import MarginalLogLikelihood
from torch import Tensor


class CVFolds(NamedTuple):
    train_X: Tensor
    test_X: Tensor
    train_Y: Tensor
    test_Y: Tensor
    train_Yvar: Optional[Tensor] = None
    test_Yvar: Optional[Tensor] = None


class CVResults(NamedTuple):
    model: GPyTorchModel
    posterior: GPyTorchPosterior
    observed_Y: Tensor
    observed_Yvar: Optional[Tensor] = None


def gen_loo_cv_folds(
    train_X: Tensor, train_Y: Tensor, train_Yvar: Optional[Tensor] = None
) -> CVFolds:
    r"""Generate LOO CV folds w.r.t. to `n`.

    Args:
        train_X: A `n x d` or `batch_shape x n x d` (batch mode) tensor of training
            features.
        train_Y: A `n x (m)` or `batch_shape x n x (m)` (batch mode) tensor of
            training observations.
        train_Yvar: A `batch_shape x n x (m)` or `batch_shape x n x (m)`
            (batch mode) tensor of observed measurement noise.

    Returns:
        CVFolds tuple with the following fields

        - train_X: A `n x (n-1) x d` or `batch_shape x n x (n-1) x d` tensor of
          training features.
        - test_X: A `n x 1 x d` or `batch_shape x n x 1 x d` tensor of test features.
        - train_Y: A `n x (n-1) x m` or `batch_shape x n x (n-1) x m` tensor of
          training observations.
        - test_Y: A `n x 1 x m` or `batch_shape x n x 1 x m` tensor of test
          observations.
        - train_Yvar: A `n x (n-1) x m` or `batch_shape x n x (n-1) x m` tensor
          of observed measurement noise.
        - test_Yvar: A `n x 1 x m` or `batch_shape x n x 1 x m` tensor of observed
          measurement noise.

    Example:
        >>> train_X = torch.rand(10, 1)
        >>> train_Y = torch.sin(6 * train_X) + 0.2 * torch.rand_like(train_X)
        >>> cv_folds = gen_loo_cv_folds(train_X, train_Y)
    """
    masks = torch.eye(train_X.shape[-2], dtype=torch.uint8, device=train_X.device)
    masks = masks.to(dtype=torch.bool)
    if train_Y.dim() < train_X.dim():
        # add output dimension
        train_Y = train_Y.unsqueeze(-1)
        if train_Yvar is not None:
            train_Yvar = train_Yvar.unsqueeze(-1)
    train_X_cv = torch.cat(
        [train_X[..., ~m, :].unsqueeze(dim=-3) for m in masks], dim=-3
    )
    test_X_cv = torch.cat([train_X[..., m, :].unsqueeze(dim=-3) for m in masks], dim=-3)
    train_Y_cv = torch.cat(
        [train_Y[..., ~m, :].unsqueeze(dim=-3) for m in masks], dim=-3
    )
    test_Y_cv = torch.cat([train_Y[..., m, :].unsqueeze(dim=-3) for m in masks], dim=-3)
    if train_Yvar is None:
        train_Yvar_cv = None
        test_Yvar_cv = None
    else:
        train_Yvar_cv = torch.cat(
            [train_Yvar[..., ~m, :].unsqueeze(dim=-3) for m in masks], dim=-3
        )
        test_Yvar_cv = torch.cat(
            [train_Yvar[..., m, :].unsqueeze(dim=-3) for m in masks], dim=-3
        )
    return CVFolds(
        train_X=train_X_cv,
        test_X=test_X_cv,
        train_Y=train_Y_cv,
        test_Y=test_Y_cv,
        train_Yvar=train_Yvar_cv,
        test_Yvar=test_Yvar_cv,
    )


def batch_cross_validation(
    model_cls: Type[GPyTorchModel],
    mll_cls: Type[MarginalLogLikelihood],
    cv_folds: CVFolds,
    fit_args: Optional[Dict[str, Any]] = None,
    observation_noise: bool = False,
) -> CVResults:
    r"""Perform cross validation by using gpytorch batch mode.

    Args:
        model_cls: A GPyTorchModel class. This class must initialize the likelihood
            internally. Note: Multi-task GPs are not currently supported.
        mll_cls: A MarginalLogLikelihood class.
        cv_folds: A CVFolds tuple.
        fit_args: Arguments passed along to fit_gpytorch_model

    Returns:
        A CVResults tuple with the following fields

        - model: GPyTorchModel for batched cross validation
        - posterior: GPyTorchPosterior where the mean has shape `n x 1 x m` or
          `batch_shape x n x 1 x m`
        - observed_Y: A `n x 1 x m` or `batch_shape x n x 1 x m` tensor of observations.
        - observed_Yvar: A `n x 1 x m` or `batch_shape x n x 1 x m` tensor of observed
          measurement noise.

    Example:
        >>> train_X = torch.rand(10, 1)
        >>> train_Y = torch.sin(6 * train_X) + 0.2 * torch.rand_like(train_X)
        >>> cv_folds = gen_loo_cv_folds(train_X, train_Y)
        >>> cv_results = batch_cross_validation(
        >>>     SingleTaskGP,
        >>>     ExactMarginalLogLikelihood,
        >>>     cv_folds,
        >>> )

    WARNING: This function is currently very memory inefficient, use it only
        for problems of small size.
    """
    fit_args = fit_args or {}
    kwargs = {
        "train_X": cv_folds.train_X,
        "train_Y": cv_folds.train_Y,
        "train_Yvar": cv_folds.train_Yvar,
    }
    model_cv = model_cls(**_filter_kwargs(model_cls, **kwargs))
    mll_cv = mll_cls(model_cv.likelihood, model_cv)
    mll_cv.to(cv_folds.train_X)
    mll_cv = fit_gpytorch_model(mll_cv, **fit_args)

    # Evaluate on the hold-out set in batch mode
    with torch.no_grad():
        posterior = model_cv.posterior(
            cv_folds.test_X, observation_noise=observation_noise
        )

    return CVResults(
        model=model_cv,
        posterior=posterior,
        observed_Y=cv_folds.test_Y,
        observed_Yvar=cv_folds.test_Yvar,
    )
