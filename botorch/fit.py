#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Utilities for model fitting.
"""

from __future__ import annotations

import logging
import warnings
from copy import deepcopy
from typing import Any, Callable

from botorch.exceptions.errors import BotorchError, UnsupportedError
from botorch.exceptions.warnings import BotorchWarning, OptimizationWarning
from botorch.models.converter import batched_to_model_list, model_list_to_batched
from botorch.models.gp_regression import HeteroskedasticSingleTaskGP
from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel
from botorch.optim.fit import fit_gpytorch_scipy
from botorch.optim.utils import sample_all_priors
from gpytorch.mlls.marginal_log_likelihood import MarginalLogLikelihood
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood


FAILED_CONVERSION_MSG = (
    "Failed to convert ModelList to batched model. "
    "Performing joint instead of sequential fitting."
)


def fit_gpytorch_model(
    mll: MarginalLogLikelihood, optimizer: Callable = fit_gpytorch_scipy, **kwargs: Any
) -> MarginalLogLikelihood:
    r"""Fit hyperparameters of a GPyTorch model.

    On optimizer failures, a new initial condition is sampled from the
    hyperparameter priors and optimization is retried. The maximum number of
    retries can be passed in as a `max_retries` kwarg (default is 5).

    Optimizer functions are in botorch.optim.fit.

    Args:
        mll: MarginalLogLikelihood to be maximized.
        optimizer: The optimizer function.
        kwargs: Arguments passed along to the optimizer function, including
            `max_retries` and `sequential` (controls the fitting of `ModelListGP`
            and `BatchedMultiOutputGPyTorchModel` models) or `approx_mll`
            (whether to use gpytorch's approximate MLL computation).

    Returns:
        MarginalLogLikelihood with optimized parameters.

    Example:
        >>> gp = SingleTaskGP(train_X, train_Y)
        >>> mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        >>> fit_gpytorch_model(mll)
    """
    sequential = kwargs.pop("sequential", True)
    max_retries = kwargs.pop("max_retries", 5)
    if isinstance(mll, SumMarginalLogLikelihood) and sequential:
        for mll_ in mll.mlls:
            fit_gpytorch_model(
                mll=mll_, optimizer=optimizer, max_retries=max_retries, **kwargs
            )
        return mll
    elif (
        isinstance(mll.model, BatchedMultiOutputGPyTorchModel)
        and mll.model._num_outputs > 1
        and sequential
    ):
        tf = None
        try:  # check if backwards-conversion is possible
            # remove the outcome transform since the training targets are already
            # transformed and the outcome transform cannot currently be split.
            # TODO: support splitting outcome transforms.
            if hasattr(mll.model, "outcome_transform"):
                tf = mll.model.outcome_transform
                mll.model.outcome_transform = None
            model_list = batched_to_model_list(mll.model)
            model_ = model_list_to_batched(model_list)
            mll_ = SumMarginalLogLikelihood(model_list.likelihood, model_list)
            fit_gpytorch_model(
                mll=mll_,
                optimizer=optimizer,
                sequential=True,
                max_retries=max_retries,
                **kwargs,
            )
            model_ = model_list_to_batched(mll_.model)
            mll.model.load_state_dict(model_.state_dict())
            # setting the transformed inputs is necessary because gpytorch
            # stores the raw training inputs on the ExactGP in the
            # ExactGP.__init__ call. At evaluation time, the test inputs will
            # already be in the transformed space if some transforms have
            # transform_on_eval set to False. ExactGP.__call__ will
            # concatenate the test points with the training inputs. Therefore,
            # it is important to set the ExactGP's train_inputs to also be
            # transformed data using all transforms (including the transforms
            # with transform_on_train set to True).
            mll.train()
            _set_transformed_inputs(mll=mll)
            if tf is not None:
                mll.model.outcome_transform = tf
            return mll.eval()
        # NotImplementedError is omitted since it derives from RuntimeError
        except (UnsupportedError, RuntimeError, AttributeError):
            warnings.warn(FAILED_CONVERSION_MSG, BotorchWarning)
            if tf is not None:
                mll.model.outcome_transform = tf
            return fit_gpytorch_model(
                mll=mll, optimizer=optimizer, sequential=False, max_retries=max_retries
            )
    # retry with random samples from the priors upon failure
    mll.train()
    original_state_dict = deepcopy(mll.model.state_dict())
    retry = 0
    while retry < max_retries:
        with warnings.catch_warnings(record=True) as ws:
            if retry > 0:  # use normal initial conditions on first try
                mll.model.load_state_dict(original_state_dict)
                sample_all_priors(mll.model)
            mll, _ = optimizer(mll, track_iterations=False, **kwargs)
            if not any(issubclass(w.category, OptimizationWarning) for w in ws):
                _set_transformed_inputs(mll=mll)
                mll.eval()
                return mll
            retry += 1
            logging.log(logging.DEBUG, f"Fitting failed on try {retry}.")

    warnings.warn("Fitting failed on all retries.", OptimizationWarning)
    return mll.eval()


def _set_transformed_inputs(mll: MarginalLogLikelihood) -> None:
    r"""Update training inputs with transformed inputs.

    Args:
        mll: The marginal likelihood.
    """
    models = (
        mll.model.models if isinstance(mll, SumMarginalLogLikelihood) else [mll.model]
    )
    for m in models:
        if hasattr(m, "input_transform"):
            X_tf = m.input_transform.preprocess_transform(m.train_inputs[0])
            if not hasattr(m, "set_train_data"):
                raise BotorchError(
                    "fit_gpytorch_model requires that a model has a set_train_data "
                    "method when an input_transform is used."
                )
            m.set_train_data(X_tf, strict=False)
            # TODO: override set_train_data in HeteroskedasticSingleTaskGP to do this
            # automatically
            if isinstance(m, HeteroskedasticSingleTaskGP):
                m.likelihood.noise_covar.noise_model.set_train_data(X_tf, strict=False)
