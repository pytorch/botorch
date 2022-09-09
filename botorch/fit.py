#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
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
from typing import Any, Callable, Union

from botorch.exceptions.errors import UnsupportedError
from botorch.exceptions.warnings import BotorchWarning, OptimizationWarning
from botorch.models.converter import batched_to_model_list, model_list_to_batched
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.fully_bayesian_multitask import SaasFullyBayesianMultiTaskGP

from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel
from botorch.optim.fit import fit_gpytorch_scipy
from botorch.optim.utils import sample_all_priors
from botorch.settings import debug
from gpytorch.mlls.marginal_log_likelihood import MarginalLogLikelihood
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from linear_operator.utils.errors import NotPSDError
from pyro.infer.mcmc import MCMC, NUTS


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
        with warnings.catch_warnings(record=True) as ws, debug(True):
            # Make sure we catch all OptimizationWarnings.
            warnings.simplefilter("always", category=OptimizationWarning)
            if retry > 0:  # use normal initial conditions on first try
                mll.model.load_state_dict(original_state_dict)
                sample_all_priors(mll.model)
            try:
                mll, _ = optimizer(mll, track_iterations=False, **kwargs)
            except NotPSDError:
                retry += 1
                logging.log(
                    logging.DEBUG,
                    f"Fitting failed on try {retry} due to a NotPSDError.",
                )
                continue
        has_optwarning = False
        for w in ws:
            # Do not count reaching `maxiter` as an optimization failure.
            if "ITERATIONS REACHED LIMIT" in str(w.message):
                logging.log(
                    logging.DEBUG,
                    "Fitting ended early due to reaching the iteration limit.",
                )
                continue
            has_optwarning |= issubclass(w.category, OptimizationWarning)
            warnings.warn(w.message, w.category)
        if not has_optwarning:
            mll.eval()
            return mll
        retry += 1
        logging.log(logging.DEBUG, f"Fitting failed on try {retry}.")

    warnings.warn("Fitting failed on all retries.", RuntimeWarning)
    return mll.eval()


def fit_fully_bayesian_model_nuts(
    model: Union[SaasFullyBayesianSingleTaskGP, SaasFullyBayesianMultiTaskGP],
    max_tree_depth: int = 6,
    warmup_steps: int = 512,
    num_samples: int = 256,
    thinning: int = 16,
    disable_progbar: bool = False,
) -> None:
    r"""Fit a fully Bayesian model using the No-U-Turn-Sampler (NUTS)


    Args:
        model: SaasFullyBayesianSingleTaskGP to be fitted.
        max_tree_depth: Maximum tree depth for NUTS
        warmup_steps: The number of burn-in steps for NUTS.
        num_samples:  The number of MCMC samples. Note that with thinning,
            num_samples / thinning samples are retained.
        thinning: The amount of thinning. Every nth sample is retained.
        disable_progbar: A boolean indicating whether to print the progress
            bar and diagnostics during MCMC.

    Example:
        >>> gp = SaasFullyBayesianSingleTaskGP(train_X, train_Y)
        >>> fit_fully_bayesian_model_nuts(gp)
    """
    model.train()

    # Do inference with NUTS
    nuts = NUTS(
        model.pyro_model.sample,
        jit_compile=True,
        full_mass=True,
        ignore_jit_warnings=True,
        max_tree_depth=max_tree_depth,
    )
    mcmc = MCMC(
        nuts,
        warmup_steps=warmup_steps,
        num_samples=num_samples,
        disable_progbar=disable_progbar,
    )
    mcmc.run()

    # Get final MCMC samples from the Pyro model
    mcmc_samples = model.pyro_model.postprocess_mcmc_samples(
        mcmc_samples=mcmc.get_samples()
    )
    for k, v in mcmc_samples.items():
        mcmc_samples[k] = v[::thinning]

    # Load the MCMC samples back into the BoTorch model
    model.load_mcmc_samples(mcmc_samples)
    model.eval()
