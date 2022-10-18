#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Model fitting routines."""

from __future__ import annotations

import logging
from contextlib import nullcontext
from re import compile, Pattern
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Type, Union
from warnings import catch_warnings, simplefilter, warn, WarningMessage

from botorch.exceptions.errors import ModelFittingError, UnsupportedError
from botorch.exceptions.warnings import BotorchWarning, OptimizationWarning
from botorch.models.converter import batched_to_model_list, model_list_to_batched
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.fully_bayesian_multitask import SaasFullyBayesianMultiTaskGP

from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.optim.fit import fit_gpytorch_scipy
from botorch.optim.utils import (
    allclose_mll,
    del_attribute_ctx,
    parameter_rollback_ctx,
    requires_grad_ctx,
    sample_all_priors,
    state_rollback_ctx,
    Tkwargs,
)
from botorch.settings import debug
from botorch.utils.dispatcher import Dispatcher, MDNotImplementedError
from gpytorch.likelihoods import Likelihood
from gpytorch.mlls.marginal_log_likelihood import MarginalLogLikelihood
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from linear_operator.utils.errors import NotPSDError
from pyro.infer.mcmc import MCMC, NUTS
from torch import device, mean, Tensor

OptimizerType = Callable[[MarginalLogLikelihood], Tuple[MarginalLogLikelihood, Any]]
DEFAULT_LOGGING_PATTERNS: Dict[int, Pattern] = {
    logging.DEBUG: compile(  # catch warning corresponding to `maxiter` and `maxfun`
        "TOTAL NO. of (ITERATIONS REACHED LIMIT|f AND g EVALUATIONS EXCEEDS LIMIT)"
    )
}


def DEFAULT_WARNING_FILTER(
    w: WarningMessage,
    logging_patterns: Dict[int, Pattern] = DEFAULT_LOGGING_PATTERNS,
) -> bool:
    r"""Default warning resolution policy: retry upon encountering an
    OptimizationWarning that does not match any logging pattern.

    Args:
        w: Candidate for filtering.
        logging_patterns: Dictionary mapping logging levels to regular expressions.
            Warning messages are compared against these expressions and matches are
            awarded first-come-first-serve when iterating through the dictionary.

    Returns:
        Boolean indicating whether the warning is unresolved.
    """
    for level, pattern in logging_patterns.items():
        if pattern.search(str(w.message)):
            logging.log(level, w.message)
            return False

    # Rethrow OptimizationWarnings but mark them as resolved
    if not issubclass(w.category, OptimizationWarning):
        warn(w.message, w.category)
        return False

    return True


# Dispatcher for `fit_gpytorch_mll`
def _type_bypassing_encoder(arg: Any) -> Type:
    # Allow type variables to be passed as pre-encoded arguments
    return arg if isinstance(arg, type) else type(arg)


dispatcher = Dispatcher("fit_gpytorch_mll", encoder=_type_bypassing_encoder)


def fit_gpytorch_mll(
    mll: MarginalLogLikelihood,
    optimizer: Optional[Callable] = None,
    optimizer_kwargs: Optional[dict] = None,
    **kwargs: Any,
) -> MarginalLogLikelihood:
    r"""Clearing house for fitting models passed as GPyTorch MarginalLogLikelihoods.

    Args:
        mll: A GPyTorch MarginalLogLikelihood instance.
        optimizer: User specified optimization algorithm. When `optimizer is None`,
            this keyword argument is omitted when calling the dispatcher.
        optimizer_kwargs: A dictionary of keyword arguments passed when
            calling `optimizer`.
        **kwargs: Keyword arguments passed down through the dispatcher to
            fit subroutines. Unexpected keywords are ignored.

    Returns:
        The `mll` instance. If fitting succeeded, then `mll` will be in evaluation mode,
        i.e. `mll.training == False`. Otherwise, `mll` will be in training mode.
    """
    if optimizer is not None:  # defer to per-method defaults
        kwargs["optimizer"] = optimizer

    return dispatcher(
        mll,
        type(mll.likelihood),
        type(mll.model),
        optimizer_kwargs=optimizer_kwargs,
        **kwargs,
    )


def fit_gpytorch_model(
    mll: MarginalLogLikelihood,
    optimizer: Optional[OptimizerType] = None,
    optimizer_kwargs: Optional[dict] = None,
    exclude: Optional[Iterable[str]] = None,
    max_retries: Optional[int] = None,
    **kwargs: Any,
) -> MarginalLogLikelihood:
    r"""Convenience method for fitting GPyTorch models using legacy API. For more
    details, see `fit_gpytorch_mll`.

    Args:
        mll: A GPyTorch MarginalLogLikelihood instance.
        optimizer: User specified optimization algorithm. When `optimizer is None`,
            this keyword argument is omitted when calling the dispatcher from inside
            `fit_gpytorch_mll`.
        exclude: Legacy argument for specifying parameters `x` that should be held fixed
            during optimization. Internally, used to temporarily set `x.requires_grad`
            to False.
        max_retries: Legacy name for `max_attempts`. When `max_retries is None`,
            this keyword argument is omitted when calling `fit_gpytorch_mll`.
    """
    warn(
        "`fit_gpytorch_model` is marked for deprecation, consider using "
        "`fit_gpytorch_mll` instead.",
        DeprecationWarning,
    )
    if max_retries is not None:
        kwargs["max_attempts"] = max_retries

    optimizer_kwargs = {} if optimizer_kwargs is None else optimizer_kwargs
    for key in ("bounds", "options", "track_iterations", "approx_mll"):
        if key not in kwargs:
            continue

        val = kwargs.pop(key)
        if key in optimizer_kwargs and val is not optimizer_kwargs[key]:
            raise SyntaxError(f"keyword argument repeated: {key}")

        optimizer_kwargs[key] = val

    with (
        nullcontext()
        if exclude is None
        else requires_grad_ctx(mll, assignments={name: False for name in exclude})
    ):
        try:
            mll = fit_gpytorch_mll(
                mll,
                optimizer=optimizer,
                optimizer_kwargs=optimizer_kwargs,
                **kwargs,
            )
        except ModelFittingError as err:
            warn(str(err), RuntimeWarning)

    return mll


@dispatcher.register(MarginalLogLikelihood, object, object)
def _fit_fallback(
    mll: MarginalLogLikelihood,
    _: Type[object],
    __: Type[object],
    *,
    optimizer: Optional[Callable] = fit_gpytorch_scipy,
    optimizer_kwargs: Optional[dict] = None,
    max_attempts: int = 5,
    warning_filter: Callable[[WarningMessage], bool] = DEFAULT_WARNING_FILTER,
    caught_exception_types: Tuple[Type[BaseException], ...] = (NotPSDError,),
    **ignore: Any,
) -> MarginalLogLikelihood:
    r"""Generic fallback method for fitting Gaussian processes.

    Attempts to fit a model using the provided optimizer, then determines whether or
    not to retry by evaluating a given policy on emitted warning messages. The first
    attempt is run using the initialized parameter values; subsequent attempts begin
    by resampling tunable parameters.

    Args:
        optimizer: The underlying optimization algorithm to run.
        optimizer_kwargs: Keyword arguments passed when calling `optimizer`.
        max_attempts: The maximum number of fit attempts allowed. The attempt budget
            is NOT shared between calls to this method.
        warning_filter: A function used to filter warnings produced when calling
            `optimizer`. Any unfiltered warnings will be rethrown and trigger a
            model fitting retry.
        caught_exception_types: A tuple of exception types whose instances should
            be redirected to `logging.DEBUG`.
        **ignore: This function ignores unrecognized keyword arguments.

    Returns:
        The `mll` instance. If fitting succeeded, then `mll` will be in evaluation mode,
        i.e. `mll.training == False`. Otherwise, `mll` will be in training mode.
    """
    ckpt: Dict[str, Tuple[Tensor, Tkwargs]] = None  # lazy CPU-based checkpoint
    ckpt_nograd: Dict[str, Tuple[Tensor, Tkwargs]] = None  # subset for fixed parameters
    optimizer_kwargs = {} if optimizer_kwargs is None else optimizer_kwargs

    mll.train()
    for attempt in range(1, 1 + max_attempts):
        # Wrap with rollback contextmanager so each loop iteration reloads the original
        # state_dict upon exiting (unless `ckpt` is cleared).
        with state_rollback_ctx(mll, checkpoint=ckpt, device=device("cpu")) as ckpt:
            if ckpt_nograd is None:
                ckpt_nograd = {  # reuse cached values from primary checkpoint
                    k: ckpt[k] for k, v in mll.named_parameters() if not v.requires_grad
                }

            if attempt > 1:  # maybe resample parameters that require gradients
                with parameter_rollback_ctx(mll, checkpoint=ckpt_nograd):
                    sample_all_priors(mll.model)

            try:
                # Fit the model
                with catch_warnings(record=True) as warning_list, debug(True):
                    simplefilter("always", category=OptimizationWarning)
                    mll, _ = optimizer(mll, **optimizer_kwargs)

                # Resolve warning messages and determine whether or not to retry
                done = True
                for unresolved_warning in filter(warning_filter, warning_list):
                    warn(unresolved_warning.message, unresolved_warning.category)
                    done = False

                if done:
                    ckpt.clear()  # do not rollback upon exiting
                    return mll.eval()

                # Ensure mll is in the right mode if fitting failed
                mll = mll if mll.training else mll.train()
                logging.log(
                    logging.DEBUG,
                    f"Fit attempt #{attempt} of {max_attempts} triggered retry policy"
                    f"{'.' if attempt == max_attempts else '; retrying...'}",
                )

            except caught_exception_types as err:
                logging.log(
                    logging.DEBUG,
                    f"Fit attempt #{attempt} of {max_attempts} failed with exception: "
                    f"{err}",
                )

    raise ModelFittingError("All attempts to fit the model have failed.")


@dispatcher.register(SumMarginalLogLikelihood, Likelihood, ModelListGP)
def _fit_list(
    mll: SumMarginalLogLikelihood,
    _: Type[Likelihood],
    __: Type[ModelListGP],
    **kwargs: Any,
) -> SumMarginalLogLikelihood:
    r"""Fitting routine for lists of independent Gaussian processes.

    Args:
        **kwargs: Passed to each of `mll.mlls`.

    Returns:
        The `mll` instance. If fitting succeeded for all of `mll.mlls`, then `mll` will
        be in evaluation mode, i.e. `mll.training == False`. Otherwise, `mll` will be in
        training mode.
    """
    mll.train()
    for sub_mll in mll.mlls:
        fit_gpytorch_mll(sub_mll, **kwargs)

    return mll.eval() if not any(sub_mll.training for sub_mll in mll.mlls) else mll


@dispatcher.register(MarginalLogLikelihood, Likelihood, BatchedMultiOutputGPyTorchModel)
def _fit_multioutput_independent(
    mll: MarginalLogLikelihood,
    _: Type[Likelihood],
    __: Type[BatchedMultiOutputGPyTorchModel],
    *,
    sequential: bool = True,
    **kwargs: Any,
) -> MarginalLogLikelihood:
    r"""Fitting routine for multioutput Gaussian processes.

    Args:
        sequential: Boolean specifying whether or not to an attempt should be made to
            fit the model as a collection of independent GPs. Only relevant for
            certain types of GPs with independent outputs, see `batched_to_model_list`.
        **kwargs: Passed to the next method unaltered.

    Returns:
        The `mll` instance. If fitting succeeded, then `mll` will be in evaluation mode,
        i.e. `mll.training == False`. Otherwise, `mll` will be in training mode.
    """
    if (  # incompatible models
        not sequential
        or mll.model.num_outputs == 1
        or mll.likelihood is not getattr(mll.model, "likelihood", None)
    ):
        raise MDNotImplementedError  # defer to generic

    # TODO: Unpacking of OutcomeTransforms not yet supported. Targets are often
    # pre-transformed in __init__, so try fitting with outcome_transform hidden
    mll.train()
    with del_attribute_ctx(mll.model, "outcome_transform"):
        try:
            # Attempt to unpack batched model into a list of independent submodels
            unpacked_model = batched_to_model_list(mll.model)
            unpacked_mll = SumMarginalLogLikelihood(  # avg. over MLLs internally
                unpacked_model.likelihood, unpacked_model
            )
            if not allclose_mll(a=mll, b=unpacked_mll, transform_a=mean):
                raise RuntimeError(  # validate model unpacking
                    "Training loss of unpacked model differs from that of the original."
                )

            # Fit submodels independently
            unpacked_mll = fit_gpytorch_mll(unpacked_mll, **kwargs)

            # Repackage submodels and copy over state_dict
            repacked_model = model_list_to_batched(unpacked_mll.model.train())
            repacked_mll = type(mll)(repacked_model.likelihood, repacked_model)
            with state_rollback_ctx(mll, device=device("cpu")) as ckpt:
                mll.load_state_dict(repacked_mll.state_dict())
                if not allclose_mll(a=mll, b=repacked_mll):
                    raise RuntimeError(  # validate model repacking
                        "Training loss of repacked model differs from that of the "
                        "original."
                    )
                ckpt.clear()  # do not rollback when exiting
                return mll.eval()  # DONE!

        except (AttributeError, RuntimeError, UnsupportedError) as err:
            msg = f"Failed to independently fit submodels with exception: {err}"
            warn(
                f"{msg.rstrip('.')}. Deferring to generic dispatch...",
                BotorchWarning,
            )
            raise MDNotImplementedError


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
