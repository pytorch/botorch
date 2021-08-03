#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
A registry of helpers for generating inputs to acquisition function
constructors programmatically from a consistent input format.
"""
import warnings
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.analytic import (
    ExpectedImprovement,
    PosteriorMean,
    ProbabilityOfImprovement,
    UpperConfidenceBound,
    ConstrainedExpectedImprovement,
    NoisyExpectedImprovement,
)
from botorch.acquisition.monte_carlo import (
    qExpectedImprovement,
    qNoisyExpectedImprovement,
    qProbabilityOfImprovement,
    qSimpleRegret,
    qUpperConfidenceBound,
)
from botorch.acquisition.multi_objective import (
    ExpectedHypervolumeImprovement,
    MCMultiOutputObjective,
    qExpectedHypervolumeImprovement,
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.objective import (
    AnalyticMultiOutputObjective,
    IdentityAnalyticMultiOutputObjective,
    IdentityMCMultiOutputObjective,
)
from botorch.acquisition.multi_objective.utils import get_default_partitioning_alpha
from botorch.acquisition.objective import (
    MCAcquisitionObjective,
    AcquisitionObjective,
    IdentityMCObjective,
    PosteriorTransform,
    ScalarizedPosteriorTransform,
)
from botorch.exceptions.errors import UnsupportedError
from botorch.models.model import Model
from botorch.sampling.samplers import IIDNormalSampler, MCSampler, SobolQMCNormalSampler
from botorch.utils.constraints import get_outcome_constraint_transforms
from botorch.utils.containers import TrainingData
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
    NondominatedPartitioning,
)
from torch import Tensor


ACQF_INPUT_CONSTRUCTOR_REGISTRY = {}


def get_acqf_input_constructor(
    acqf_cls: Type[AcquisitionFunction],
) -> Callable[..., Dict[str, Any]]:
    r"""Get acqusition function input constructor from registry.

    Args:
        acqf_cls: The AcquisitionFunction class (not instance) for which
            to retrieve the input constructor.

    Returns:
        The input constructor associated with `acqf_cls`.

    """
    if acqf_cls not in ACQF_INPUT_CONSTRUCTOR_REGISTRY:
        raise RuntimeError(
            f"Input constructor for acquisition class `{acqf_cls.__name__}` not "
            "registered. Use the `@acqf_input_constructor` decorator to register "
            "a new method."
        )
    return ACQF_INPUT_CONSTRUCTOR_REGISTRY[acqf_cls]


def acqf_input_constructor(
    *acqf_cls: Type[AcquisitionFunction],
) -> Callable[..., AcquisitionFunction]:
    r"""Decorator for registering acquisition function input constructors.

    Args:
        acqf_cls: The AcquisitionFunction classes (not instances) for which
            to register the input constructor.
    """
    for acqf_cls_ in acqf_cls:
        if acqf_cls_ in ACQF_INPUT_CONSTRUCTOR_REGISTRY:
            raise ValueError(
                "Cannot register duplicate arg constructor for acquisition "
                f"class `{acqf_cls_.__name__}`"
            )

    def decorator(method):
        for acqf_cls_ in acqf_cls:
            _register_acqf_input_constructor(
                acqf_cls=acqf_cls_, input_constructor=method
            )
            ACQF_INPUT_CONSTRUCTOR_REGISTRY[acqf_cls_] = method
        return method

    return decorator


def _register_acqf_input_constructor(
    acqf_cls: Type[AcquisitionFunction],
    input_constructor: Callable[..., Dict[str, Any]],
) -> None:
    ACQF_INPUT_CONSTRUCTOR_REGISTRY[acqf_cls] = input_constructor


# ------------------------- Deprecation Helpers ------------------------- #


def _deprecate_objective_arg(
    posterior_transform: Optional[PosteriorTransform] = None,
    objective: Optional[AcquisitionObjective] = None,
) -> Optional[PosteriorTransform]:
    if posterior_transform is not None:
        if objective is None:
            return posterior_transform
        else:
            raise RuntimeError(
                "Got both a non-MC objective (DEPRECATED) and a posterior "
                "transform. Use only a posterior transform instead."
            )
    elif objective is not None:
        warnings.warn(
            "The `objective` argument to AnalyticAcquisitionFunctions is deprecated "
            "and will be removed in the next version. Use `posterior_transform` "
            "instead.",
            DeprecationWarning,
        )
        return ScalarizedPosteriorTransform(
            weights=objective.weights, offset=objective.offset
        )
    else:
        return None


# --------------------- Input argument constructors --------------------- #


@acqf_input_constructor(PosteriorMean)
def construct_inputs_analytic_base(
    model: Model,
    training_data: TrainingData,
    posterior_transform: Optional[PosteriorTransform] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    r"""Construct kwargs for basic analytic acquisition functions.

    Args:
        model: The model to be used in the acquisition function.
        training_data: A TrainingData object contraining the model's
            training data. `best_f` is extracted from here.
        posterior_transform: The posterior transform to be used in the
            acquisition function.

    Returns:
        A dict mapping kwarg names of the constructor to values.
    """
    return {
        "model": model,
        "posterior_transform": _deprecate_objective_arg(
            posterior_transform=posterior_transform,
            objective=kwargs.get("objective", None),
        ),
    }


@acqf_input_constructor(ExpectedImprovement, ProbabilityOfImprovement)
def construct_inputs_best_f(
    model: Model,
    training_data: TrainingData,
    posterior_transform: Optional[PosteriorTransform] = None,
    maximize: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:
    r"""Construct kwargs for the acquisition functions requiring `best_f`.

    Args:
        model: The model to be used in the acquisition function.
        training_data: A TrainingData object contraining the model's
            training data. `best_f` is extracted from here.
        posterior_transform: The posterior transform to be used in the
            acquisition function.
        maximize: If True, consider the problem a maximization problem.

    Returns:
        A dict mapping kwarg names of the constructor to values.
    """
    base_inputs = construct_inputs_analytic_base(
        model=model,
        training_data=training_data,
        posterior_transform=posterior_transform,
        **kwargs,
    )
    best_f = kwargs.get(
        "best_f",
        get_best_f_analytic(
            training_data=training_data,
            objective=kwargs.get("objective", None),
            posterior_transform=posterior_transform,
        ),
    )
    return {**base_inputs, "best_f": best_f, "maximize": maximize}


@acqf_input_constructor(UpperConfidenceBound)
def construct_inputs_ucb(
    model: Model,
    training_data: TrainingData,
    posterior_transform: Optional[PosteriorTransform] = None,
    beta: Union[float, Tensor] = 0.2,
    maximize: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:
    r"""Construct kwargs for `UpperConfidenceBound`.

    Args:
        model: The model to be used in the acquisition function.
        training_data: A TrainingData object contraining the model's
            training data. `best_f` is extracted from here.
        posterior_transform: The posterior transform to be used in the
            acquisition function.
        beta: Either a scalar or a one-dim tensor with `b` elements (batch mode)
            representing the trade-off parameter between mean and covariance
        maximize: If True, consider the problem a maximization problem.

    Returns:
        A dict mapping kwarg names of the constructor to values.
    """
    base_inputs = construct_inputs_analytic_base(
        model=model,
        training_data=training_data,
        posterior_transform=posterior_transform,
        **kwargs,
    )
    return {**base_inputs, "beta": beta, "maximize": maximize}


@acqf_input_constructor(ConstrainedExpectedImprovement)
def construct_inputs_constrained_ei(
    model: Model,
    training_data: TrainingData,
    objective_index: int,
    constraints: Dict[int, Tuple[Optional[float], Optional[float]]],
    maximize: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:
    r"""Construct kwargs for `ConstrainedExpectedImprovement`.

    Args:
        model: The model to be used in the acquisition function.
        training_data: A TrainingData object contraining the model's
            training data. `best_f` is extracted from here.
        objective_index: The index of the objective.
        constraints: A dictionary of the form `{i: [lower, upper]}`, where
            `i` is the output index, and `lower` and `upper` are lower and upper
            bounds on that output (resp. interpreted as -Inf / Inf if None)
        maximize: If True, consider the problem a maximization problem.

    Returns:
        A dict mapping kwarg names of the constructor to values.
    """
    # TODO: Implement best point computation from training data
    # best_f =
    # return {
    #     "model": model,
    #     "best_f": best_f,
    #     "objective_index": objective_index,
    #     "constraints": constraints,
    #     "maximize": maximize,
    # }
    raise NotImplementedError  # pragma: nocover


@acqf_input_constructor(NoisyExpectedImprovement)
def construct_inputs_noisy_ei(
    model: Model,
    training_data: TrainingData,
    num_fantasies: int = 20,
    maximize: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:
    r"""Construct kwargs for `NoisyExpectedImprovement`.

    Args:
        model: The model to be used in the acquisition function.
        training_data: A TrainingData object contraining the model's
            training data. `best_f` is extracted from here.
        num_fantasies: The number of fantasies to generate. The higher this
            number the more accurate the model (at the expense of model
            complexity and performance).
        maximize: If True, consider the problem a maximization problem.

    Returns:
        A dict mapping kwarg names of the constructor to values.
    """
    # TODO: Add prune_baseline functionality as for qNEI
    if not training_data.is_block_design:
        raise NotImplementedError("Currently only block designs are supported")
    return {
        "model": model,
        "X_observed": training_data.X,
        "num_fantasies": num_fantasies,
        "maximize": maximize,
    }


@acqf_input_constructor(qSimpleRegret)
def construct_inputs_mc_base(
    model: Model,
    training_data: TrainingData,
    objective: Optional[MCAcquisitionObjective] = None,
    posterior_transform: Optional[PosteriorTransform] = None,
    X_pending: Optional[Tensor] = None,
    sampler: Optional[MCSampler] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    r"""Construct kwargs for basic MC acquisition functions.

    Args:
        model: The model to be used in the acquisition function.
        training_data: A TrainingData object contraining the model's
            training data. Used e.g. to extract inputs such as `best_f`
            for expected improvement acquisition functions.
        objective: The objective to be used in the acquisition function.
        posterior_transform: The posterior transform to be used in the
            acquisition function.
        X_pending: A `batch_shape, m x d`-dim Tensor of `m` design points
            that have points that have been submitted for function evaluation
            but have not yet been evaluated.
        sampler: The sampler used to draw base samples. If omitted, uses
            the acquisition functions's default sampler.

    Returns:
        A dict mapping kwarg names of the constructor to values.
    """
    return {
        "model": model,
        "objective": objective,
        "posterior_transform": posterior_transform,
        "X_pending": X_pending,
        "sampler": sampler,
    }


@acqf_input_constructor(qExpectedImprovement)
def construct_inputs_qEI(
    model: Model,
    training_data: TrainingData,
    objective: Optional[MCAcquisitionObjective] = None,
    posterior_transform: Optional[PosteriorTransform] = None,
    X_pending: Optional[Tensor] = None,
    sampler: Optional[MCSampler] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    r"""Construct kwargs for the `qExpectedImprovement` constructor.

    Args:
        model: The model to be used in the acquisition function.
        training_data: A TrainingData object contraining the model's
            training data. Used e.g. to extract inputs such as `best_f`
            for expected improvement acquisition functions.
        objective: The objective to be used in the acquisition function.
        posterior_transform: The posterior transform to be used in the
            acquisition function.
        X_pending: A `m x d`-dim Tensor of `m` design points that have been
            submitted for function evaluation but have not yet been evaluated.
            Concatenated into X upon forward call.
        sampler: The sampler used to draw base samples. If omitted, uses
            the acquisition functions's default sampler.

    Returns:
        A dict mapping kwarg names of the constructor to values.
    """
    base_inputs = construct_inputs_mc_base(
        model=model,
        training_data=training_data,
        objective=objective,
        posterior_transform=posterior_transform,
        sampler=sampler,
        X_pending=X_pending,
    )
    # TODO: Dedup handling of this here and in the constructor (maybe via a
    # shared classmethod doing this)
    best_f = kwargs.get(
        "best_f",
        get_best_f_mc(
            training_data=training_data,
            objective=objective,
            posterior_transform=posterior_transform,
        ),
    )
    return {**base_inputs, "best_f": best_f}


@acqf_input_constructor(qNoisyExpectedImprovement)
def construct_inputs_qNEI(
    model: Model,
    training_data: TrainingData,
    objective: Optional[MCAcquisitionObjective] = None,
    posterior_transform: Optional[PosteriorTransform] = None,
    X_pending: Optional[Tensor] = None,
    sampler: Optional[MCSampler] = None,
    X_baseline: Optional[Tensor] = None,
    prune_baseline: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    r"""Construct kwargs for the `qNoisyExpectedImprovement` constructor.

    Args:
        model: The model to be used in the acquisition function.
        training_data: A TrainingData object contraining the model's
            training data. Used e.g. to extract inputs such as `best_f`
            for expected improvement acquisition functions. Only block-
            design training data currently supported.
        objective: The objective to be used in the acquisition function.
        posterior_transform: The posterior transform to be used in the
            acquisition function.
        X_pending: A `m x d`-dim Tensor of `m` design points that have been
            submitted for function evaluation but have not yet been evaluated.
            Concatenated into X upon forward call.
        sampler: The sampler used to draw base samples. If omitted, uses
            the acquisition functions's default sampler.
        X_baseline: A `batch_shape x r x d`-dim Tensor of `r` design points
            that have already been observed. These points are considered as
            the potential best design point. If omitted, use `training_data.X`.
        prune_baseline: If True, remove points in `X_baseline` that are
            highly unlikely to be the best point. This can significantly
            improve performance and is generally recommended.

    Returns:
        A dict mapping kwarg names of the constructor to values.
    """
    base_inputs = construct_inputs_mc_base(
        model=model,
        training_data=training_data,
        objective=objective,
        posterior_transform=posterior_transform,
        sampler=sampler,
        X_pending=X_pending,
    )
    if X_baseline is None:
        if not training_data.is_block_design:
            raise NotImplementedError("Currently only block designs are supported.")
        X_baseline = training_data.X
    return {
        **base_inputs,
        "X_baseline": X_baseline,
        "prune_baseline": prune_baseline,
    }


@acqf_input_constructor(qProbabilityOfImprovement)
def construct_inputs_qPI(
    model: Model,
    training_data: TrainingData,
    objective: Optional[MCAcquisitionObjective] = None,
    posterior_transform: Optional[PosteriorTransform] = None,
    X_pending: Optional[Tensor] = None,
    sampler: Optional[MCSampler] = None,
    tau: float = 1e-3,
    **kwargs: Any,
) -> Dict[str, Any]:
    r"""Construct kwargs for the `qProbabilityOfImprovement` constructor.

    Args:
        model: The model to be used in the acquisition function.
        training_data: A TrainingData object contraining the model's
            training data. Used e.g. to extract inputs such as `best_f`
            for expected improvement acquisition functions.
        objective: The objective to be used in the acquisition function.
        posterior_transform: The posterior transform to be used in the
            acquisition function.
        X_pending: A `m x d`-dim Tensor of `m` design points that have been
            submitted for function evaluation but have not yet been evaluated.
            Concatenated into X upon forward call.
        sampler: The sampler used to draw base samples. If omitted, uses
            the acquisition functions's default sampler.
        tau: The temperature parameter used in the sigmoid approximation
            of the step function. Smaller values yield more accurate
            approximations of the function, but result in gradients
            estimates with higher variance.

    Returns:
        A dict mapping kwarg names of the constructor to values.
    """
    base_inputs = construct_inputs_mc_base(
        model=model,
        training_data=training_data,
        objective=objective,
        posterior_transform=posterior_transform,
        sampler=sampler,
        X_pending=X_pending,
    )
    return {
        **base_inputs,
        "tau": tau,
    }


@acqf_input_constructor(qUpperConfidenceBound)
def construct_inputs_qUCB(
    model: Model,
    training_data: TrainingData,
    objective: Optional[MCAcquisitionObjective] = None,
    posterior_transform: Optional[PosteriorTransform] = None,
    X_pending: Optional[Tensor] = None,
    sampler: Optional[MCSampler] = None,
    beta: float = 0.2,
    **kwargs: Any,
) -> Dict[str, Any]:
    r"""Construct kwargs for the `qUpperConfidenceBound` constructor.

    Args:
        model: The model to be used in the acquisition function.
        training_data: A TrainingData object contraining the model's
            training data. Used e.g. to extract inputs such as `best_f`
            for expected improvement acquisition functions.
        objective: The objective to be used in the acquisition function.
        posterior_transform: The posterior transform to be used in the
            acquisition function.
        X_pending: A `m x d`-dim Tensor of `m` design points that have been
            submitted for function evaluation but have not yet been evaluated.
            Concatenated into X upon forward call.
        sampler: The sampler used to draw base samples. If omitted, uses
            the acquisition functions's default sampler.
        beta: Controls tradeoff between mean and standard deviation in UCB.

    Returns:
        A dict mapping kwarg names of the constructor to values.
    """
    base_inputs = construct_inputs_mc_base(
        model=model,
        training_data=training_data,
        objective=objective,
        posterior_transform=posterior_transform,
        sampler=sampler,
        X_pending=X_pending,
    )
    return {**base_inputs, "beta": beta}


def _get_sampler(mc_samples: int, qmc: bool) -> MCSampler:
    """Set up MC sampler for q(N)EHVI."""
    # initialize the sampler
    seed = int(torch.randint(1, 10000, (1,)).item())
    if qmc:
        return SobolQMCNormalSampler(num_samples=mc_samples, seed=seed)
    return IIDNormalSampler(num_samples=mc_samples, seed=seed)


@acqf_input_constructor(ExpectedHypervolumeImprovement)
def construct_inputs_EHVI(
    model: Model,
    training_data: TrainingData,
    objective_thresholds: Tensor,
    objective: Optional[AnalyticMultiOutputObjective] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    r"""Construct kwargs for `ExpectedHypervolumeImprovement` constructor."""
    num_objectives = objective_thresholds.shape[0]
    if kwargs.get("outcome_constraints") is not None:
        raise NotImplementedError("EHVI does not yet support outcome constraints.")

    X_observed = training_data.X
    alpha = kwargs.get(
        "alpha",
        get_default_partitioning_alpha(num_objectives=num_objectives),
    )
    # This selects the objectives (a subset of the outcomes) and set each
    # objective threhsold to have the proper optimization direction.
    if objective is None:
        objective = IdentityAnalyticMultiOutputObjective()
    ref_point = objective(objective_thresholds)

    # Compute posterior mean (for ref point computation ref pareto frontier)
    # if one is not provided among arguments.
    Y_pmean = kwargs.get("Y_pmean")
    if Y_pmean is None:
        with torch.no_grad():
            Y_pmean = model.posterior(X_observed).mean
    if alpha > 0:
        partitioning = NondominatedPartitioning(
            ref_point=ref_point,
            Y=objective(Y_pmean),
            alpha=alpha,
        )
    else:
        partitioning = FastNondominatedPartitioning(
            ref_point=ref_point,
            Y=objective(Y_pmean),
        )

    return {
        "model": model,
        "ref_point": ref_point,
        "partitioning": partitioning,
        "objective": objective,
    }


@acqf_input_constructor(qExpectedHypervolumeImprovement)
def construct_inputs_qEHVI(
    model: Model,
    training_data: TrainingData,
    objective_thresholds: Tensor,
    objective: Optional[MCMultiOutputObjective] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    r"""Construct kwargs for `qExpectedHypervolumeImprovement` constructor."""
    X_observed = training_data.X

    # compute posterior mean (for ref point computation ref pareto frontier)
    with torch.no_grad():
        Y_pmean = model.posterior(X_observed).mean

    outcome_constraints = kwargs.pop("outcome_constraints", None)
    # For HV-based acquisition functions we pass the constraint transform directly
    if outcome_constraints is None:
        cons_tfs = None
    else:
        cons_tfs = get_outcome_constraint_transforms(outcome_constraints)
        # Adjust `Y_pmean` to contrain feasible points only.
        feas = torch.stack([c(Y_pmean) <= 0 for c in cons_tfs], dim=-1).all(dim=-1)
        Y_pmean = Y_pmean[feas]

    if objective is None:
        objective = IdentityMCMultiOutputObjective()

    ehvi_kwargs = construct_inputs_EHVI(
        model=model,
        training_data=training_data,
        objective_thresholds=objective_thresholds,
        objective=objective,
        # Pass `Y_pmean` that accounts for constraints to `construct_inputs_EHVI`
        # to ensure that correct non-dominated partitioning is produced.
        Y_pmean=Y_pmean,
        **kwargs,
    )

    sampler = kwargs.get("sampler")
    if sampler is None:
        sampler = _get_sampler(
            mc_samples=kwargs.get("mc_samples", 128), qmc=kwargs.get("qmc", True)
        )

    add_qehvi_kwargs = {
        "sampler": sampler,
        "X_pending": kwargs.get("X_pending"),
        "constraints": cons_tfs,
        "eta": kwargs.get("eta", 1e-3),
    }
    return {**ehvi_kwargs, **add_qehvi_kwargs}


@acqf_input_constructor(qNoisyExpectedHypervolumeImprovement)
def construct_inputs_qNEHVI(
    model: Model,
    training_data: TrainingData,
    objective_thresholds: Tensor,
    objective: Optional[MCMultiOutputObjective] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    r"""Construct kwargs for `qNoisyExpectedHypervolumeImprovement` constructor."""
    # This selects the objectives (a subset of the outcomes) and set each
    # objective threhsold to have the proper optimization direction.
    if objective is None:
        objective = IdentityMCMultiOutputObjective()

    outcome_constraints = kwargs.pop("outcome_constraints", None)
    if outcome_constraints is None:
        cons_tfs = None
    else:
        cons_tfs = get_outcome_constraint_transforms(outcome_constraints)

    sampler = kwargs.get("sampler")
    if sampler is None:
        sampler = _get_sampler(
            mc_samples=kwargs.get("mc_samples", 128), qmc=kwargs.get("qmc", True)
        )

    return {
        "model": model,
        "ref_point": objective(objective_thresholds),
        "X_baseline": kwargs.get("X_baseline", training_data.X),
        "sampler": sampler,
        "objective": objective,
        "constraints": cons_tfs,
        "X_pending": kwargs.get("X_pending"),
        "eta": kwargs.get("eta", 1e-3),
        "prune_baseline": kwargs.get("prune_baseline", True),
        "alpha": kwargs.get("alpha", 0.0),
        "cache_pending": kwargs.get("cache_pending", True),
        "max_iep": kwargs.get("max_iep", 0),
        "incremental_nehvi": kwargs.get("incremental_nehvi", True),
    }


def get_best_f_analytic(
    training_data: TrainingData,
    posterior_transform: Optional[PosteriorTransform] = None,
    **kwargs,
) -> Tensor:
    if not training_data.is_block_design:
        raise NotImplementedError("Currently only block designs are supported.")
    Y = training_data.Y
    posterior_transform = _deprecate_objective_arg(
        posterior_transform=posterior_transform, objective=kwargs.get("objective", None)
    )
    if posterior_transform is not None:
        return posterior_transform.evaluate(Y).max(-1).values
    if Y.shape[-1] > 1:
        raise NotImplementedError
    return Y.max(-2).values.squeeze(-1)


def get_best_f_mc(
    training_data: TrainingData,
    objective: Optional[MCAcquisitionObjective] = None,
    posterior_transform: Optional[PosteriorTransform] = None,
) -> Tensor:
    if not training_data.is_block_design:
        raise NotImplementedError("Currently only block designs are supported.")
    Y = training_data.Y
    posterior_transform = _deprecate_objective_arg(
        posterior_transform=posterior_transform,
        objective=objective
        if not isinstance(objective, MCAcquisitionObjective)
        else None,
    )
    if posterior_transform is not None:
        # retain the original tensor dimension since objective expects explicit
        # output dimension.
        Y_dim = Y.dim()
        Y = posterior_transform.evaluate(Y)
        if Y.dim() < Y_dim:
            Y = Y.unsqueeze(-1)
    if objective is None:
        if Y.shape[-1] > 1:
            raise UnsupportedError
        objective = IdentityMCObjective()
    return objective(Y).max(-1).values
