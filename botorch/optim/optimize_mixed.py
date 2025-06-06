#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import warnings
from typing import Any, Callable

import torch
from botorch.acquisition import AcquisitionFunction
from botorch.exceptions.errors import CandidateGenerationError, UnsupportedError
from botorch.exceptions.warnings import OptimizationWarning
from botorch.generation.gen import gen_candidates_scipy
from botorch.optim.initializers import initialize_q_batch
from botorch.optim.optimize import (
    _optimize_acqf,
    _validate_sequential_inputs,
    OptimizeAcqfInputs,
)
from botorch.optim.utils.acquisition_utils import fix_features, get_X_baseline
from botorch.utils.sampling import (
    draw_sobol_samples,
    HitAndRunPolytopeSampler,
    sparse_to_dense_constraints,
)
from botorch.utils.transforms import unnormalize
from pyre_extensions import assert_is_instance, none_throws
from torch import Tensor
from torch.quasirandom import SobolEngine

# Default values.
# NOTE: When changing a default, update the corresponding value in the docstrings.
STD_CONT_PERTURBATION = 0.1
RAW_SAMPLES = 1024  # Number of candidates from which to select starting points.
NUM_RESTARTS = 20  # Number of restarts of optimizer with different starting points.
MAX_BATCH_SIZE = 2048  # Maximum batch size.
MAX_ITER_ALTER = 64  # Maximum number of alternating iterations.
MAX_ITER_DISCRETE = 4  # Maximum number of discrete iterations.
MAX_ITER_CONT = 8  # Maximum number of continuous iterations.
# Maximum number of discrete values for a discrete dimension.
# If there are more values for a dimension, we will use continuous
# relaxation to optimize it.
MAX_DISCRETE_VALUES = 20
# Maximum number of iterations for optimizing the continuous relaxation
# during initialization
MAX_ITER_INIT = 100
CONVERGENCE_TOL = 1e-8  # Optimizer convergence tolerance.
DUPLICATE_TOL = 1e-6  # Tolerance for deduplicating initial candidates.

SUPPORTED_OPTIONS = {
    "initialization_strategy",
    "tol",
    "maxiter_alternating",
    "maxiter_discrete",
    "maxiter_continuous",
    "maxiter_init",
    "max_discrete_values",
    "num_spray_points",
    "std_cont_perturbation",
    "batch_limit",
    "init_batch_limit",
}
SUPPORTED_INITIALIZATION = {"continuous_relaxation", "equally_spaced", "random"}


def _setup_continuous_relaxation(
    discrete_dims: list[int],
    bounds: Tensor,
    max_discrete_values: int,
    post_processing_func: Callable[[Tensor], Tensor] | None,
) -> tuple[list[int], Callable[[Tensor], Tensor] | None]:
    r"""Update `discrete_dims` and `post_processing_func` to use
    continuous relaxation for discrete dimensions that have more than
    `max_discrete_values` values. These dimensions are removed from
    `discrete_dims` and `post_processing_func` is updated to round
    them to the nearest integer.
    """
    discrete_dims_t = torch.tensor(discrete_dims, dtype=torch.long)
    num_discrete_values = (
        bounds[1, discrete_dims_t] - bounds[0, discrete_dims_t]
    ).cpu()
    dims_to_relax = discrete_dims_t[num_discrete_values > max_discrete_values]
    if dims_to_relax.numel() == 0:
        # No dimension needs continuous relaxation.
        return discrete_dims, post_processing_func
    # Remove relaxed dims from `discrete_dims`.
    discrete_dims = list(set(discrete_dims).difference(dims_to_relax.tolist()))

    def new_post_processing_func(X: Tensor) -> Tensor:
        r"""Round the relaxed dimensions to the nearest integer and apply the original
        `post_processing_func`."""
        X[..., dims_to_relax] = X[..., dims_to_relax].round()
        if post_processing_func is not None:
            X = post_processing_func(X)
        return X

    return discrete_dims, new_post_processing_func


def _filter_infeasible(
    X: Tensor, inequality_constraints: list[tuple[Tensor, Tensor, float]] | None
) -> Tensor:
    r"""Filters infeasible points from a set of points.

    NOTE: This function only supports intra-point constraints. This is validated
        in `optimize_acqf_mixed_alternating`, so we do not repeat the
        validation in here.

    Args:
        X: A tensor of points of shape `n x d`.
        inequality_constraints: A list of tuples (indices, coefficients, rhs),
            with each tuple encoding an inequality constraint of the form
            `\sum_i (X[indices[i]] * coefficients[i]) >= rhs`. `indices` and
            `coefficients` should be torch tensors. See the docstring of
            `make_scipy_linear_constraints` for an example.

    Returns:
        The tensor `X` with infeasible points removed.
    """
    if inequality_constraints is None:
        return X
    is_feasible = torch.ones(X.shape[:-1], device=X.device, dtype=torch.bool)
    for idx, coef, rhs in inequality_constraints:
        is_feasible &= (X[..., idx] * coef).sum(dim=-1) >= rhs
    return X[is_feasible]


def get_nearest_neighbors(
    current_x: Tensor,
    bounds: Tensor,
    discrete_dims: Tensor,
) -> Tensor:
    r"""Generate all 1-Manhattan distance neighbors of a given input. The neighbors
    are generated for the discrete dimensions only.

    NOTE: This assumes that `current_x` is detached and uses in-place operations,
    which are known to be incompatible with autograd.

    Args:
        current_x: The design to find the neighbors of. A tensor of shape `d`.
        bounds: A `2 x d` tensor of lower and upper bounds for each column of `X`.
        discrete_dims: A tensor of indices corresponding to binary and
            integer parameters.

    Returns:
        A tensor of shape `num_neighbors x d`, denoting all unique 1-Manhattan
        distance neighbors.
    """
    num_discrete = len(discrete_dims)
    diag_ones = torch.eye(num_discrete, dtype=current_x.dtype, device=current_x.device)
    # Neighbors obtained by increasing a discrete dimension by one.
    plus_neighbors = current_x.repeat(num_discrete, 1)
    plus_neighbors[:, discrete_dims] += diag_ones
    plus_neighbors.clamp_(max=bounds[1])
    # Neighbors obtained by decreasing a discrete dimension by one.
    minus_neighbors = current_x.repeat(num_discrete, 1)
    minus_neighbors[:, discrete_dims] -= diag_ones
    minus_neighbors.clamp_(min=bounds[0])
    unique_neighbors = torch.cat([minus_neighbors, plus_neighbors], dim=0).unique(dim=0)
    # Also remove current_x if it is in unique_neighbors.
    unique_neighbors = unique_neighbors[~(unique_neighbors == current_x).all(dim=-1)]
    return unique_neighbors


def get_spray_points(
    X_baseline: Tensor,
    cont_dims: Tensor,
    discrete_dims: Tensor,
    bounds: Tensor,
    num_spray_points: int,
    std_cont_perturbation: float = STD_CONT_PERTURBATION,
) -> Tensor:
    r"""Generate spray points by perturbing the Pareto optimal points.

    Given the points on the Pareto frontier, we create perturbations (spray points)
    by adding Gaussian perturbation to the continuous parameters and 1-Manhattan
    distance neighbors of the discrete (binary and integer) parameters.

    Args:
        X_baseline: Tensor of best acquired points across BO run.
        cont_dims: Indices of continuous parameters/input dimensions.
        discrete_dims: Indices of binary/integer parameters/input dimensions.
        bounds: A `2 x d` tensor of lower and upper bounds for each column of `X`.
        num_spray_points: Number of spray points to return.
        std_cont_perturbation: standard deviation of Normal perturbations of
            continuous dimensions. Default is STD_CONT_PERTURBATION = 0.2.

    Returns:
        A (num_spray_points x d)-dim tensor of perturbed points.
    """
    dim = bounds.shape[-1]
    device, dtype = X_baseline.device, X_baseline.dtype
    perturb_nbors = torch.zeros(0, dim, device=device, dtype=dtype)
    for x in X_baseline:
        discrete_perturbs = get_nearest_neighbors(
            current_x=x, bounds=bounds, discrete_dims=discrete_dims
        )
        discrete_perturbs = discrete_perturbs[
            torch.randint(len(discrete_perturbs), (num_spray_points,), device=device)
        ]
        cont_perturbs = x[cont_dims] + std_cont_perturbation * torch.randn(
            num_spray_points, len(cont_dims), device=device, dtype=dtype
        )
        cont_perturbs = cont_perturbs.clamp_(
            min=bounds[0, cont_dims], max=bounds[1, cont_dims]
        )
        nbds = torch.zeros(num_spray_points, dim, device=device, dtype=dtype)
        nbds[..., discrete_dims] = discrete_perturbs[..., discrete_dims]
        nbds[..., cont_dims] = cont_perturbs
        perturb_nbors = torch.cat([perturb_nbors, nbds], dim=0)
    return perturb_nbors


def sample_feasible_points(
    opt_inputs: OptimizeAcqfInputs,
    discrete_dims: Tensor,
    num_points: int,
) -> Tensor:
    r"""Sample feasible points from the optimization domain.

    Feasibility is determined according to the discrete dimensions taking
    integer values and the inequality constraints being satisfied.

    If there are no inequality constraints, Sobol is used to generate the base points.
    Otherwise, we use the polytope sampler to generate the base points. The base points
    are then rounded to the nearest integer values for the discrete dimensions, and
    the infeasible points are filtered out (in case rounding leads to infeasibility).

    This method will do 10 attempts to generate `num_points` feasible points, and
    return the points generated so far. If no points are generated, it will error out.

    Args:
        opt_inputs: Common set of arguments for acquisition optimization.
        discrete_dims: A tensor of indices corresponding to binary and
            integer parameters.
        num_points: The number of points to sample.

    Returns:
        A tensor of shape `num_points x d` containing the sampled points.
    """
    bounds = opt_inputs.bounds
    all_points = torch.empty(
        0, bounds.shape[-1], device=bounds.device, dtype=bounds.dtype
    )
    constraints = opt_inputs.inequality_constraints
    if constraints is None:
        # Generate base points using Sobol.
        sobol_engine = SobolEngine(dimension=bounds.shape[-1], scramble=True)

        def generator(n: int) -> Tensor:
            samples = sobol_engine.draw(n=n, dtype=bounds.dtype).to(bounds.device)
            return unnormalize(X=samples, bounds=bounds)

    else:
        # Generate base points using polytope sampler.
        # Since we may generate many times, we initialize the sampler with burn-in
        # to reduce the start-up cost for subsequent calls.
        A, b = sparse_to_dense_constraints(d=bounds.shape[-1], constraints=constraints)
        polytope_sampler = HitAndRunPolytopeSampler(
            bounds=bounds, inequality_constraints=(-A, -b)
        )

        def generator(n: int) -> Tensor:
            return polytope_sampler.draw(n=n)

    for _ in range(10):
        num_remaining = num_points - len(all_points)
        if num_remaining <= 0:
            break
        # Generate twice as many, since we're likely to filter out some points.
        base_points = generator(n=num_remaining * 2)
        # Round the discrete dimensions to the nearest integer.
        base_points[:, discrete_dims] = base_points[:, discrete_dims].round()
        # Fix the fixed features.
        base_points = fix_features(
            X=base_points, fixed_features=opt_inputs.fixed_features
        )
        # Filter out infeasible points.
        feasible_points = _filter_infeasible(
            X=base_points, inequality_constraints=constraints
        )
        all_points = torch.cat([all_points, feasible_points], dim=0)

    if len(all_points) == 0:
        raise CandidateGenerationError(
            "Could not generate any feasible starting points for mixed optimizer."
        )
    return all_points[:num_points]


def generate_starting_points(
    opt_inputs: OptimizeAcqfInputs,
    discrete_dims: Tensor,
    cont_dims: Tensor,
) -> tuple[Tensor, Tensor]:
    """Generate initial starting points for the alternating optimization.

    This method attempts to generate the initial points using the specified
    options and completes any missing points using `sample_feasible_points`.

    Args:
        opt_inputs: Common set of arguments for acquisition optimization.
            This function utilizes `acq_function`, `bounds`, `num_restarts`,
            `raw_samples`, `options`, `fixed_features` and constraints
            from `opt_inputs`.
        discrete_dims: A tensor of indices corresponding to integer and
            binary parameters.
        cont_dims: A tensor of indices corresponding to continuous parameters.

    Returns:
        A tuple of two tensors: a (num_restarts x d)-dim tensor of starting points
        and a (num_restarts)-dim tensor of their respective acquisition values.
        In rare cases, this method may return fewer than `num_restarts` points.
    """
    bounds = opt_inputs.bounds
    binary_dims = []
    for dim in discrete_dims:
        if bounds[0, dim] == 0 and bounds[1, dim] == 1:
            binary_dims.append(dim)
    num_binary = len(binary_dims)
    num_integer = len(discrete_dims) - num_binary
    num_restarts = opt_inputs.num_restarts
    raw_samples = none_throws(opt_inputs.raw_samples)

    options = opt_inputs.options or {}
    initialization_strategy = options.get(
        "initialization_strategy",
        (
            "equally_spaced"
            if num_integer == 0 and num_binary >= 2
            else "continuous_relaxation"
        ),
    )
    if initialization_strategy not in SUPPORTED_INITIALIZATION:
        raise UnsupportedError(  # pragma: no cover
            f"Unsupported initialization strategy: {initialization_strategy}."
            f"Supported strategies are: {SUPPORTED_INITIALIZATION}."
        )

    # Initialize `x_init_candts` here so that it's always defined as a tensor.
    x_init_candts = torch.empty(
        0, bounds.shape[-1], device=bounds.device, dtype=bounds.dtype
    )
    if initialization_strategy == "continuous_relaxation":
        try:
            # Optimize the acquisition function with continuous relaxation.
            updated_opt_inputs = dataclasses.replace(
                opt_inputs,
                q=1,
                return_best_only=False,
                options={
                    "maxiter": options.get("maxiter_init", MAX_ITER_INIT),
                    "batch_limit": options.get("batch_limit", MAX_BATCH_SIZE),
                    "init_batch_limit": options.get("init_batch_limit", MAX_BATCH_SIZE),
                },
            )
            x_init_candts, _ = _optimize_acqf(opt_inputs=updated_opt_inputs)
            x_init_candts = x_init_candts.squeeze(-2).detach()
        except Exception as e:
            warnings.warn(
                "Failed to initialize using continuous relaxation. Using "
                "`sample_feasible_points` for initialization. Original error "
                f"message: {e}",
                OptimizationWarning,
                stacklevel=2,
            )

    if len(x_init_candts) == 0:
        # Generate Sobol points as a fallback for `continuous_relaxation` and for
        # further refinement in `equally_spaced` strategy.
        x_init_candts = draw_sobol_samples(bounds=bounds, n=raw_samples, q=1)
        x_init_candts = x_init_candts.squeeze(-2)

    if initialization_strategy == "equally_spaced":
        if num_integer > 0:
            raise ValueError(  # pragma: no cover
                "Equally spaced initialization is not supported with non-binary "
                "discrete variables."
            )
        # Picking initial points by equally spaced number of features/binary inputs.
        k = torch.randint(
            low=0,
            high=num_binary,
            size=(raw_samples,),
            dtype=torch.int64,
            device=bounds.device,
        )
        x_init_candts[:, binary_dims] = 0
        binary_dims_t = torch.as_tensor(binary_dims, device=bounds.device)
        for i, xi in enumerate(x_init_candts):
            rand_binary_dims = binary_dims_t[
                torch.randperm(num_binary, device=xi.device)[: k[i]]
            ]
            x_init_candts[i, rand_binary_dims] = 1

    num_spray_points = assert_is_instance(
        options.get("num_spray_points", 20 if num_integer == 0 else 0), int
    )
    if (
        num_spray_points > 0
        and (X_baseline := get_X_baseline(acq_function=opt_inputs.acq_function))
        is not None
    ):
        perturb_nbors = get_spray_points(
            X_baseline=X_baseline,
            cont_dims=cont_dims,
            discrete_dims=discrete_dims,
            bounds=bounds,
            num_spray_points=num_spray_points,
            std_cont_perturbation=assert_is_instance(
                options.get("std_cont_perturbation", STD_CONT_PERTURBATION), float
            ),
        )
        x_init_candts = torch.cat([x_init_candts, perturb_nbors], dim=0)

    # Process the candidates to make sure they are all feasible.
    x_init_candts[..., discrete_dims] = x_init_candts[..., discrete_dims].round()
    x_init_candts = fix_features(
        X=x_init_candts, fixed_features=opt_inputs.fixed_features
    )
    x_init_candts = _filter_infeasible(
        X=x_init_candts, inequality_constraints=opt_inputs.inequality_constraints
    )

    # If there are fewer than `num_restarts` feasible points, attempt to generate more.
    if len(x_init_candts) < num_restarts:
        new_x_init = sample_feasible_points(
            opt_inputs=opt_inputs,
            discrete_dims=discrete_dims,
            num_points=num_restarts - len(x_init_candts),
        )
        x_init_candts = torch.cat([x_init_candts, new_x_init], dim=0)

    with torch.no_grad():
        acq_vals = torch.cat(
            [
                opt_inputs.acq_function(X_.unsqueeze(-2))
                for X_ in x_init_candts.split(
                    options.get("init_batch_limit", MAX_BATCH_SIZE)
                )
            ]
        )
    if len(x_init_candts) > num_restarts:
        # If there are more than `num_restarts` feasible points, select a diverse
        # set of initializers using Boltzmann sampling.
        x_init_candts, acq_vals = initialize_q_batch(
            X=x_init_candts, acq_vals=acq_vals, n=num_restarts
        )
    return x_init_candts, acq_vals


def discrete_step(
    opt_inputs: OptimizeAcqfInputs,
    discrete_dims: Tensor,
    current_x: Tensor,
) -> tuple[Tensor, Tensor]:
    """Discrete nearest neighbour search.

    Args:
        opt_inputs: Common set of arguments for acquisition optimization.
            This function utilizes `acq_function`, `bounds`, `options`
            and constraints from `opt_inputs`.
        discrete_dims: A tensor of indices corresponding to binary and
            integer parameters.
        current_x: Starting point. A tensor of shape `d`.

    Returns:
        A tuple of two tensors: a (d)-dim tensor of optimized point
            and a scalar tensor of correspondins acquisition value.
    """
    with torch.no_grad():
        current_acqval = opt_inputs.acq_function(current_x.unsqueeze(0))
    options = opt_inputs.options or {}
    for _ in range(
        assert_is_instance(options.get("maxiter_discrete", MAX_ITER_DISCRETE), int)
    ):
        x_neighbors = get_nearest_neighbors(
            current_x=current_x.detach(),
            bounds=opt_inputs.bounds,
            discrete_dims=discrete_dims,
        )
        x_neighbors = _filter_infeasible(
            X=x_neighbors, inequality_constraints=opt_inputs.inequality_constraints
        )
        if x_neighbors.numel() == 0:
            # Exit gracefully with last point if there are no feasible neighbors.
            break
        with torch.no_grad():
            acq_vals = torch.cat(
                [
                    opt_inputs.acq_function(X_.unsqueeze(-2))
                    for X_ in x_neighbors.split(
                        options.get("init_batch_limit", MAX_BATCH_SIZE)
                    )
                ]
            )
        argmax = acq_vals.argmax()
        improvement = acq_vals[argmax] - current_acqval
        if improvement > 0:
            current_acqval, current_x = acq_vals[argmax], x_neighbors[argmax]
        if improvement <= options.get("tol", CONVERGENCE_TOL):
            break
    return current_x, current_acqval


def continuous_step(
    opt_inputs: OptimizeAcqfInputs,
    discrete_dims: Tensor,
    current_x: Tensor,
) -> tuple[Tensor, Tensor]:
    """Continuous search using L-BFGS-B through optimize_acqf.

    Args:
        opt_inputs: Common set of arguments for acquisition optimization.
            This function utilizes `acq_function`, `bounds`, `options`,
            `fixed_features` and constraints from `opt_inputs`.
        discrete_dims: A tensor of indices corresponding to binary and
            integer parameters.
        current_x: Starting point. A tensor of shape `d`.

    Returns:
        A tuple of two tensors: a (1 x d)-dim tensor of optimized points
            and a (1)-dim tensor of acquisition values.
    """
    options = opt_inputs.options or {}
    if len(discrete_dims) == len(current_x):  # nothing continuous to optimize
        with torch.no_grad():
            return current_x, opt_inputs.acq_function(current_x.unsqueeze(0))

    updated_opt_inputs = dataclasses.replace(
        opt_inputs,
        q=1,
        num_restarts=1,
        raw_samples=None,
        batch_initial_conditions=current_x.unsqueeze(0),
        fixed_features={
            **dict(zip(discrete_dims.tolist(), current_x[discrete_dims])),
            **(opt_inputs.fixed_features or {}),
        },
        options={
            "maxiter": options.get("maxiter_continuous", MAX_ITER_CONT),
            "tol": options.get("tol", CONVERGENCE_TOL),
            "batch_limit": options.get("batch_limit", MAX_BATCH_SIZE),
        },
    )
    return _optimize_acqf(opt_inputs=updated_opt_inputs)


def optimize_acqf_mixed_alternating(
    acq_function: AcquisitionFunction,
    bounds: Tensor,
    discrete_dims: list[int],
    options: dict[str, Any] | None = None,
    q: int = 1,
    raw_samples: int = RAW_SAMPLES,
    num_restarts: int = NUM_RESTARTS,
    post_processing_func: Callable[[Tensor], Tensor] | None = None,
    sequential: bool = True,
    fixed_features: dict[int, float] | None = None,
    inequality_constraints: list[tuple[Tensor, Tensor, float]] | None = None,
) -> tuple[Tensor, Tensor]:
    r"""
    Optimizes acquisition function over mixed binary and continuous input spaces.
    Multiple random restarting starting points are picked by evaluating a large set
    of initial candidates. From each starting point, alternating discrete local search
    and continuous optimization via (L-BFGS) is performed for a fixed number of
    iterations.

    NOTE: This method assumes that all discrete variables are integer valued.
    The discrete dimensions that have more than
    `options.get("max_discrete_values", MAX_DISCRETE_VALUES)` values will
    be optimized using continuous relaxation.

    # TODO: Support categorical variables.

    Args:
        acq_function: BoTorch Acquisition function.
        bounds: A `2 x d` tensor of lower and upper bounds for each column of `X`.
        discrete_dims: A list of indices corresponding to integer and binary parameters.
        options: Dictionary specifying optimization options. Supports the following:
        - "initialization_strategy": Strategy used to generate the initial candidates.
            "random", "continuous_relaxation" or "equally_spaced" (linspace style).
        - "tol": The algorithm terminates if the absolute improvement in acquisition
            value of one iteration is smaller than this number.
        - "maxiter_alternating": Number of alternating steps. Defaults to 64.
        - "maxiter_discrete": Maximum number of iterations in each discrete step.
            Defaults to 4.
        - "maxiter_continuous": Maximum number of iterations in each continuous step.
            Defaults to 8.
        - "max_discrete_values": Maximum number of values for a discrete dimension
            to be optimized using discrete step / local search. The discrete dimensions
            with more values will be optimized using continuous relaxation.
        - "num_spray_points": Number of spray points (around `X_baseline`) to add to
            the points generated by the initialization strategy. Defaults to 20 if
            all discrete variables are binary and to 0 otherwise.
        - "std_cont_perturbation": Standard deviation of the normal perturbations of
            the continuous variables used to generate the spray points.
            Defaults to 0.1.
        - "batch_limit": The maximum batch size for jointly evaluating candidates
            during optimization.
        - "init_batch_limit": The maximum batch size for jointly evaluating candidates
            during initialization. During initialization, candidates are evaluated
            in a `no_grad` context, which reduces memory usage. As a result,
            `init_batch_limit` can be set to a larger value than `batch_limit`.
            Defaults to `batch_limit`, if given.
        q: Number of candidates.
        raw_samples: Number of initial candidates used to select starting points from.
            Defaults to 1024.
        num_restarts: Number of random restarts. Defaults to 20.
        post_processing_func: A function that post-processes an optimization result
            appropriately (i.e., according to `round-trip` transformations).
        sequential: Whether to use joint or sequential optimization across q-batch.
            This currently only supports sequential optimization.
        fixed_features: A map `{feature_index: value}` for features that
            should be fixed to a particular value during generation.
        inequality_constraints: A list of tuples (indices, coefficients, rhs),
            with each tuple encoding an inequality constraint of the form
            `\sum_i (X[indices[i]] * coefficients[i]) >= rhs`. `indices` and
            `coefficients` should be torch tensors. See the docstring of
            `make_scipy_linear_constraints` for an example.

    Returns:
        A tuple of two tensors: a (q x d)-dim tensor of optimized points
            and a (q)-dim tensor of their respective acquisition values.
    """
    if sequential is False:  # pragma: no cover
        raise NotImplementedError(
            "`optimize_acqf_mixed_alternating` only supports "
            "sequential optimization."
        )

    fixed_features = fixed_features or {}
    options = options or {}
    options.setdefault("batch_limit", MAX_BATCH_SIZE)
    options.setdefault("init_batch_limit", options["batch_limit"])
    if not (keys := set(options.keys())).issubset(SUPPORTED_OPTIONS):
        unsupported_keys = keys.difference(SUPPORTED_OPTIONS)
        raise UnsupportedError(
            f"Received an unsupported option {unsupported_keys}. {SUPPORTED_OPTIONS=}."
        )

    # Update discrete dims and post processing functions to account for any
    # dimensions that should be using continuous relaxation.
    discrete_dims, post_processing_func = _setup_continuous_relaxation(
        discrete_dims=discrete_dims,
        bounds=bounds,
        max_discrete_values=assert_is_instance(
            options.get("max_discrete_values", MAX_DISCRETE_VALUES), int
        ),
        post_processing_func=post_processing_func,
    )

    opt_inputs = OptimizeAcqfInputs(
        acq_function=acq_function,
        bounds=bounds,
        q=q,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        options=options,
        inequality_constraints=inequality_constraints,
        equality_constraints=None,
        nonlinear_inequality_constraints=None,
        fixed_features=fixed_features,
        post_processing_func=post_processing_func,
        batch_initial_conditions=None,
        return_best_only=True,
        gen_candidates=gen_candidates_scipy,
        sequential=sequential,
    )
    _validate_sequential_inputs(opt_inputs=opt_inputs)

    base_X_pending = acq_function.X_pending if q > 1 else None
    dim = bounds.shape[-1]
    tkwargs: dict[str, Any] = {"device": bounds.device, "dtype": bounds.dtype}
    # Remove fixed features from dims, so they don't get optimized.
    discrete_dims = [dim for dim in discrete_dims if dim not in fixed_features]
    if len(discrete_dims) == 0:
        return _optimize_acqf(opt_inputs=opt_inputs)
    if not (
        isinstance(discrete_dims, list)
        and len(set(discrete_dims)) == len(discrete_dims)
        and min(discrete_dims) >= 0
        and max(discrete_dims) <= dim - 1
    ):
        raise ValueError(
            "`discrete_dims` must be a list with unique integers "
            "between 0 and num_dims - 1."
        )
    discrete_dims_t = torch.tensor(
        discrete_dims, dtype=torch.long, device=tkwargs["device"]
    )
    cont_dims = complement_indices_like(indices=discrete_dims_t, d=dim)
    # Fixed features are all in cont_dims. Remove them, so they don't get optimized.
    ff_idcs = torch.tensor(
        list(fixed_features.keys()), dtype=torch.long, device=tkwargs["device"]
    )
    cont_dims = cont_dims[(cont_dims.unsqueeze(-1) != ff_idcs).all(dim=-1)]
    candidates = torch.empty(0, dim, **tkwargs)
    for _q in range(q):
        # Generate starting points.
        best_X, best_acq_val = generate_starting_points(
            opt_inputs=opt_inputs,
            discrete_dims=discrete_dims_t,
            cont_dims=cont_dims,
        )

        # TODO: Eliminate this for loop. Tensors being unequal sizes could potentially
        # be handled by concatenating them rather than stacking, and keeping a list
        # of indices.
        for i in range(num_restarts):
            alternate_steps = 0
            while alternate_steps < options.get("maxiter_alternating", MAX_ITER_ALTER):
                starting_acq_val = best_acq_val[i].clone()
                alternate_steps += 1
                for step in (discrete_step, continuous_step):
                    best_X[i], best_acq_val[i] = step(
                        opt_inputs=opt_inputs,
                        discrete_dims=discrete_dims_t,
                        current_x=best_X[i],
                    )

                improvement = best_acq_val[i] - starting_acq_val
                if improvement < options.get("tol", CONVERGENCE_TOL):
                    # Check for convergence
                    break

        new_candidate = best_X[torch.argmax(best_acq_val)].unsqueeze(0)
        candidates = torch.cat([candidates, new_candidate], dim=-2)
        # Update pending points to include the new candidate.
        if q > 1:
            acq_function.set_X_pending(
                torch.cat([base_X_pending, candidates], dim=-2)
                if base_X_pending is not None
                else candidates
            )
    if q > 1:
        acq_function.set_X_pending(base_X_pending)

    if post_processing_func is not None:
        candidates = post_processing_func(candidates)

    with torch.no_grad():
        acq_value = acq_function(candidates)  # compute joint acquisition value
    return candidates, acq_value


def complement_indices_like(indices: Tensor, d: int) -> Tensor:
    r"""Computes a tensor of complement indices: {range(d) \\ indices}.
    Same as complement_indices but returns an integer tensor like indices.
    """
    return torch.tensor(
        complement_indices(indices.tolist(), d),
        device=indices.device,
        dtype=indices.dtype,
    )


def complement_indices(indices: list[int], d: int) -> list[int]:
    r"""Computes a list of complement indices: {range(d) \\ indices}.

    Args:
        indices: a list of integers.
        d: an integer dimension in which to compute the complement.

    Returns:
        A list of integer indices.
    """
    return sorted(set(range(d)).difference(indices))
