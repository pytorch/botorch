#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Objective Modules to be used with acquisition functions.
"""

from __future__ import annotations

import inspect
import warnings
from abc import ABC, abstractmethod
from typing import Callable, List, Optional

import torch
from botorch.posteriors.gpytorch import GPyTorchPosterior, scalarize_posterior
from botorch.utils import apply_constraints
from torch import Tensor
from torch.nn import Module


class AcquisitionObjective(Module, ABC):
    r"""Abstract base class for objectives."""

    ...


class ScalarizedObjective(AcquisitionObjective):
    r"""Affine objective to be used with analytic acquisition functions.

    For a Gaussian posterior at a single point (`q=1`) with mean `mu` and
    covariance matrix `Sigma`, this yields a single-output posterior with mean
    `weights^T * mu` and variance `weights^T Sigma w`.

    Example:
        Example for a model with two outcomes:

        >>> weights = torch.tensor([0.5, 0.25])
        >>> objective = ScalarizedObjective(weights)
        >>> EI = ExpectedImprovement(model, best_f=0.1, objective=objective)
    """

    def __init__(self, weights: Tensor, offset: float = 0.0) -> None:
        r"""Affine objective.

        Args:
            weights: A one-dimensional tensor with `m` elements representing the
                linear weights on the outputs.
            offset: An offset to be added to posterior mean.
        """
        if weights.dim() != 1:
            raise ValueError("weights must be a one-dimensional tensor.")
        super().__init__()
        self.register_buffer("weights", weights)
        self.offset = offset

    def forward(self, posterior: GPyTorchPosterior) -> GPyTorchPosterior:
        r"""Compute the posterior of the affine transformation.

        Args:
            posterior: A posterior with the same number of outputs as the
                elements in `self.weights`.

        Returns:
            A single-output posterior.
        """
        return scalarize_posterior(
            posterior=posterior, weights=self.weights, offset=self.offset
        )


class MCAcquisitionObjective(AcquisitionObjective):
    r"""Abstract base class for MC-based objectives."""

    @abstractmethod
    def forward(self, samples: Tensor, X: Optional[Tensor] = None) -> Tensor:
        r"""Evaluate the objective on the samples.

        Args:
            samples: A `sample_shape x batch_shape x q x m`-dim Tensors of
                samples from a model posterior.
            X: A `batch_shape x q x d`-dim tensor of inputs. Relevant only if
                the objective depends on the inputs explicitly.

        Returns:
            Tensor: A `sample_shape x batch_shape x q`-dim Tensor of objective
            values (assuming maximization).

        This method is usually not called directly, but via the objectives

        Example:
            >>> # `__call__` method:
            >>> samples = sampler(posterior)
            >>> outcome = mc_obj(samples)
        """
        pass  # pragma: no cover


class IdentityMCObjective(MCAcquisitionObjective):
    r"""Trivial objective extracting the last dimension.

    Example:
        >>> identity_objective = IdentityMCObjective()
        >>> samples = sampler(posterior)
        >>> objective = identity_objective(samples)
    """

    def forward(self, samples: Tensor, X: Optional[Tensor] = None) -> Tensor:
        return samples.squeeze(-1)


class LinearMCObjective(MCAcquisitionObjective):
    r"""Linear objective constructed from a weight tensor.

    For input `samples` and `mc_obj = LinearMCObjective(weights)`, this produces
    `mc_obj(samples) = sum_{i} weights[i] * samples[..., i]`

    Example:
        Example for a model with two outcomes:

        >>> weights = torch.tensor([0.75, 0.25])
        >>> linear_objective = LinearMCObjective(weights)
        >>> samples = sampler(posterior)
        >>> objective = linear_objective(samples)
    """

    def __init__(self, weights: Tensor) -> None:
        r"""Linear Objective.

        Args:
            weights: A one-dimensional tensor with `m` elements representing the
                linear weights on the outputs.
        """
        super().__init__()
        if weights.dim() != 1:
            raise ValueError("weights must be a one-dimensional tensor.")
        self.register_buffer("weights", weights)

    def forward(self, samples: Tensor, X: Optional[Tensor] = None) -> Tensor:
        r"""Evaluate the linear objective on the samples.

        Args:
            samples: A `sample_shape x batch_shape x q x m`-dim tensors of
                samples from a model posterior.
            X: A `batch_shape x q x d`-dim tensor of inputs. Relevant only if
                the objective depends on the inputs explicitly.

        Returns:
            A `sample_shape x batch_shape x q`-dim tensor of objective values.
        """
        if samples.shape[-1] != self.weights.shape[-1]:
            raise RuntimeError("Output shape of samples not equal to that of weights")
        return torch.einsum("...m, m", [samples, self.weights])


class GenericMCObjective(MCAcquisitionObjective):
    r"""Objective generated from a generic callable.

    Allows to construct arbitrary MC-objective functions from a generic
    callable. In order to be able to use gradient-based acquisition function
    optimization it should be possible to backpropagate through the callable.

    Example:
        >>> generic_objective = GenericMCObjective(
                lambda Y, X: torch.sqrt(Y).sum(dim=-1),
            )
        >>> samples = sampler(posterior)
        >>> objective = generic_objective(samples)
    """

    def __init__(self, objective: Callable[[Tensor, Optional[Tensor]], Tensor]) -> None:
        r"""Objective generated from a generic callable.

        Args:
            objective: A callable `f(samples, X)` mapping a
                `sample_shape x batch-shape x q x m`-dim Tensor `samples` and
                an optional `batch-shape x q x d`-dim Tensor `X` to a
                `sample_shape x batch-shape x q`-dim Tensor of objective values.
        """
        super().__init__()
        if len(inspect.signature(objective).parameters) == 1:
            warnings.warn(
                "The `objective` callable of `GenericMCObjective` is expected to "
                "take two arguments. Passing a callable that expects a single "
                "argument will result in an error in future versions.",
                DeprecationWarning,
            )

            def obj(samples: Tensor, X: Optional[Tensor] = None) -> Tensor:
                return objective(samples)

            self.objective = obj
        else:
            self.objective = objective

    def forward(self, samples: Tensor, X: Optional[Tensor] = None) -> Tensor:
        r"""Evaluate the feasibility-weigthed objective on the samples.

        Args:
            samples: A `sample_shape x batch_shape x q x m`-dim Tensors of
                samples from a model posterior.
            X: A `batch_shape x q x d`-dim tensor of inputs. Relevant only if
                the objective depends on the inputs explicitly.

        Returns:
            A `sample_shape x batch_shape x q`-dim Tensor of objective values
            weighted by feasibility (assuming maximization).
        """
        return self.objective(samples, X=X)


class ConstrainedMCObjective(GenericMCObjective):
    r"""Feasibility-weighted objective.

    An Objective allowing to maximize some scalable objective on the model
    outputs subject to a number of constraints. Constraint feasibilty is
    approximated by a sigmoid function.

    `mc_acq(X) = objective(X) * prod_i (1  - sigmoid(constraint_i(X)))`
    TODO: Document functional form exactly.

    See `botorch.utils.objective.apply_constraints` for details on the constarint
    handling.

    Example:
        >>> bound = 0.0
        >>> objective = lambda Y: Y[..., 0]
        >>> # apply non-negativity constraint on f(x)[1]
        >>> constraint = lambda Y: bound - Y[..., 1]
        >>> constrained_objective = ConstrainedMCObjective(objective, [constraint])
        >>> samples = sampler(posterior)
        >>> objective = constrained_objective(samples)
    """

    def __init__(
        self,
        objective: Callable[[Tensor, Optional[Tensor]], Tensor],
        constraints: List[Callable[[Tensor], Tensor]],
        infeasible_cost: float = 0.0,
        eta: float = 1e-3,
    ) -> None:
        r"""Feasibility-weighted objective.

        Args:
            objective: A callable `f(samples, X)` mapping a
                `sample_shape x batch-shape x q x m`-dim Tensor `samples` and
                an optional `batch-shape x q x d`-dim Tensor `X` to a
                `sample_shape x batch-shape x q`-dim Tensor of objective values.
            constraints: A list of callables, each mapping a Tensor of dimension
                `sample_shape x batch-shape x q x m` to a Tensor of dimension
                `sample_shape x batch-shape x q`, where negative values imply
                feasibility.
            infeasible_cost: The cost of a design if all associated samples are
                infeasible.
            eta: The temperature parameter of the sigmoid function approximating
                the constraint.
        """
        super().__init__(objective=objective)
        self.constraints = constraints
        self.eta = eta
        self.register_buffer("infeasible_cost", torch.as_tensor(infeasible_cost))

    def forward(self, samples: Tensor, X: Optional[Tensor] = None) -> Tensor:
        r"""Evaluate the feasibility-weighted objective on the samples.

        Args:
            samples: A `sample_shape x batch_shape x q x m`-dim Tensors of
                samples from a model posterior.
            X: A `batch_shape x q x d`-dim tensor of inputs. Relevant only if
                the objective depends on the inputs explicitly.

        Returns:
            A `sample_shape x batch_shape x q`-dim Tensor of objective values
            weighted by feasibility (assuming maximization).
        """
        obj = super().forward(samples=samples)
        return apply_constraints(
            obj=obj,
            constraints=self.constraints,
            samples=samples,
            infeasible_cost=self.infeasible_cost,
            eta=self.eta,
        )
