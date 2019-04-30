#!/usr/bin/env python3

r"""
Objective Modules to be used with acquisition functions.
"""

from abc import ABC, abstractmethod
from typing import Callable, List

import torch
from botorch.utils import apply_constraints
from torch import Tensor
from torch.nn import Module


class MCAcquisitionObjective(Module, ABC):
    r"""Abstract base class for MC-based objectives.

    Example:
        >>> # This method is usually not called directly, but via the objective's
        >>> # `__call__` method:
        >>> samples = sampler(posterior)
        >>> outcome = mc_obj(samples)
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, samples: Tensor) -> Tensor:
        r"""Evaluate the objective on the samples.

        Args:
            samples: A `sample_shape x batch_shape x q x o`-dim Tensors of
                samples from a model posterior.

        Returns:
            Tensor: A `sample_shape x batch_shape x q`-dim Tensor of objective
            values (assuming maximization).
        """
        pass  # pragma: no cover


class IdentityMCObjective(MCAcquisitionObjective):
    r"""Trivial objective extracting the last dimension.

    Example:
        >>> identity_objective = IdentityMCObjective()
        >>> samples = sampler(posterior)
        >>> objective = identity_objective(samples)
    """

    def forward(self, samples: Tensor) -> Tensor:
        return samples.squeeze(-1)


class LinearMCObjective(MCAcquisitionObjective):
    r"""Linear objective constructed from a weight tensor.

    For input `samples` and `mc_obj = LinearMCObjective(weights)`, this produces
    `mc_obj(samples) = sum_{i} weights[i] * samples[..., i]`

    Example:
        >>> # example for a two outcomes
        >>> weights = torch.tensor([0.75, 0.25])
        >>> linear_objective = LinearMCObjective(weights)
        >>> samples = sampler(posterior)
        >>> objective = linear_objective(samples)
    """

    def __init__(self, weights: Tensor) -> None:
        r"""Linear Objective.

        Args:
            weights: A one-dimensional tensor with `o` elements representing the
                linear weights on the outputs.
        """
        super().__init__()
        if weights.dim() != 1:
            raise ValueError("weights must be a one-dimensional tensor.")
        self.register_buffer("weights", weights)

    def forward(self, samples: Tensor) -> Tensor:
        r"""Evaluate the linear objective on the samples.

        Args:
            samples: A `sample_shape x batch_shape x q x o`-dim tensors of
                samples from a model posterior.

        Returns:
            A `sample_shape x batch_shape x q`-dim tensor of objective values.
        """
        if samples.shape[-1] != self.weights.shape[-1]:
            raise RuntimeError("Output shape of samples not equal to that of weights")
        return (samples * self.weights).sum(dim=-1)


class GenericMCObjective(MCAcquisitionObjective):
    r"""Objective generated from a generic callable.

    Allows to construct arbitrary MC-objective functions from a generic
    callable. In order to be able to use gradient-based acquisition function
    optimization it should be possible to backpropagate through the callable.

    Example:
        >>> generic_objective = GenericMCObjective(lambda Y: torch.sqrt(Y).sum(dim=-1))
        >>> samples = sampler(posterior)
        >>> objective = generic_objective(samples)
    """

    def __init__(self, objective: Callable[[Tensor], Tensor]) -> None:
        r"""Objective generated from a generic callable.

        Args:
            objective: A callable mapping a `sample_shape x batch-shape x q x o`-
                dim Tensor to a `sample_shape x batch-shape x q`-dim Tensor of
                objective values.
        """
        super().__init__()
        self.objective = objective

    def forward(self, samples: Tensor) -> Tensor:
        r"""Evaluate the feasibility-weigthed objective on the samples.

        Args:
            samples: A `sample_shape x batch_shape x q x o`-dim Tensors of
                samples from a model posterior.

        Returns:
            A `sample_shape x batch_shape x q`-dim Tensor of objective values
            weighted by feasibility (assuming maximization).
        """
        return self.objective(samples)


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
        objective: Callable[[Tensor], Tensor],
        constraints: List[Callable[[Tensor], Tensor]],
        infeasible_cost: float = 0.0,
        eta: float = 1e-3,
    ) -> None:
        r"""Feasibility-weighted objective.

        Args:
            objective: A callable mapping a `sample_shape x batch-shape x q x o`-
                dim Tensor to a `sample_shape x batch-shape x q`-dim Tensor of
                objective values.
            constraints: A list of callables, each mapping a Tensor of dimension
                `sample_shape x batch-shape x q x o` to a Tensor of dimension
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
        self.register_buffer("infeasible_cost", torch.tensor(infeasible_cost))

    def forward(self, samples: Tensor) -> Tensor:
        r"""Evaluate the feasibility-weighted objective on the samples.

        Args:
            samples: A `sample_shape x batch_shape x q x o`-dim Tensors of
                samples from a model posterior.

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
