#!/usr/bin/env python3

"""
Objective Modules
"""

from abc import ABC, abstractmethod
from typing import Callable, List

import torch
from torch import Tensor
from torch.nn import Module


class MCAcquisitionObjective(Module, ABC):
    """Abstract base class for MC-based objectives."""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, samples: Tensor) -> Tensor:
        """Evaluate the objective on the samples.

        Args:
            samples: A `sample_shape x batch_shape x q x t`-dim Tensors of
                samples from a model posterior.

        Returns:
            Tensor: A `sample_shape x batch_shape x q`-dim Tensor of objective
                values (assuming maximization).
        """
        pass


class GenericMCObjective(MCAcquisitionObjective):
    """Objective generated from a generic callable.

    TODO: description + math
    """

    def __init__(self, objective: Callable[[Tensor], Tensor]) -> None:
        """Objective generated from a generic callable.

        Args:
            objective: A callable mapping a `sample_shape x batch-shape x q x t`-
                dim Tensor to a `sample_shape x batch-shape x q`-dim Tensor of
                objective values.
        """
        super().__init__()
        self.objective = objective

    def forward(self, samples: Tensor) -> Tensor:
        """Evaluate the feasibility-weigthed objective on the samples.

        Args:
            samples: A `sample_shape x batch_shape x q x t`-dim Tensors of
                samples from a model posterior.

        Returns:
            Tensor: A `sample_shape x batch_shape x q`-dim Tensor of objective
                values weighted by feasibility (assuming maximization).
        """
        return self.objective(samples)


class ConstrainedMCObjective(GenericMCObjective):
    """Feasibility-weighted objective.

    TODO: description + math
    """

    def __init__(
        self,
        objective: Callable[[Tensor], Tensor],
        constraints: List[Callable[[Tensor], Tensor]],
        infeasible_cost: float = 0.0,
    ) -> None:
        """Feasibility-weighted objective.

        Args:
            objective: A callable mapping a `sample_shape x batch-shape x q x t`-
                dim Tensor to a `sample_shape x batch-shape x q`-dim Tensor of
                objective values.
            constraints: A list of callables, each mapping a Tensor of dimension
                `sample_shape x batch-shape x q x t` to a Tensor of dimension
                `sample_shape x batch-shape x q`, where negative values imply
                feasibility.
            infeasible_cost: The cost of a design if all associated samples are
                infeasible.
        """
        super().__init__(objective=objective)
        self.constraints = constraints
        self.register_buffer("infeasible_cost", torch.tensor(infeasible_cost))

    def forward(self, samples: Tensor) -> Tensor:
        """Evaluate the feasibility-weigthed objective on the samples.

        Args:
            samples: A `sample_shape x batch_shape x q x t`-dim Tensors of
                samples from a model posterior.

        Returns:
            Tensor: A `sample_shape x batch_shape x q`-dim Tensor of objective
                values weighted by feasibility (assuming maximization).
        """
        obj = super().forward(samples=samples)
        # apply the constraints to the objective first; then, compare with
        # best_f (which could be -M if no feasible point has been found)
        # TODO (T41447357): Fix this so we can backprop
        all_con_feasible = torch.ones(obj.shape, dtype=torch.uint8, device=obj.device)
        for constraint in self.constraints:
            # con has dimensions n_samples x b x q
            con = constraint(samples)
            this_con_feasible = con <= 0
            all_con_feasible.mul_(this_con_feasible)
        obj[~all_con_feasible] = -self.infeasible_cost
        return obj


class IdentityMCObjective(MCAcquisitionObjective):
    """Trivial objective extracting the last dimension."""

    def forward(self, samples: Tensor) -> Tensor:
        return samples.squeeze(-1)


class LinearMCObjective(MCAcquisitionObjective):
    """Linear objective constructed from a weight tensor."""

    def __init__(self, weights: Tensor) -> None:
        """Linear Objective.

        Args:
            weights: A one-dimensional tensor with `t` elements representing the
                linear weights on the outputs.
        """
        super().__init__()
        if weights.dim() != 1:
            raise ValueError("weights must be a one-dimensional tensor.")
        self.register_buffer("weights", weights)

    def forward(self, samples: Tensor) -> Tensor:
        """Evaluate the linear objective on the samples.

        Args:
            samples: A `sample_shape x batch_shape x q x t`-dim Tensors of
                samples from a model posterior.

        Returns:
            Tensor: A `sample_shape x batch_shape x q`-dim Tensor of objective
                values.
        """
        if samples.shape[-1] != self.weights.shape[-1]:
            raise RuntimeError("Output shape of samples not equal to that of weights")
        return (samples * self.weights).sum(dim=-1)
