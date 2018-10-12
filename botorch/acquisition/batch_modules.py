#!/usr/bin/env python3

from typing import Callable, List, Optional

import torch
from gpytorch import Module

from .functional import (
    batch_expected_improvement,
    batch_probability_of_improvement,
    batch_simple_regret,
    batch_upper_confidence_bound,
)
from .modules import AcquisitionFunction


"""
Wraps the batch acquisition functions defined in botorch.acquisition.functional
into BatchAcquisitionFunction gpytorch modules.
"""


class BatchAcquisitionFunction(AcquisitionFunction):
    def forward(self, candidate_set: torch.Tensor) -> torch.Tensor:
        """Takes in a `b x q x d` candidate_set Tensor of `b` t-batches with `q`
        `d`-dimensional design points each, and returns a one-dimensional Tensor
        with `b` elements."""
        raise NotImplementedError("BatchAcquisitionFunction cannot be used directly")


class qExpectedImprovement(BatchAcquisitionFunction):
    """TODO"""

    def __init__(
        self,
        model: Module,
        best_f: float,
        objective: Callable[[torch.Tensor], torch.Tensor] = lambda Y: Y,
        constraints: Optional[List[Callable[[torch.Tensor], torch.Tensor]]] = None,
        mc_samples: int = 5000,
    ) -> None:
        super(qExpectedImprovement, self).__init__(model)
        self.best_f = best_f
        self.objective = objective
        self.constraints = constraints
        self.mc_samples = mc_samples

    def forward(self, candidate_set: torch.Tensor) -> torch.Tensor:
        return batch_expected_improvement(
            X=candidate_set,
            model=self.model,
            best_f=self.best_f,
            objective=self.objective,
            constraints=self.constraints,
            mc_samples=self.mc_samples,
        )


class qProbabilityOfImprovement(BatchAcquisitionFunction):
    """TODO"""

    def __init__(self, model: Module, best_f: float, mc_samples: int = 5000) -> None:
        super(qProbabilityOfImprovement, self).__init__(model)
        self.best_f = best_f
        self.mc_samples = mc_samples

    def forward(self, candidate_set: torch.Tensor) -> torch.Tensor:
        return batch_probability_of_improvement(
            X=candidate_set,
            model=self.model,
            best_f=self.best_f,
            mc_samples=self.mc_samples,
        )


class qUpperConfidenceBound(BatchAcquisitionFunction):
    """TODO"""

    def __init__(self, model: Module, beta: float, mc_samples: int = 5000) -> None:
        super(qUpperConfidenceBound, self).__init__(model)
        self.beta = beta
        self.mc_samples = mc_samples

    def forward(self, candidate_set: torch.Tensor) -> torch.Tensor:
        return batch_upper_confidence_bound(
            X=candidate_set,
            model=self.model,
            beta=self.beta,
            mc_samples=self.mc_samples,
        )


class qSimpleRegret(BatchAcquisitionFunction):
    """TODO"""

    def __init__(self, model: Module, mc_samples: int = 5000) -> None:
        super(qSimpleRegret, self).__init__(model)
        self.mc_samples = mc_samples

    def forward(self, candidate_set: torch.Tensor) -> torch.Tensor:
        return batch_simple_regret(
            X=candidate_set, model=self.model, mc_samples=self.mc_samples
        )
