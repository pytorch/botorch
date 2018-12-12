#!/usr/bin/env python3

from abc import ABC, abstractmethod
from typing import Union

from torch import Tensor
from torch.nn import Module

from ..models.model import Model
from .functional.acquisition import (
    expected_improvement,
    max_value_entropy_search,
    posterior_mean,
    probability_of_improvement,
    upper_confidence_bound,
)


class AcquisitionFunction(Module, ABC):
    """Abstract module class for wrapping acquisition functions"""

    def __init__(self, model: Model) -> None:
        super().__init__()
        self.model = model

    @abstractmethod
    def forward(self, X: Tensor) -> Tensor:
        """Evaluate the acquisition function on candidate set X.

        Args:
            X: Either a `n x d` or a `b x n x d` (batch mode) Tensor of `n`
                individual design points in `d` dimensions each (if operating in
                batch mode, AcquisitionFunction must be instantiated with a model
                in batch mode). Points are evaluated independently (i.e. covariance
                across the different points is not considered).

        Returns:
            A one-dim tensor with `n` elements or a `b x n` tensor (batch mode)
                corresponding to the acquisition function values of the respective
                design points.
        """
        pass

    def extract_candidates(self, X: Tensor) -> Tensor:
        """Perform any final operations on the candidate set X post-optimization of the
            acquisition function.

        Args:
            X: optimized `b x q x d` Tensor or points

        Returns:
            Tensor created from X
        """
        return X


class ExpectedImprovement(AcquisitionFunction):
    """Single-outcome expected improvement (assumes maximization)

    Args:
        model: A fitted single-outcome GP model (must be in batch mode if
            candidate sets X will be)
        best_f: Either a scalar or a one-dim tensor with `b` elements (batch mode)
            representing the best function value observed so far (assumed noiseless)
    """

    def __init__(self, model: Model, best_f: Union[float, Tensor]) -> None:
        super().__init__(model)
        self.best_f = best_f

    def forward(self, X: Tensor) -> Tensor:
        return expected_improvement(X=X, model=self.model, best_f=self.best_f)


class PosteriorMean(AcquisitionFunction):
    """Single-outcome posterior mean

    Args:
        model: A fitted single-outcome GP model (must be in batch mode if
            candidate sets X will be)
    """

    def forward(self, X: Tensor) -> Tensor:
        return posterior_mean(X=X, model=self.model)


class ProbabilityOfImprovement(AcquisitionFunction):
    """Single-outcome probability of improvement (assumes maximization)

    Args:
        model: A fitted single-outcome GP model (must be in batch mode if
            candidate sets X will be)
        best_f: Either a scalar or a one-dim tensor with `b` elements (batch mode)
            representing the best function value observed so far (assumed noiseless)
    """

    def __init__(self, model: Model, best_f: Union[float, Tensor]) -> None:
        super().__init__(model)
        self.best_f = best_f

    def forward(self, X: Tensor) -> Tensor:
        return probability_of_improvement(X=X, model=self.model, best_f=self.best_f)


class UpperConfidenceBound(AcquisitionFunction):
    """Single-outcome probability of improvement (assumes maximization)

    Args:
        model: A fitted single-outcome GP model (must be in batch mode if
            candidate sets X will be)
        beta: Either a scalar or a one-dim tensor with `b` elements (batch mode)
            representing the trade-off parameter between mean and covariance
    """

    def __init__(self, model: Model, beta: Union[float, Tensor]) -> None:
        super().__init__(model)
        self.beta = beta

    def forward(self, X: Tensor) -> Tensor:
        return upper_confidence_bound(X=X, model=self.model, beta=self.beta)


class MaxValueEntropySearch(AcquisitionFunction):
    """NOT YET IMPLEMENTED"""

    def __init__(self, model: Model, num_samples: int) -> None:
        super().__init__(model)
        self.num_samples = num_samples

    def forward(self, X: Tensor) -> Tensor:
        return max_value_entropy_search(
            X=X, model=self.model, num_samples=self.num_samples
        )
