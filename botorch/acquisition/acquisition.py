#!/usr/bin/env python3

"""
Abstract base module for all botorch acquisition functions.
"""

from abc import ABC, abstractmethod

from torch import Tensor
from torch.nn import Module

from ..models.model import Model


class AcquisitionFunction(Module, ABC):
    """Abstract base class for acquisition functions."""

    def __init__(self, model: Model) -> None:
        """Constructor for the AcquisitionFunction base class.

        Args:
            model: A fitted model.
        """
        super().__init__()
        self.add_module("model", model)

    @abstractmethod
    def forward(self, X: Tensor) -> Tensor:
        """Evaluate the acquisition function on the candidate set X.

        Args:
            X: A `(b) x q x d`-dim Tensor of `(b)` t-batches with `q` `d`-dim
                design points each.

        Returns:
            A `(b)`-dim Tensor of acquisition funciton values at the given
                design points `X`.
        """
        pass
