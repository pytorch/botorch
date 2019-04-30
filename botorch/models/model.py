#! /usr/bin/env python3

r"""
Abstract base module for all BoTorch models.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional

from torch import Tensor
from torch.nn import Module

from ..posteriors import Posterior


class Model(Module, ABC):
    r"""Abstract base class for BoTorch models."""

    @abstractmethod
    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: bool = False,
        **kwargs: Any,
    ) -> Posterior:
        r"""Computes the posterior over model outputs at the provided points.

        Args:
            X: A `b x q x d`-dim Tensor, where `d` is the dimension of the
                feature space, `q` is the number of points considered jointly,
                and `b` is the batch dimension.
            output_indices: A list of indices, corresponding to the outputs over
                which to compute the posterior (if the model is multi-output).
                Can be used to speed up computation if only a subset of the
                model's outputs are required for optimization. If omitted,
                computes the posterior over all model outputs.
            observation_noise: If True, add observation noise to the posterior.

        Returns:
            A `Posterior` object, representing a batch of `b` joint distributions
            over `q` points and `o` outputs each.
        """
        pass  # pragma: no cover
