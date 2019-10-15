#! /usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

r"""
Deterministic Models. Simple wrappers that allow the usage of deterministic
mappings via the BoTorch Model and Posterior APIs. Useful e.g. for defining
known cost functions for cost-aware acquisition utilities.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional, Union

import torch
from torch import Tensor

from ..exceptions.errors import UnsupportedError
from ..posteriors.deterministic import DeterministicPosterior
from .model import Model


class DeterministicModel(Model, ABC):
    r"""Abstract base class for deterministic models."""

    @abstractmethod
    def forward(self, X: Tensor) -> Tensor:
        r"""Compute the (deterministic) model output at X.

        Args:
            X: A `batch_shape x n x d`-dim input tensor `X`.

        Returns:
            A `batch_shape x n x m`-dimensional output tensor (the outcome
            dimension `m` must be explicit if `m=1`).
        """
        pass  # pragma: no cover

    def posterior(
        self, X: Tensor, output_indices: Optional[List[int]] = None, **kwargs: Any
    ) -> DeterministicPosterior:
        r"""Compute the (deterministic) posterior at X."""
        if kwargs.get("observation_noise") is not None:
            # TODO: Consider returning an MVN here instead
            raise UnsupportedError(
                "Deterministic models do not support observation noise."
            )
        values = self.forward(X)
        if output_indices is not None:
            values = values[..., output_indices]
        return DeterministicPosterior(values=values)


class GenericDeterministicModel(DeterministicModel):
    r"""A generic deterministic model constructed from a callable."""

    def __init__(self, f: Callable[[Tensor], Tensor]) -> None:
        r"""A generic deterministic model constructed from a callable.

        Args:
            f: A callable mapping a `batch_shape x n x d`-dim input tensor `X`
                to a `batch_shape x n x m`-dimensional output tensor (the
                outcome dimension `m` must be explicit, even if `m=1`).
        """
        super().__init__()
        self._f = f

    def forward(self, X: Tensor) -> Tensor:
        r"""Compute the (deterministic) model output at X.

        Args:
            X: A `batch_shape x n x d`-dim input tensor `X`.

        Returns:
            A `batch_shape x n x m`-dimensional output tensor.
        """
        return self._f(X)


class AffineDeterministicModel(DeterministicModel):
    r"""An affine deterministic model."""

    def __init__(self, a: Tensor, b: Union[Tensor, float] = 0.01) -> None:
        r"""Affine deterministic model from weights and offset terms.

        A simple model of the form

            y[..., m] = b[m] + sum_{i=1}^d a[i, m] * X[..., i]

        Args:
            a: A `d x m`-dim tensor of linear weights, where `m` is the number
                of outputs (must be explicit if `m=1`)
            b: The affine (offset) term. Either a float (for single-output
                models or if the offset is shared), or a `m`-dim tensor (with
                different offset values for for the `m` different outputs).
        """
        if not a.ndim == 2:
            raise ValueError("a must be two-dimensional")
        if not torch.is_tensor(b):
            b = torch.tensor([b])
        if not b.ndim == 1:
            raise ValueError("b nust be one-dimensional")
        super().__init__()
        self.register_buffer("a", a)
        self.register_buffer("b", b.expand(a.size(-1)))

    def forward(self, X: Tensor) -> Tensor:
        return self.b + torch.einsum("...d,dm", X, self.a)
