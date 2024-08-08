#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Deterministic Models: Simple wrappers that allow the usage of deterministic
mappings via the BoTorch Model and Posterior APIs.

Deterministic models are useful for expressing known input-output relationships
within the BoTorch Model API. This is useful e.g. for multi-objective
optimization with known objective functions (e.g. the number of parameters of a
Neural Network in the context of Neural Architecture Search is usually a known
function of the architecture configuration), or to encode cost functions for
cost-aware acquisition utilities. Cost-aware optimization is desirable when
evaluations have a cost that is heterogeneous, either in the inputs `X` or in a
particular fidelity parameter that directly encodes the fidelity of the
observation. `GenericDeterministicModel` supports arbitrary deterministic
functions, while `AffineFidelityCostModel` is a particular cost model for
multi-fidelity optimization. Other use cases of deterministic models include
representing approximate GP sample paths, e.g. Matheron paths obtained
with `get_matheron_path_model`, which allows them to be substituted in acquisition
functions or in other places where a `Model` is expected.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Callable, Optional, Union

import torch
from botorch.models.ensemble import EnsembleModel
from botorch.models.model import Model
from torch import Tensor


class DeterministicModel(EnsembleModel):
    """Abstract base class for deterministic models."""

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

    def _forward(self, X: Tensor) -> Tensor:
        r"""Compatibilizes the `DeterministicModel` with `EnsemblePosterior`"""
        return self.forward(X=X).unsqueeze(-3)


class GenericDeterministicModel(DeterministicModel):
    r"""A generic deterministic model constructed from a callable.

    Example:
        >>> f = lambda x: x.sum(dim=-1, keep_dims=True)
        >>> model = GenericDeterministicModel(f)
    """

    def __init__(self, f: Callable[[Tensor], Tensor], num_outputs: int = 1) -> None:
        r"""
        Args:
            f: A callable mapping a `batch_shape x n x d`-dim input tensor `X`
                to a `batch_shape x n x m`-dimensional output tensor (the
                outcome dimension `m` must be explicit, even if `m=1`).
            num_outputs: The number of outputs `m`.
        """
        super().__init__()
        self._f = f
        self._num_outputs = num_outputs

    def subset_output(self, idcs: list[int]) -> GenericDeterministicModel:
        r"""Subset the model along the output dimension.

        Args:
            idcs: The output indices to subset the model to.

        Returns:
            The current model, subset to the specified output indices.
        """

        def f_subset(X: Tensor) -> Tensor:
            return self._f(X)[..., idcs]

        return self.__class__(f=f_subset, num_outputs=len(idcs))

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
        self._num_outputs = a.size(-1)

    def subset_output(self, idcs: list[int]) -> AffineDeterministicModel:
        r"""Subset the model along the output dimension.

        Args:
            idcs: The output indices to subset the model to.

        Returns:
            The current model, subset to the specified output indices.
        """
        a_sub = self.a.detach()[..., idcs].clone()
        b_sub = self.b.detach()[..., idcs].clone()
        return self.__class__(a=a_sub, b=b_sub)

    def forward(self, X: Tensor) -> Tensor:
        return self.b + torch.einsum("...d,dm", X, self.a)


class PosteriorMeanModel(DeterministicModel):
    """A deterministic model that always returns the posterior mean."""

    def __init__(self, model: Model) -> None:
        r"""
        Args:
            model: The base model.
        """
        super().__init__()
        self.model = model

    def forward(self, X: Tensor) -> Tensor:
        return self.model.posterior(X).mean


class FixedSingleSampleModel(DeterministicModel):
    r"""
    A deterministic model defined by a single sample `w`.

    Given a base model `f` and a fixed sample `w`, the model always outputs

        y = f_mean(x) + f_stddev(x) * w

    We assume the outcomes are uncorrelated here.
    """

    def __init__(
        self,
        model: Model,
        w: Optional[Tensor] = None,
        dim: Optional[int] = None,
        jitter: Optional[float] = 1e-8,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.dtype] = None,
    ) -> None:
        r"""
        Args:
            model: The base model.
            w: A 1-d tensor with length model.num_outputs.
                If None, draw it from a standard normal distribution.
            dim: dimensionality of w.
                If None and w is not provided, draw w samples of size model.num_outputs.
            jitter: jitter value to be added for numerical stability, 1e-8 by default.
            dtype: dtype for w if specified
            device: device for w if specified
        """
        super().__init__()
        self.model = model
        self._num_outputs = model.num_outputs
        self.jitter = jitter
        if w is None:
            self.w = (
                torch.randn(model.num_outputs, dtype=dtype, device=device)
                if dim is None
                else torch.randn(dim, dtype=dtype, device=device)
            )
        else:
            self.w = w

    def forward(self, X: Tensor) -> Tensor:
        post = self.model.posterior(X)
        return post.mean + torch.sqrt(post.variance + self.jitter) * self.w.to(X)
