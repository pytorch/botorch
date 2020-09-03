#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Optional

import torch
from botorch.exceptions.errors import BotorchTensorDimensionError
from botorch.utils.sampling import draw_sobol_normal_samples
from botorch.utils.transforms import normalize_indices
from gpytorch.constraints import Interval
from gpytorch.module import Module as GPyTorchModule
from torch import Tensor, nn
from torch.nn import Module, ModuleDict


EPS = 1e-4


class InputTransform(Module, ABC):
    r"""Abstract base class for input transforms."""

    # a boolean indicating whether to apply the transform in eval() mode.
    transform_on_eval: bool

    def forward(self, X: Tensor) -> Tensor:
        r"""Transform the inputs to a model.

        Args:
            X: A `batch_shape x n x d`-dim tensor of inputs.

        Returns:
            A `batch_shape x n x d`-dim tensor of transformed inputs.
        """
        if self.training or self.transform_on_eval:
            return self.transform(X)
        return X

    @abstractmethod
    def transform(self, X: Tensor) -> Tensor:
        r"""Transform the inputs to a model.

        Args:
            X: A `batch_shape x n x d`-dim tensor of inputs.

        Returns:
            A `batch_shape x n x d`-dim tensor of transformed inputs.
        """
        pass

    def untransform(self, X: Tensor) -> Tensor:
        r"""Un-transform the inputs to a model.

        Args:
            X: A `batch_shape x n x d`-dim tensor of transformed inputs.

        Returns:
            A `batch_shape x n x d`-dim tensor of un-transformed inputs.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement the `untransform` method"
        )


class ChainedInputTransform(InputTransform, ModuleDict):
    r"""An input transform representing the chaining of individual transforms"""

    def __init__(
        self, transform_on_eval: bool = True, **transforms: InputTransform
    ) -> None:
        r"""Chaining of input transforms.

        Args:
            transform_on_eval: A boolean indicating whether to apply the
                transforms in eval() mode. Default: True
            transforms: The transforms to chain. Internally, the names of the
                kwargs are used as the keys for accessing the individual
                transforms on the module.
        """
        self.transform_on_eval = transform_on_eval
        super().__init__(transforms)

    def transform(self, X: Tensor) -> Tensor:
        r"""Transform the inputs to a model.

        Individual transforms are applied in sequence.

        Args:
            X: A `batch_shape x n x d`-dim tensor of inputs.

        Returns:
            A `batch_shape x n x d`-dim tensor of transformed inputs.
        """
        for tf in self.values():
            X = tf.forward(X)
        return X

    def untransform(self, X: Tensor) -> Tensor:
        r"""Un-transform the inputs to a model.

        Un-transforms of the individual transforms are applied in reverse sequence.

        Args:
            X: A `batch_shape x n x d`-dim tensor of transformed inputs.

        Returns:
            A `batch_shape x n x d`-dim tensor of un-transformed inputs.
        """
        for tf in reversed(self.values()):
            X = tf.untransform(X)
        return X


class Normalize(InputTransform):
    r"""Normalize the inputs to the unit cube.

    If no explicit bounds are provided this module is stateful: If in train mode,
    calling `forward` updates the module state (i.e. the normalizing bounds). If
    in eval mode, calling `forward` simply applies the normalization using the
    current module state.
    """

    def __init__(
        self,
        d: int,
        bounds: Optional[Tensor] = None,
        batch_shape: torch.Size = torch.Size(),  # noqa: B008
        transform_on_eval: bool = True,
    ) -> None:
        r"""Normalize the inputs to the unit cube.

        Args:
            d: The dimension of the input space.
            bounds: If provided, use these bounds to normalize the inputs. If
                omitted, learn the bounds in train mode.
            batch_shape: The batch shape of the inputs (asssuming input tensors
                of shape `batch_shape x n x d`). If provided, perform individual
                normalization per batch, otherwise uses a single normalization.
            transform_on_eval: A boolean indicating whether to apply the
                transform in eval() mode. Default: True
        """
        super().__init__()
        if bounds is not None:
            if bounds.size(-1) != d:
                raise BotorchTensorDimensionError(
                    "Incompatible dimensions of provided bounds"
                )
            mins = bounds[..., 0:1, :]
            ranges = bounds[..., 1:2, :] - mins
            self.learn_bounds = False
        else:
            mins = torch.zeros(*batch_shape, 1, d)
            ranges = torch.zeros(*batch_shape, 1, d)
            self.learn_bounds = True
        self.register_buffer("mins", mins)
        self.register_buffer("ranges", ranges)
        self._d = d
        self.transform_on_eval = transform_on_eval

    def transform(self, X: Tensor) -> Tensor:
        r"""Normalize the inputs.

        If no explicit bounds are provided, this is stateful: In train mode,
        calling `forward` updates the module state (i.e. the normalizing bounds).
        In eval mode, calling `forward` simply applies the normalization using
        the current module state.

        Args:
            X: A `batch_shape x n x d`-dim tensor of inputs.

        Returns:
            A `batch_shape x n x d`-dim tensor of inputs normalized to the
            module's bounds.
        """
        if self.learn_bounds and self.training:
            if X.size(-1) != self.mins.size(-1):
                raise BotorchTensorDimensionError(
                    f"Wrong input. dimension. Received {X.size(-1)}, "
                    f"expected {self.mins.size(-1)}"
                )
            self.mins = X.min(dim=-2, keepdim=True)[0]
            self.ranges = X.max(dim=-2, keepdim=True)[0] - self.mins
        return (X - self.mins) / self.ranges

    def untransform(self, X: Tensor) -> Tensor:
        r"""Un-normalize the inputs.

        Args:
            X: A `batch_shape x n x d`-dim tensor of normalized inputs.

        Returns:
            A `batch_shape x n x d`-dim tensor of un-normalized inputs.
        """
        return self.mins + X * self.ranges

    @property
    def bounds(self) -> Tensor:
        r"""The bounds used for normalizing the inputs."""
        return torch.cat([self.mins, self.mins + self.ranges], dim=-2)


@dataclass
class CategoricalSpec:
    idx: int
    num_categories: int


@dataclass
class LatentCategoricalSpec(CategoricalSpec):
    latent_dim: int = 2


class EmbeddingTransform(InputTransform, GPyTorchModule):
    r"""Abstract base class for Embedding-based transforms"""
    _emb_dim: int
    _transformed_dim: int
    dim: int
    categ_idcs: Tensor
    non_categ_mask: Tensor

    def transform(self, X: Tensor) -> Tensor:
        r"""Transform categorical variables to use one-hot representation."""
        X_emb = torch.empty(
            X.shape[:-1] + torch.Size([self._emb_dim]), dtype=X.dtype, device=X.device
        )
        start_idx = 0
        for idx in self.categ_idcs:
            emb_table = self.get_emb_table(idx)
            emb_dim = emb_table.shape[-1]
            end_idx = start_idx + emb_dim
            emb = emb_table.index_select(dim=0, index=X[..., idx].view(-1).long())
            X_emb[..., start_idx:end_idx] = emb.view(
                X_emb.shape[:-1] + torch.Size([emb_dim])
            )
            start_idx = end_idx
        return torch.cat([X[..., self.non_categ_mask], X_emb], dim=-1)

    def get_emb_table(self, idx: int) -> Tensor:
        r"""Get the embedding table for the specified categorical feature.

        Args:
            idx: The index of the categorical feature

        Returns:
            A `num_categories x emb_dim`-dim tensor containing the embeddings
            for each category.
        """
        pass

    def transform_bounds(self, bounds: Tensor) -> Tensor:
        r"""Update bounds based on embedding transform.

        Args:
            bounds: A `2 x d`-dim tensor of lower and upper bounds

        Returns:
            A x `2 x d_cont + d_emb`-dim tensor of lower and upper bounds
        """
        d_cont = self.dim - self.categ_idcs.shape[0]
        tf_bounds = torch.zeros(
            2, d_cont + self._emb_dim, dtype=bounds.dtype, device=bounds.device
        )
        tf_bounds[:, :d_cont] = bounds[:, self.non_categ_mask]
        tf_bounds[1, d_cont:] = 1
        return tf_bounds


class OneHot(EmbeddingTransform):
    def __init__(
        self,
        categorical_specs: List[CategoricalSpec],
        dim: int,
        transform_on_eval: bool = False,
    ) -> None:
        r"""Initialize input transform.

        Args:
            categorical_specs: A list of CategoricalSpec objects.
            dim: raw dimension
            transform_on_eval: A boolean indicating whether to apply the
                transform in eval() mode. Default: False
        """
        GPyTorchModule.__init__(self)
        self.dim = dim
        self.transform_on_eval = transform_on_eval
        self._emb_dim = 0
        categ_idcs = []
        for c in categorical_specs:
            nlzd_idx = normalize_indices([c.idx], dim)[0]
            categ_idcs.append(nlzd_idx)
            self._emb_dim += c.num_categories
            self.register_buffer(f"emb_tables_{nlzd_idx}", torch.eye(c.num_categories))
        self.register_buffer("categ_idcs", torch.tensor(categ_idcs, dtype=torch.long))
        non_categ_mask = torch.ones(dim, dtype=bool)
        non_categ_mask[self.categ_idcs] = 0
        self.register_buffer("non_categ_mask", non_categ_mask)

    def get_emb_table(self, idx: int) -> Tensor:
        r"""Get the embedding table for the specified categorical feature.

        Args:
            idx: The index of the categorical feature

        Returns:
            A `num_categories x emb_dim`-dim tensor containing the embeddings
            for each category.
        """
        return getattr(self, f"emb_tables_{idx}")

    def untransform(self, X: Tensor, **kwargs) -> Tensor:
        r"""Untransform X to represent categoricals as integers.

        The transformation assigns the category to be the index corresponding to the
        maximum value. Note: this is not differentiable.

        Args:
            X: A `batch_shape x n x d_cont + d_emb`-dim tensor of transformed valiues

        Returns:
            The untransformed tensor.
        """
        new_X = torch.empty(
            X.shape[:-1] + torch.Size([self.dim]), dtype=X.dtype, device=X.device
        )
        num_non_categ_features = X.shape[-1] - self._emb_dim
        new_X[..., self.non_categ_mask] = X[..., :num_non_categ_features]
        start_idx = self.dim - self.categ_idcs.shape[0]
        for idx in self.categ_idcs.tolist():
            emb_table = self.get_emb_table(idx)
            emb_dim = emb_table.shape[-1]
            end_idx = start_idx + emb_dim
            int_categories = X[..., start_idx:end_idx].argmax(dim=-1).to(dtype=X.dtype)
            new_X[..., idx] = int_categories
            start_idx = end_idx
        return new_X


class LatentCategoricalEmbedding(EmbeddingTransform):
    r"""Latent embeddings for categorical variables.

    Note: this current uses the same latent embeddings across batched.
    This means that a batched multi-output model will use the same latent
    embeddings for all outputs.
    """

    def __init__(
        self,
        categorical_specs: List[LatentCategoricalSpec],
        dim: int,
        transform_on_eval: bool = False,
    ) -> None:
        r"""Initialize input transform.

        Args:
            categorical_specs: A list of LatentCategoricalSpec objects.
            dim: the total dimension of the inputs.
            transform_on_eval: A boolean indicating whether to apply the
                transform in eval() mode. Default: False
        """
        GPyTorchModule.__init__(self)
        self.dim = dim
        self.transform_on_eval = transform_on_eval
        self._emb_dim = 0
        categ_idcs = []
        # TODO: replace with ParameterDict when supported in GPyTorch
        for c in categorical_specs:
            nlzd_idx = normalize_indices([c.idx], dim)[0]
            categ_idcs.append(nlzd_idx)

            init_bounds = torch.full((2, c.latent_dim), EPS)
            init_bounds[1] = 1 - EPS
            # initialize embeddings with sobol points in [0.25, 0.75]^d_latent
            init_emb_table = draw_sobol_normal_samples(n=c.num_categories, d=c.latent_dim).squeeze(
                0
            )
            constraint = Interval(lower_bound=EPS, upper_bound=1.0 - EPS)
            self.register_parameter(
                f"raw_latent_emb_tables_{nlzd_idx}",
                # nn.Parameter(constraint.inverse_transform(init_emb_table)),
                nn.Parameter(init_emb_table),
            )
            self.register_constraint(
                param_name=f"raw_latent_emb_tables_{nlzd_idx}", constraint=constraint
            )
            self._emb_dim += c.latent_dim
        self.register_buffer("categ_idcs", torch.tensor(categ_idcs, dtype=torch.long))
        non_categ_mask = torch.ones(dim, dtype=bool)
        non_categ_mask[self.categ_idcs] = 0
        self.register_buffer("non_categ_mask", non_categ_mask)

    def get_emb_table(self, idx: int) -> Tensor:
        r"""Get the embedding table for the specified categorical feature.

        Args:
            idx: The index of the categorical feature

        Returns:
            A `num_categories x latent_dim`-dim tensor containing the embeddings
            for each category.
        """
        constraint = getattr(self, f"raw_latent_emb_tables_{idx}_constraint")
        raw_emb_table = getattr(self, f"raw_latent_emb_tables_{idx}")
        return constraint.transform(raw_emb_table)

    def untransform(
        self,
        X: Tensor,
        dist_func: Optional[Callable[[Tensor, Tensor, int], Tensor]] = None,
    ) -> Tensor:
        r"""Untransform X to represent categoricals as integers.

        The transformation assigns the category to be the index corresponding to the
        closest embedding. Note: this is not differentiable.

        Args:
            X: A `batch_shape x n x d_cont + d_latent`-dim tensor of transformed valiues
            dist_func: A broadcastable distance function mapping a two input tensors with
                shapes `batch_shape x n x 1 x d_latent` and `n_categories x d_latent` and
                an integer starting index to to a `batch_shape x n x n_categories`-dim
                tensor of distances. The default is L2 distance.

        Returns:
            The untransformed tensor.
        """
        new_X = torch.empty(
            X.shape[:-1] + torch.Size([self.dim]), dtype=X.dtype, device=X.device
        )
        num_non_categ_features = X.shape[-1] - self._emb_dim
        new_X[..., self.non_categ_mask] = X[..., :num_non_categ_features]
        start_idx = self.dim - self.categ_idcs.shape[0]
        for idx in self.categ_idcs.tolist():
            emb_table = self.get_emb_table(idx)
            emb_dim = emb_table.shape[-1]
            end_idx = start_idx + emb_dim
            x = X[..., start_idx:end_idx].unsqueeze(-2)
            x_emb = emb_table.unsqueeze(-3)
            if dist_func is not None:
                dist = dist_func(x, x_emb, start_idx)
            else:
                dist = torch.norm(x - x_emb, dim=-1)
            int_categories = dist.argmin(dim=-1).to(dtype=X.dtype)
            new_X[..., idx] = int_categories
            start_idx = end_idx
        return new_X
