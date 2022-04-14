#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Input Transformations.

These classes implement a variety of transformations for
input parameters including: learned input warping functions,
rounding functions, and log transformations. The input transformation
is typically part of a Model and applied within the model.forward()
method.

"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import List, Optional, Union

import torch
from botorch.exceptions.errors import BotorchTensorDimensionError
from botorch.models.transforms.utils import expand_and_copy_tensor
from botorch.utils.rounding import approximate_round
from gpytorch import Module as GPyTorchModule
from gpytorch.constraints import GreaterThan
from gpytorch.priors import Prior
from torch import nn, Tensor
from torch.distributions import Kumaraswamy
from torch.nn import Module, ModuleDict


class InputTransform(ABC):
    r"""Abstract base class for input transforms.

    Note: Input transforms must inherit from `torch.nn.Module`. This
        is deferred to the subclasses to avoid any potential conflict
        between `gpytorch.module.Module` and `torch.nn.Module` in `Warp`.
    """

    def forward(self, X: Tensor) -> Tensor:
        r"""Transform the inputs to a model.

        Args:
            X: A `batch_shape x n x d`-dim tensor of inputs.

        Returns:
            A `batch_shape x n x d`-dim tensor of transformed inputs.
        """
        return self.transform(X)

    @abstractmethod
    def transform(self, X: Tensor) -> Tensor:
        r"""Transform the inputs to a model.

        Args:
            X: A `batch_shape x n x d`-dim tensor of inputs.

        Returns:
            A `batch_shape x n x d`-dim tensor of transformed inputs.
        """
        pass  # pragma: no cover

    def untransform(self, X: Tensor) -> Tensor:
        r"""Un-transform the inputs to a model.

        Args:
            X: A `batch_shape x n x d`-dim tensor of transformed inputs.

        Returns:
            A `batch_shape x n x d`-dim tensor of un-transformed inputs.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement the `untransform` method."
        )

    def equals(self, other: InputTransform) -> bool:
        r"""Check if another input transform is equivalent.

        Note: The reason that a custom equals method is defined rather than
        defining an __eq__ method is because defining an __eq__ method sets
        the __hash__ method to None. Hashing modules is currently used in
        pytorch. See https://github.com/pytorch/pytorch/issues/7733.

        Args:
            other: Another input transform.

        Returns:
            A boolean indicating if the other transform is equivalent.
        """
        other_state_dict = other.state_dict()
        return type(self) == type(other) and all(
            torch.allclose(v, other_state_dict[k].to(v))
            for k, v in self.state_dict().items()
        )


class ChainedInputTransform(InputTransform, ModuleDict):
    r"""An input transform representing the chaining of individual transforms."""

    def __init__(self, **transforms: InputTransform) -> None:
        r"""Chaining of input transforms.

        Args:
            transforms: The transforms to chain. Internally, the names of the
                kwargs are used as the keys for accessing the individual
                transforms on the module.

        Example:
            >>> tf1 = Normalize(d=2)
            >>> tf2 = Normalize(d=2)
            >>> tf = ChainedInputTransform(tf1=tf1, tf2=tf2)
            >>> list(tf.keys())
            ['tf1', 'tf2']
            >>> tf["tf1"]
            Normalize()

        """
        super().__init__(OrderedDict(transforms))

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

    def equals(self, other: InputTransform) -> bool:
        r"""Check if another input transform is equivalent.

        Args:
            other: Another input transform.

        Returns:
            A boolean indicating if the other transform is equivalent.
        """
        return super().equals(other=other) and all(
            t1 == t2 for t1, t2 in zip(self.values(), other.values())
        )


class ReversibleInputTransform(InputTransform, ABC):
    r"""An abstract class for a reversible input transform.

    Properties:
        reverse: A boolean indicating if the functionality of transform
            and untransform methods should be swapped.
    """
    reverse: bool

    def transform(self, X: Tensor) -> Tensor:
        r"""Transform the inputs.

        Args:
            X: A `batch_shape x n x d`-dim tensor of inputs.

        Returns:
            A `batch_shape x n x d`-dim tensor of transformed inputs.
        """
        return self._untransform(X) if self.reverse else self._transform(X)

    def untransform(self, X: Tensor) -> Tensor:
        r"""Un-transform the inputs.

        Args:
            X: A `batch_shape x n x d`-dim tensor of inputs.

        Returns:
            A `batch_shape x n x d`-dim tensor of un-transformed inputs.
        """
        return self._transform(X) if self.reverse else self._untransform(X)

    @abstractmethod
    def _transform(self, X: Tensor) -> Tensor:
        r"""Forward transform the inputs.

        Args:
            X: A `batch_shape x n x d`-dim tensor of inputs.

        Returns:
            A `batch_shape x n x d`-dim tensor of transformed inputs.
        """
        pass  # pragma: no cover

    @abstractmethod
    def _untransform(self, X: Tensor) -> Tensor:
        r"""Reverse transform the inputs.

        Args:
            X: A `batch_shape x n x d`-dim tensor of inputs.

        Returns:
            A `batch_shape x n x d`-dim tensor of transformed inputs.
        """
        pass  # pragma: no cover

    def equals(self, other: InputTransform) -> bool:
        r"""Check if another input transform is equivalent.

        Args:
            other: Another input transform.

        Returns:
            A boolean indicating if the other transform is equivalent.
        """
        return super().equals(other=other) and (self.reverse == other.reverse)


class Normalize(ReversibleInputTransform, Module):
    r"""Normalize the inputs to the unit cube.

    If no explicit bounds are provided this module is stateful: If in train mode,
    calling `forward` updates the module state (i.e. the normalizing bounds). If
    in eval mode, calling `forward` simply applies the normalization using the
    current module state.
    """

    def __init__(
        self,
        d: int,
        indices: Optional[List[int]] = None,
        bounds: Optional[Tensor] = None,
        batch_shape: torch.Size = torch.Size(),  # noqa: B008
        reverse: bool = False,
        min_range: float = 1e-8,
    ) -> None:
        r"""Normalize the inputs to the unit cube.

        Args:
            d: The dimension of the input space.
            indices: The indices of the inputs to normalize. If omitted,
                take all dimensions of the inputs into account.
            bounds: If provided, use these bounds to normalize the inputs. If
                omitted, learn the bounds in train mode.
            batch_shape: The batch shape of the inputs (asssuming input tensors
                of shape `batch_shape x n x d`). If provided, perform individual
                normalization per batch, otherwise uses a single normalization.
            reverse: A boolean indicating whether the forward pass should untransform
                the inputs.
            min_range: Amount of noise to add to the range to ensure no division by
                zero errors.
        """
        super().__init__()
        if (indices is not None) and (len(indices) == 0):
            raise ValueError("`indices` list is empty!")
        if (indices is not None) and (len(indices) > 0):
            indices = torch.tensor(indices, dtype=torch.long)
            if len(indices) > d:
                raise ValueError("Can provide at most `d` indices!")
            if (indices > d - 1).any():
                raise ValueError("Elements of `indices` have to be smaller than `d`!")
            if len(indices.unique()) != len(indices):
                raise ValueError("Elements of `indices` tensor must be unique!")
            self.indices = indices
        if bounds is not None:
            if bounds.size(-1) != d:
                raise BotorchTensorDimensionError(
                    "Dimensions of provided `bounds` are incompatible with `d`!"
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
        self.reverse = reverse
        self.batch_shape = batch_shape
        self.min_range = min_range

    def _transform(self, X: Tensor) -> Tensor:
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
                    f"Wrong input dimension. Received {X.size(-1)}, "
                    f"expected {self.mins.size(-1)}."
                )
            self.mins = X.min(dim=-2, keepdim=True)[0]
            ranges = X.max(dim=-2, keepdim=True)[0] - self.mins
            ranges[torch.where(ranges <= self.min_range)] = self.min_range
            self.ranges = ranges
        if hasattr(self, "indices"):
            X_new = X.clone()
            X_new[..., self.indices] = (
                X_new[..., self.indices] - self.mins[..., self.indices]
            ) / self.ranges[..., self.indices]
            return X_new
        return (X - self.mins) / self.ranges

    def _untransform(self, X: Tensor) -> Tensor:
        r"""Un-normalize the inputs.

        Args:
            X: A `batch_shape x n x d`-dim tensor of normalized inputs.

        Returns:
            A `batch_shape x n x d`-dim tensor of un-normalized inputs.
        """
        if hasattr(self, "indices"):
            X_new = X.clone()
            X_new[..., self.indices] = (
                self.mins[..., self.indices]
                + X_new[..., self.indices] * self.ranges[..., self.indices]
            )
            return X_new
        return self.mins + X * self.ranges

    @property
    def bounds(self) -> Tensor:
        r"""The bounds used for normalizing the inputs."""
        return torch.cat([self.mins, self.mins + self.ranges], dim=-2)

    def equals(self, other: InputTransform) -> bool:
        r"""Check if another input transform is equivalent.

        Args:
            other: Another input transform.

        Returns:
            A boolean indicating if the other transform is equivalent.
        """
        if hasattr(self, "indices") == hasattr(other, "indices"):
            if hasattr(self, "indices"):
                return (
                    super().equals(other=other)
                    and (self._d == other._d)
                    and (self.learn_bounds == other.learn_bounds)
                    and (self.indices == other.indices).all()
                )
            else:
                return (
                    super().equals(other=other)
                    and (self._d == other._d)
                    and (self.learn_bounds == other.learn_bounds)
                )
        return False


class InputStandardize(ReversibleInputTransform, Module):
    r"""Standardize inputs (zero mean, unit variance).

    In train mode, calling `forward` updates the module state
    (i.e. the mean/std normalizing constants). If in eval mode, calling `forward`
    simply applies the standardization using the current module state.
    """

    def __init__(
        self,
        d: int,
        indices: Optional[List[int]] = None,
        batch_shape: torch.Size = torch.Size(),  # noqa: B008
        reverse: bool = False,
        min_std: float = 1e-8,
    ) -> None:
        r"""Standardize inputs (zero mean, unit variance).

        Args:
            d: The dimension of the input space.
            indices: The indices of the inputs to standardize. If omitted,
                take all dimensions of the inputs into account.
            batch_shape: The batch shape of the inputs (asssuming input tensors
                of shape `batch_shape x n x d`). If provided, perform individual
                normalization per batch, otherwise uses a single normalization.
            reverse: A boolean indicating whether the forward pass should untransform
                the inputs.
            min_std: Amount of noise to add to the standard deviation to ensure no
                division by zero errors.
        """
        super().__init__()
        if (indices is not None) and (len(indices) == 0):
            raise ValueError("`indices` list is empty!")
        if (indices is not None) and (len(indices) > 0):
            indices = torch.tensor(indices, dtype=torch.long)
            if len(indices) > d:
                raise ValueError("Can provide at most `d` indices!")
            if (indices > d - 1).any():
                raise ValueError("Elements of `indices` have to be smaller than `d`!")
            if len(indices.unique()) != len(indices):
                raise ValueError("Elements of `indices` tensor must be unique!")
            self.indices = indices
        self.register_buffer("means", torch.zeros(*batch_shape, 1, d))
        self.register_buffer("stds", torch.ones(*batch_shape, 1, d))
        self._d = d
        self.batch_shape = batch_shape
        self.min_std = min_std
        self.reverse = reverse
        self.learn_bounds = True

    def _transform(self, X: Tensor) -> Tensor:
        r"""Standardize the inputs.

        In train mode, calling `forward` updates the module state
        (i.e. the mean/std normalizing constants). If in eval mode, calling `forward`
        simply applies the standardization using the current module state.

        Args:
            X: A `batch_shape x n x d`-dim tensor of inputs.

        Returns:
            A `batch_shape x n x d`-dim tensor of inputs normalized to the
            module's bounds.
        """
        if self.training and self.learn_bounds:
            if X.size(-1) != self.means.size(-1):
                raise BotorchTensorDimensionError(
                    f"Wrong input. dimension. Received {X.size(-1)}, "
                    f"expected {self.means.size(-1)}"
                )
            self.means = X.mean(dim=-2, keepdim=True)
            self.stds = X.std(dim=-2, keepdim=True)

            self.stds = torch.clamp(self.stds, min=self.min_std)
        if hasattr(self, "indices"):
            X_new = X.clone()
            X_new[..., self.indices] = (
                X_new[..., self.indices] - self.means[..., self.indices]
            ) / self.stds[..., self.indices]
            return X_new
        return (X - self.means) / self.stds

    def _untransform(self, X: Tensor) -> Tensor:
        r"""Un-standardize the inputs.

        Args:
            X: A `batch_shape x n x d`-dim tensor of normalized inputs.

        Returns:
            A `batch_shape x n x d`-dim tensor of un-normalized inputs.
        """
        if hasattr(self, "indices"):
            X_new = X.clone()
            X_new[..., self.indices] = (
                self.means[..., self.indices]
                + X_new[..., self.indices] * self.stds[..., self.indices]
            )
            return X_new
        return self.means.to(X) + self.stds.to(X) * X

    def equals(self, other: InputTransform) -> bool:
        r"""Check if another input transform is equivalent.

        Args:
            other: Another input transform.

        Returns:
            A boolean indicating if the other transform is equivalent.
        """
        if hasattr(self, "indices") == hasattr(other, "indices"):
            if hasattr(self, "indices"):
                return (
                    super().equals(other=other)
                    and (self._d == other._d)
                    and (self.indices == other.indices).all()
                )
            else:
                return super().equals(other=other) and (self._d == other._d)
        return False


class Round(InputTransform, Module):
    r"""A rounding transformation for integer inputs.

    This will typically be used in conjunction with normalization as
    follows:

    In eval() mode (i.e. after training), the inputs pass
    would typically be normalized to the unit cube (e.g. during candidate
    optimization). 1. These are unnormalized back to the raw input space.
    2. The integers are rounded. 3. All values are normalized to the unit
    cube.

    In train() mode, the inputs can either (a) be normalized to the unit
    cube or (b) provided using their raw values. In the case of (a)
    transform_on_train should be set to True, so that the normalized inputs
    are unnormalized before rounding. In the case of (b) transform_on_train
    should be set to False, so that the raw inputs are rounded and then
    normalized to the unit cube.

    This transformation uses differentiable approximate rounding by default.
    The rounding function is approximated with a piece-wise function where
    each piece is a hyperbolic tangent function.

    Example:
        >>> unnormalize_tf = Normalize(
        >>>     d=d,
        >>>     bounds=bounds,
        >>>     transform_on_eval=True,
        >>>     transform_on_train=True,
        >>>     reverse=True,
        >>> )
        >>> round_tf = Round(integer_indices)
        >>> normalize_tf = Normalize(d=d, bounds=bounds)
        >>> tf = ChainedInputTransform(
        >>>     tf1=unnormalize_tf, tf2=round_tf, tf3=normalize_tf
        >>> )
    """

    def __init__(
        self,
        indices: List[int],
        approximate: bool = True,
        tau: float = 1e-3,
    ) -> None:
        r"""Initialize transform.

        Args:
            indices: The indices of the integer inputs.
            approximate: A boolean indicating whether approximate or exact
                rounding should be used. Default: approximate.
            tau: The temperature parameter for approximate rounding.
        """
        super().__init__()
        self.register_buffer("indices", torch.tensor(indices, dtype=torch.long))
        self.approximate = approximate
        self.tau = tau

    def transform(self, X: Tensor) -> Tensor:
        r"""Round the inputs.

        Args:
            X: A `batch_shape x n x d`-dim tensor of inputs.

        Returns:
            A `batch_shape x n x d`-dim tensor of rounded inputs.
        """
        X_rounded = X.clone()
        X_int = X_rounded[..., self.indices]
        if self.approximate:
            X_int = approximate_round(X_int, tau=self.tau)
        else:
            X_int = X_int.round()
        X_rounded[..., self.indices] = X_int
        return X_rounded

    def equals(self, other: InputTransform) -> bool:
        r"""Check if another input transform is equivalent.

        Args:
            other: Another input transform.

        Returns:
            A boolean indicating if the other transform is equivalent.
        """
        return (
            super().equals(other=other)
            and self.approximate == other.approximate
            and self.tau == other.tau
        )


class Log10(ReversibleInputTransform, Module):
    r"""A base-10 log transformation."""

    def __init__(
        self,
        indices: List[int],
        reverse: bool = False,
    ) -> None:
        r"""Initialize transform.

        Args:
            indices: The indices of the inputs to log transform.
            reverse: A boolean indicating whether the forward pass should untransform
                the inputs.
        """
        super().__init__()
        self.register_buffer("indices", torch.tensor(indices, dtype=torch.long))
        self.reverse = reverse

    def _transform(self, X: Tensor) -> Tensor:
        r"""Log transform the inputs.

        Args:
            X: A `batch_shape x n x d`-dim tensor of inputs.

        Returns:
            A `batch_shape x n x d`-dim tensor of transformed inputs.
        """
        X_new = X.clone()
        X_new[..., self.indices] = X_new[..., self.indices].log10()
        return X_new

    def _untransform(self, X: Tensor) -> Tensor:
        r"""Reverse the log transformation.

        Args:
            X: A `batch_shape x n x d`-dim tensor of normalized inputs.

        Returns:
            A `batch_shape x n x d`-dim tensor of un-normalized inputs.
        """
        X_new = X.clone()
        X_new[..., self.indices] = 10.0 ** X_new[..., self.indices]
        return X_new


class Warp(ReversibleInputTransform, GPyTorchModule):
    r"""A transform that uses learned input warping functions.

    Each specified input dimension is warped using the CDF of a
    Kumaraswamy distribution. Typically, MAP estimates of the
    parameters of the Kumaraswamy distribution, for each input
    dimension, are learned jointly with the GP hyperparameters.

    TODO: implement support using independent warping functions
    for each output in batched multi-output and multi-task models.

    For now, ModelListGPs should be used to learn independent warping
    functions for each output.
    """

    # TODO: make minimum value dtype-dependent
    _min_concentration_level = 1e-4

    def __init__(
        self,
        indices: List[int],
        reverse: bool = False,
        eps: float = 1e-7,
        concentration1_prior: Optional[Prior] = None,
        concentration0_prior: Optional[Prior] = None,
        batch_shape: Optional[torch.Size] = None,
    ) -> None:
        r"""Initialize transform.

        Args:
            indices: The indices of the inputs to warp.
            reverse: A boolean indicating whether the forward pass should untransform
                the inputs.
            eps: A small value used to clip values to be in the interval (0, 1).
            concentration1_prior: A prior distribution on the concentration1 parameter
                of the Kumaraswamy distribution.
            concentration0_prior: A prior distribution on the concentration0 parameter
                of the Kumaraswamy distribution.
            batch_shape: The batch shape.
        """
        super().__init__()
        self.register_buffer("indices", torch.tensor(indices, dtype=torch.long))
        self.reverse = reverse
        self.batch_shape = batch_shape or torch.Size([])
        self._X_min = eps
        self._X_range = 1 - 2 * eps
        if len(self.batch_shape) > 0:
            # Note: this follows the gpytorch shape convention for lengthscales
            # There is ongoing discussion about the extra `1`.
            # TODO: update to follow new gpytorch convention resulting from
            # https://github.com/cornellius-gp/gpytorch/issues/1317
            batch_shape = self.batch_shape + torch.Size([1])
        else:
            batch_shape = self.batch_shape
        for i in (0, 1):
            p_name = f"concentration{i}"
            self.register_parameter(
                p_name,
                nn.Parameter(torch.full(batch_shape + self.indices.shape, 1.0)),
            )
        if concentration0_prior is not None:
            self.register_prior(
                "concentration0_prior",
                concentration0_prior,
                lambda m: m.concentration0,
                lambda m, v: m._set_concentration(i=0, value=v),
            )
        if concentration1_prior is not None:
            self.register_prior(
                "concentration1_prior",
                concentration1_prior,
                lambda m: m.concentration1,
                lambda m, v: m._set_concentration(i=1, value=v),
            )
        for i in (0, 1):
            p_name = f"concentration{i}"
            constraint = GreaterThan(
                self._min_concentration_level,
                transform=None,
                # set the initial value to be the identity transformation
                initial_value=1.0,
            )
            self.register_constraint(param_name=p_name, constraint=constraint)

    def _set_concentration(self, i: int, value: Union[float, Tensor]) -> None:
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.concentration0)
        self.initialize(**{f"concentration{i}": value})

    def _transform(self, X: Tensor) -> Tensor:
        r"""Warp the inputs through the Kumaraswamy CDF.

        Args:
            X: A `input_batch_shape x (batch_shape) x n x d`-dim tensor of inputs.
                batch_shape here can either be self.batch_shape or 1's such that
                it is broadcastable with self.batch_shape if self.batch_shape is set.

        Returns:
            A `input_batch_shape x (batch_shape) x n x d`-dim tensor of transformed
                inputs.
        """
        X_tf = expand_and_copy_tensor(X=X, batch_shape=self.batch_shape)
        k = Kumaraswamy(
            concentration1=self.concentration1, concentration0=self.concentration0
        )
        # normalize to [eps, 1-eps]
        X_tf[..., self.indices] = k.cdf(
            torch.clamp(
                X_tf[..., self.indices] * self._X_range + self._X_min,
                self._X_min,
                1.0 - self._X_min,
            )
        )
        return X_tf

    def _untransform(self, X: Tensor) -> Tensor:
        r"""Warp the inputs through the Kumaraswamy inverse CDF.

        Args:
            X: A `input_batch_shape x batch_shape x n x d`-dim tensor of inputs.

        Returns:
            A `input_batch_shape x batch_shape x n x d`-dim tensor of transformed
                inputs.
        """
        if len(self.batch_shape) > 0:
            if self.batch_shape != X.shape[-2 - len(self.batch_shape) : -2]:
                raise BotorchTensorDimensionError(
                    "The right most batch dims of X must match self.batch_shape: "
                    f"({self.batch_shape})."
                )
        X_tf = X.clone()
        k = Kumaraswamy(
            concentration1=self.concentration1, concentration0=self.concentration0
        )
        # unnormalize from [eps, 1-eps] to [0,1]
        X_tf[..., self.indices] = (
            (k.icdf(X_tf[..., self.indices]) - self._X_min) / self._X_range
        ).clamp(0.0, 1.0)
        return X_tf


class FilterFeatures(InputTransform, Module):

    r"""A transform that filters the input with a given set of features indices.

    As an example, this can be used in a multiobjective optimization with `ModelListGP`
    in which the specific models only share subsets of features (feature selection).
    A reason could be that it is known that specific features do not have any impact on
    a specific objective but they need to be included in the model for another one.
    """

    def __init__(
        self,
        feature_indices: Tensor,
    ) -> None:
        r"""Filter features from a model.

        Args:
            feature_set: An one-dim tensor denoting the indices of the features to be
                kept and fed to the model.
        """
        super().__init__()
        if feature_indices.dim() != 1:
            raise ValueError("`feature_indices` must be a one-dimensional tensor!")
        if feature_indices.dtype != torch.int64:
            raise ValueError("`feature_indices` tensor must be int64/long!")
        if (feature_indices < 0).any():
            raise ValueError(
                "Elements of `feature_indices` have to be larger/equal to zero!"
            )
        if len(feature_indices.unique()) != len(feature_indices):
            raise ValueError("Elements of `feature_indices` tensor must be unique!")
        self.register_buffer("feature_indices", feature_indices)

    def transform(self, X: Tensor) -> Tensor:
        r"""Transform the inputs by keeping only the in `feature_indices` specified
        feature indices and filtering out the others.

        Args:
            X: A `batch_shape x n x d`-dim tensor of inputs.

        Returns:
            A `batch_shape x n x e`-dim tensor of filtered inputs,
                where `e` is the length of `feature_indices`.
        """
        return X[..., self.feature_indices]

    def equals(self, other: InputTransform) -> bool:
        r"""Check if another input transform is equivalent.

        Args:
            other: Another input transform

        Returns:
            A boolean indicating if the other transform is equivalent.
        """
        if len(self.feature_indices) != len(other.feature_indices):
            return False
        return super().equals(other=other)
