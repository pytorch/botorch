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
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from botorch.exceptions.errors import BotorchTensorDimensionError
from botorch.models.transforms.utils import expand_and_copy_tensor
from botorch.models.utils import fantasize
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

    Properties:
        is_one_to_many: A boolean denoting whether the transform produces
            multiple values for each input.
        transform_on_train: A boolean indicating whether to apply the
            transform in train() mode.
        transform_on_eval: A boolean indicating whether to apply the
            transform in eval() mode.
        transform_on_fantasize: A boolean indicating whether to apply
            the transform when called from within a `fantasize` call.

    :meta private:
    """

    is_one_to_many: bool = False
    transform_on_eval: bool
    transform_on_train: bool
    transform_on_fantasize: bool

    def forward(self, X: Tensor) -> Tensor:
        r"""Transform the inputs to a model.

        Args:
            X: A `batch_shape x n x d`-dim tensor of inputs.

        Returns:
            A `batch_shape x n' x d`-dim tensor of transformed inputs.
        """
        if self.training:
            if self.transform_on_train:
                return self.transform(X)
        elif self.transform_on_eval:
            if fantasize.off() or self.transform_on_fantasize:
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
        return (
            type(self) == type(other)
            and (self.transform_on_train == other.transform_on_train)
            and (self.transform_on_eval == other.transform_on_eval)
            and (self.transform_on_fantasize == other.transform_on_fantasize)
            and all(
                torch.allclose(v, other_state_dict[k].to(v))
                for k, v in self.state_dict().items()
            )
        )

    def preprocess_transform(self, X: Tensor) -> Tensor:
        r"""Apply transforms for preprocessing inputs.

        The main use cases for this method are 1) to preprocess training data
        before calling `set_train_data` and 2) preprocess `X_baseline` for noisy
        acquisition functions so that `X_baseline` is "preprocessed" with the
        same transformations as the cached training inputs.

        Args:
            X: A `batch_shape x n x d`-dim tensor of inputs.

        Returns:
            A `batch_shape x n x d`-dim tensor of (transformed) inputs.
        """
        if self.transform_on_train:
            # We need to disable learning of bounds here.
            # See why: https://github.com/pytorch/botorch/issues/1078.
            if hasattr(self, "learn_bounds"):
                learn_bounds = self.learn_bounds
                self.learn_bounds = False
                result = self.transform(X)
                self.learn_bounds = learn_bounds
                return result
            else:
                return self.transform(X)
        return X


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
        self.transform_on_train = False
        self.transform_on_eval = False
        self.transform_on_fantasize = False
        for tf in transforms.values():
            self.is_one_to_many |= tf.is_one_to_many
            self.transform_on_train |= tf.transform_on_train
            self.transform_on_eval |= tf.transform_on_eval
            self.transform_on_fantasize |= tf.transform_on_fantasize

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

    def preprocess_transform(self, X: Tensor) -> Tensor:
        r"""Apply transforms for preprocessing inputs.

        The main use cases for this method are 1) to preprocess training data
        before calling `set_train_data` and 2) preprocess `X_baseline` for noisy
        acquisition functions so that `X_baseline` is "preprocessed" with the
        same transformations as the cached training inputs.

        Args:
            X: A `batch_shape x n x d`-dim tensor of inputs.

        Returns:
            A `batch_shape x n x d`-dim tensor of (transformed) inputs.
        """
        for tf in self.values():
            X = tf.preprocess_transform(X)
        return X


class ReversibleInputTransform(InputTransform, ABC):
    r"""An abstract class for a reversible input transform.

    Properties:
        reverse: A boolean indicating if the functionality of transform
            and untransform methods should be swapped.

    :meta private:
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
        transform_on_train: bool = True,
        transform_on_eval: bool = True,
        transform_on_fantasize: bool = True,
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
            batch_shape: The batch shape of the inputs (assuming input tensors
                of shape `batch_shape x n x d`). If provided, perform individual
                normalization per batch, otherwise uses a single normalization.
            transform_on_train: A boolean indicating whether to apply the
                transforms in train() mode. Default: True.
            transform_on_eval: A boolean indicating whether to apply the
                transform in eval() mode. Default: True.
            transform_on_fantasize: A boolean indicating whether to apply the
                transform when called from within a `fantasize` call. Default: True.
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
        self.transform_on_train = transform_on_train
        self.transform_on_eval = transform_on_eval
        self.transform_on_fantasize = transform_on_fantasize
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

            n = len(self.batch_shape) + 2
            if X.ndim < n:
                raise ValueError(
                    f"`X` must have at least {n} dimensions, {n - 2} batch and 2 innate"
                    f" , but has {X.ndim}."
                )

            # Move extra batch and innate batch (i.e. marginal) dimensions to the right
            batch_ndim = min(len(self.batch_shape), X.ndim - 2)  # batch rank of `X`
            _X = X.permute(
                *range(X.ndim - batch_ndim - 2, X.ndim - 2),  # module batch dims
                X.ndim - 1,  # input dim
                *range(X.ndim - batch_ndim - 2),  # other dims, to be reduced over
                X.ndim - 2,  # marginal dim
            ).reshape(*self.batch_shape, 1, X.shape[-1], -1)

            # Extract minimums and ranges
            self.mins = _X.min(dim=-1).values  # batch_shape x (1, d)
            self.ranges = (_X.max(dim=-1).values - self.mins).clip(min=self.min_range)

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
        transform_on_train: bool = True,
        transform_on_eval: bool = True,
        transform_on_fantasize: bool = True,
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
            transform_on_train: A boolean indicating whether to apply the
                transforms in train() mode. Default: True
            transform_on_eval: A boolean indicating whether to apply the
                transform in eval() mode. Default: True
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
        self.transform_on_train = transform_on_train
        self.transform_on_eval = transform_on_eval
        self.transform_on_fantasize = transform_on_fantasize
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

            n = len(self.batch_shape) + 2
            if X.ndim < n:
                raise ValueError(
                    f"`X` must have at least {n} dimensions, {n - 2} batch and 2 innate"
                    f" , but has {X.ndim}."
                )

            # Aggregate means and standard deviations over extra batch and marginal dims
            batch_ndim = min(len(self.batch_shape), X.ndim - 2)  # batch rank of `X`
            reduce_dims = (*range(X.ndim - batch_ndim - 2), X.ndim - 2)
            self.stds, self.means = (
                values.unsqueeze(-2)
                for values in torch.std_mean(X, dim=reduce_dims, unbiased=True)
            )
            self.stds.clamp_(min=self.min_std)

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
        transform_on_train: bool = True,
        transform_on_eval: bool = True,
        transform_on_fantasize: bool = True,
        approximate: bool = True,
        tau: float = 1e-3,
    ) -> None:
        r"""Initialize transform.

        Args:
            indices: The indices of the integer inputs.
            transform_on_train: A boolean indicating whether to apply the
                transforms in train() mode. Default: True.
            transform_on_eval: A boolean indicating whether to apply the
                transform in eval() mode. Default: True.
            transform_on_fantasize: A boolean indicating whether to apply the
                transform when called from within a `fantasize` call. Default: True.
            approximate: A boolean indicating whether approximate or exact
                rounding should be used. Default: approximate.
            tau: The temperature parameter for approximate rounding.
        """
        super().__init__()
        self.transform_on_train = transform_on_train
        self.transform_on_eval = transform_on_eval
        self.transform_on_fantasize = transform_on_fantasize
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
        transform_on_train: bool = True,
        transform_on_eval: bool = True,
        transform_on_fantasize: bool = True,
        reverse: bool = False,
    ) -> None:
        r"""Initialize transform.

        Args:
            indices: The indices of the inputs to log transform.
            transform_on_train: A boolean indicating whether to apply the
                transforms in train() mode. Default: True.
            transform_on_eval: A boolean indicating whether to apply the
                transform in eval() mode. Default: True.
            transform_on_fantasize: A boolean indicating whether to apply the
                transform when called from within a `fantasize` call. Default: True.
            reverse: A boolean indicating whether the forward pass should untransform
                the inputs.
        """
        super().__init__()
        self.register_buffer("indices", torch.tensor(indices, dtype=torch.long))
        self.transform_on_train = transform_on_train
        self.transform_on_eval = transform_on_eval
        self.transform_on_fantasize = transform_on_fantasize
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
        transform_on_train: bool = True,
        transform_on_eval: bool = True,
        transform_on_fantasize: bool = True,
        reverse: bool = False,
        eps: float = 1e-7,
        concentration1_prior: Optional[Prior] = None,
        concentration0_prior: Optional[Prior] = None,
        batch_shape: Optional[torch.Size] = None,
    ) -> None:
        r"""Initialize transform.

        Args:
            indices: The indices of the inputs to warp.
            transform_on_train: A boolean indicating whether to apply the
                transforms in train() mode. Default: True.
            transform_on_eval: A boolean indicating whether to apply the
                transform in eval() mode. Default: True.
            transform_on_fantasize: A boolean indicating whether to apply the
                transform when called from within a `fantasize` call. Default: True.
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
        self.transform_on_train = transform_on_train
        self.transform_on_eval = transform_on_eval
        self.transform_on_fantasize = transform_on_fantasize
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


class AppendFeatures(InputTransform, Module):
    r"""A transform that appends the input with a given set of features either
    provided beforehand or generated on the fly via a callable.

    As an example, the predefined set of features can be used with
    `RiskMeasureMCObjective` to optimize risk measures as described in
    [Cakmak2020risk]_. A tutorial notebook implementing the rhoKG acqusition
    function introduced in [Cakmak2020risk]_ can be found at
    https://botorch.org/tutorials/risk_averse_bo_with_environmental_variables.

    The steps for using this to obtain samples of a risk measure are as follows:

    -   Train a model on `(x, w)` inputs and the corresponding observations;

    -   Pass in an instance of `AppendFeatures` with the `feature_set` denoting the
        samples of `W` as the `input_transform` to the trained model;

    -   Call `posterior(...).rsample(...)` on the model with `x` inputs only to
        get the joint posterior samples over `(x, w)`s, where the `w`s come
        from the `feature_set`;

    -   Pass these posterior samples through the `RiskMeasureMCObjective` of choice to
        get the samples of the risk measure.

    Note: The samples of the risk measure obtained this way are in general biased
    since the `feature_set` does not fully represent the distribution of the
    environmental variable.

    Possible examples for using a callable include statistical models that are built on
    PyTorch, built-in mathematical operations such as torch.sum, or custom scripted
    functions. By this, this input transform allows for advanced feature engineering
    and transfer learning models within the optimization loop.

    Example:
        >>> # We consider 1D `x` and 1D `w`, with `W` having a
        >>> # uniform distribution over [0, 1]
        >>> model = SingleTaskGP(
        ...     train_X=torch.rand(10, 2),
        ...     train_Y=torch.randn(10, 1),
        ...     input_transform=AppendFeatures(feature_set=torch.rand(10, 1))
        ... )
        >>> mll = ExactMarginalLogLikelihood(model.likelihood, model)
        >>> fit_gpytorch_model(mll)
        >>> test_x = torch.rand(3, 1)
        >>> # `posterior_samples` is a `10 x 30 x 1`-dim tensor
        >>> posterior_samples = model.posterior(test_x).rsamples(torch.size([10]))
        >>> risk_measure = VaR(alpha=0.8, n_w=10)
        >>> # `risk_measure_samples` is a `10 x 3`-dim tensor of samples of the
        >>> # risk measure VaR
        >>> risk_measure_samples = risk_measure(posterior_samples)
    """

    is_one_to_many: bool = True

    def __init__(
        self,
        feature_set: Optional[Tensor] = None,
        f: Optional[Callable[[Tensor], Tensor]] = None,
        indices: Optional[List[int]] = None,
        fkwargs: Optional[Dict[str, Any]] = None,
        skip_expand: bool = False,
        transform_on_train: bool = False,
        transform_on_eval: bool = True,
        transform_on_fantasize: bool = False,
    ) -> None:
        r"""Append `feature_set` to each input or generate a set of features to
        append on the fly via a callable.

        Args:
            feature_set: An `n_f x d_f`-dim tensor denoting the features to be
                appended to the inputs. Default: None.
            f: A callable mapping a `batch_shape x q x d`-dim input tensor `X`
                to a `batch_shape x q x n_f x d_f`-dimensional output tensor.
                Default: None.
            indices: List of indices denoting the indices of the features to be
                passed into f. Per default all features are passed to `f`.
                Default: None.
            fkwargs: Dictionary of keyword arguments passed to the callable `f`.
                Default: None.
            skip_expand: A boolean indicating whether to expand the input tensor
                before appending features. This is intended for use with an
                `InputPerturbation`. If `True`, the input tensor will be expected
                to be of shape `batch_shape x (q * n_f) x d`. Not implemented
                in combination with a callable.
            transform_on_train: A boolean indicating whether to apply the
                transforms in train() mode. Default: False.
            transform_on_eval: A boolean indicating whether to apply the
                transform in eval() mode. Default: True.
            transform_on_fantasize: A boolean indicating whether to apply the
                transform when called from within a `fantasize` call. Default: False.
        """
        super().__init__()
        if (feature_set is None) and (f is None):
            raise ValueError(
                "Either a `feature_set` or a callable `f` has to be provided."
            )
        if (feature_set is not None) and (f is not None):
            raise ValueError(
                "Only one can be used: either `feature_set` or callable `f`."
            )
        if feature_set is not None:
            if feature_set.dim() != 2:
                raise ValueError("`feature_set` must be an `n_f x d_f`-dim tensor!")
            self.register_buffer("feature_set", feature_set)
            self._f = None
        if f is not None:
            if skip_expand:
                raise ValueError(
                    "`skip_expand` option is not supported in case of using a callable"
                )
            if (indices is not None) and (len(indices) == 0):
                raise ValueError("`indices` list is empty!")
            if indices is not None:
                indices = torch.tensor(indices, dtype=torch.long)
                if len(indices.unique()) != len(indices):
                    raise ValueError("Elements of `indices` tensor must be unique!")
                self.indices = indices
            else:
                self.indices = slice(None)
            self._f = f
            self.fkwargs = fkwargs or {}

        self.skip_expand = skip_expand
        self.transform_on_train = transform_on_train
        self.transform_on_eval = transform_on_eval
        self.transform_on_fantasize = transform_on_fantasize

    def transform(self, X: Tensor) -> Tensor:
        r"""Transform the inputs by appending `feature_set` to each input or
        by generating a set of features to be appended on the fly via a callable.

        For each `1 x d`-dim element in the input tensor, this will produce
        an `n_f x (d + d_f)`-dim tensor with `feature_set` appended as the last `d_f`
        dimensions. For a generic `batch_shape x q x d`-dim `X`, this translates to a
        `batch_shape x (q * n_f) x (d + d_f)`-dim output, where the values corresponding
        to `X[..., i, :]` are found in `output[..., i * n_f: (i + 1) * n_f, :]`.

        Note: Adding the `feature_set` on the `q-batch` dimension is necessary to avoid
        introducing additional bias by evaluating the inputs on independent GP
        sample paths.

        Args:
            X: A `batch_shape x q x d`-dim tensor of inputs. If `self.skip_expand` is
                `True`, then `X` should be of shape `batch_shape x (q * n_f) x d`,
                typically obtained by passing a `batch_shape x q x d` shape input
                through an `InputPerturbation` with `n_f` perturbation values.

        Returns:
            A `batch_shape x (q * n_f) x (d + d_f)`-dim tensor of appended inputs.
        """
        if self._f is not None:
            expanded_features = self._f(X[..., self.indices], **self.fkwargs)
            n_f = expanded_features.shape[-2]
        else:
            n_f = self.feature_set.shape[-2]

        if self.skip_expand:
            expanded_X = X.view(*X.shape[:-2], -1, n_f, X.shape[-1])
        else:
            expanded_X = X.unsqueeze(dim=-2).expand(*X.shape[:-1], n_f, -1)

        if self._f is None:
            expanded_features = self.feature_set.expand(*expanded_X.shape[:-1], -1)

        appended_X = torch.cat([expanded_X, expanded_features], dim=-1)
        return appended_X.view(*X.shape[:-2], -1, appended_X.shape[-1])


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
        transform_on_train: bool = True,
        transform_on_eval: bool = True,
        transform_on_fantasize: bool = True,
    ) -> None:
        r"""Filter features from a model.

        Args:
            feature_set: An one-dim tensor denoting the indices of the features to be
                kept and fed to the model.
            transform_on_train: A boolean indicating whether to apply the
                transforms in train() mode. Default: True.
            transform_on_eval: A boolean indicating whether to apply the
                transform in eval() mode. Default: True.
            transform_on_fantasize: A boolean indicating whether to apply the
                transform when called from within a `fantasize` call. Default: True.
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
        self.transform_on_train = transform_on_train
        self.transform_on_eval = transform_on_eval
        self.transform_on_fantasize = transform_on_fantasize
        self.register_buffer("feature_indices", feature_indices)

    def transform(self, X: Tensor) -> Tensor:
        r"""Transform the inputs by keeping only the in `feature_indices` specified
        feature indices and filtering out the others.

        Args:
            X: A `batch_shape x q x d`-dim tensor of inputs.

        Returns:
            A `batch_shape x q x e`-dim tensor of filtered inputs,
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


class InputPerturbation(InputTransform, Module):
    r"""A transform that adds the set of perturbations to the given input.

    Similar to `AppendFeatures`, this can be used with `RiskMeasureMCObjective`
    to optimize risk measures. See `AppendFeatures` for additional discussion
    on optimizing risk measures.

    A tutorial notebook using this with `qNoisyExpectedImprovement` can be found at
    https://botorch.org/tutorials/risk_averse_bo_with_input_perturbations.
    """

    is_one_to_many: bool = True

    def __init__(
        self,
        perturbation_set: Union[Tensor, Callable[[Tensor], Tensor]],
        bounds: Optional[Tensor] = None,
        multiplicative: bool = False,
        transform_on_train: bool = False,
        transform_on_eval: bool = True,
        transform_on_fantasize: bool = False,
    ) -> None:
        r"""Add `perturbation_set` to each input.

        Args:
            perturbation_set: An `n_p x d`-dim tensor denoting the perturbations
                to be added to the inputs. Alternatively, this can be a callable that
                returns `batch x n_p x d`-dim tensor of perturbations for input of
                shape `batch x d`. This is useful for heteroscedastic perturbations.
            bounds: A `2 x d`-dim tensor of lower and upper bounds for each
                column of the input. If given, the perturbed inputs will be
                clamped to these bounds.
            multiplicative: A boolean indicating whether the input perturbations
                are additive or multiplicative. If True, inputs will be multiplied
                with the perturbations.
            transform_on_train: A boolean indicating whether to apply the
                transforms in train() mode. Default: False.
            transform_on_eval: A boolean indicating whether to apply the
                transform in eval() mode. Default: True.
            transform_on_fantasize: A boolean indicating whether to apply the
                transform when called from within a `fantasize` call. Default: False.
        """
        super().__init__()
        if isinstance(perturbation_set, Tensor):
            if perturbation_set.dim() != 2:
                raise ValueError("`perturbation_set` must be an `n_p x d`-dim tensor!")
            self.register_buffer("perturbation_set", perturbation_set)
        else:
            self.perturbation_set = perturbation_set
        if bounds is not None:
            if (
                isinstance(perturbation_set, Tensor)
                and bounds.shape[-1] != perturbation_set.shape[-1]
            ):
                raise ValueError(
                    "`bounds` must have the same number of columns (last dimension) as "
                    f"the `perturbation_set`! Got {bounds.shape[-1]} and "
                    f"{perturbation_set.shape[-1]}."
                )
            self.register_buffer("bounds", bounds)
        else:
            self.bounds = None
        self.multiplicative = multiplicative
        self.transform_on_train = transform_on_train
        self.transform_on_eval = transform_on_eval
        self.transform_on_fantasize = transform_on_fantasize

    def transform(self, X: Tensor) -> Tensor:
        r"""Transform the inputs by adding `perturbation_set` to each input.

        For each `1 x d`-dim element in the input tensor, this will produce
        an `n_p x d`-dim tensor with the `perturbation_set` added to the input.
        For a generic `batch_shape x q x d`-dim `X`, this translates to a
        `batch_shape x (q * n_p) x d`-dim output, where the values corresponding
        to `X[..., i, :]` are found in `output[..., i * n_w: (i + 1) * n_w, :]`.

        Note: Adding the `perturbation_set` on the `q-batch` dimension is necessary
        to avoid introducing additional bias by evaluating the inputs on independent
        GP sample paths.

        Args:
            X: A `batch_shape x q x d`-dim tensor of inputs.

        Returns:
            A `batch_shape x (q * n_p) x d`-dim tensor of perturbed inputs.
        """
        if isinstance(self.perturbation_set, Tensor):
            perturbations = self.perturbation_set
        else:
            perturbations = self.perturbation_set(X)
        expanded_X = X.unsqueeze(dim=-2).expand(
            *X.shape[:-1], perturbations.shape[-2], -1
        )
        expanded_perturbations = perturbations.expand(*expanded_X.shape[:-1], -1)
        if self.multiplicative:
            perturbed_inputs = expanded_X * expanded_perturbations
        else:
            perturbed_inputs = expanded_X + expanded_perturbations
        perturbed_inputs = perturbed_inputs.reshape(*X.shape[:-2], -1, X.shape[-1])
        if self.bounds is not None:
            perturbed_inputs = torch.maximum(
                torch.minimum(perturbed_inputs, self.bounds[1]), self.bounds[0]
            )
        return perturbed_inputs
