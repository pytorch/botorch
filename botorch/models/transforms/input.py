#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
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
from botorch.distributions.distributions import Kumaraswamy
from botorch.exceptions.errors import BotorchTensorDimensionError
from botorch.utils.rounding import approximate_round
from gpytorch import Module as GPyTorchModule
from gpytorch.constraints import GreaterThan
from gpytorch.priors import Prior
from torch import Tensor, nn
from torch.nn import Module, ModuleDict


class InputTransform(ABC):
    r"""Abstract base class for input transforms.

    Properties:
        transform_on_train: A boolean indicating whether to apply the
            transform in train() mode.
        transform_on_eval: A boolean indicating whether to apply the
            transform in eval() mode.
        transform_on_preprocess: A boolean indicating whether to apply
            the transform when preprocessing inputs.
    """

    transform_on_eval: bool
    transform_on_train: bool
    transform_on_preprocess: bool

    def forward(self, X: Tensor) -> Tensor:
        r"""Transform the inputs to a model.

        Args:
            X: A `batch_shape x n x d`-dim tensor of inputs.

        Returns:
            A `batch_shape x n x d`-dim tensor of transformed inputs.
        """
        if (self.training and self.transform_on_train) or (
            not self.training and self.transform_on_eval
        ):
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
            f"{self.__class__.__name__} does not implement the `untransform` method"
        )

    def equals(self, other: InputTransform) -> bool:
        r"""Check if another input transform is equivalent.

        Note: The reason that a custom equals method is definde rather than
        defining an __eq__ method is because defining an __eq__ method sets
        the __hash__ method to None. Hashing modules is currently used in
        pytorch. See https://github.com/pytorch/pytorch/issues/7733.

        Args:
            other: Another input transform

        Returns:
            A boolean indicating if the other transform is equivalent.
        """
        other_state_dict = other.state_dict()
        return (
            type(self) == type(other)
            and (self.transform_on_train == other.transform_on_train)
            and (self.transform_on_eval == other.transform_on_eval)
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
        same transformations as the cached training inputs

        Args:
            X: A `batch_shape x n x d`-dim tensor of inputs.

        Returns:
            A `batch_shape x n x d`-dim tensor of (transformed) inputs.
        """
        if self.transform_on_preprocess:
            return self.transform(X)
        return X


class ChainedInputTransform(InputTransform, ModuleDict):
    r"""An input transform representing the chaining of individual transforms"""

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
        self.transform_on_preprocess = False
        for tf in transforms.values():
            self.transform_on_train |= tf.transform_on_train
            self.transform_on_eval |= tf.transform_on_eval
            self.transform_on_preprocess |= tf.transform_on_preprocess

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
            other: Another input transform

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
        same transformations as the cached training inputs

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
            other: Another input transform

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
        bounds: Optional[Tensor] = None,
        batch_shape: torch.Size = torch.Size(),  # noqa: B008
        transform_on_train: bool = True,
        transform_on_eval: bool = True,
        transform_on_preprocess: bool = False,
        reverse: bool = False,
    ) -> None:
        r"""Normalize the inputs to the unit cube.

        Args:
            d: The dimension of the input space.
            bounds: If provided, use these bounds to normalize the inputs. If
                omitted, learn the bounds in train mode.
            batch_shape: The batch shape of the inputs (asssuming input tensors
                of shape `batch_shape x n x d`). If provided, perform individual
                normalization per batch, otherwise uses a single normalization.
            transform_on_train: A boolean indicating whether to apply the
                transforms in train() mode. Default: True
            transform_on_eval: A boolean indicating whether to apply the
                transform in eval() mode. Default: True
            transform_on_preprocess: A boolean indicating whether to apply the
                transform when preprocessing inputs. Default: False
            reverse: A boolean indicating whether the forward pass should untransform
                the inputs.
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
        self.transform_on_train = transform_on_train
        self.transform_on_eval = transform_on_eval
        self.transform_on_preprocess = transform_on_preprocess
        self.reverse = reverse
        self.batch_shape = batch_shape

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
                    f"Wrong input. dimension. Received {X.size(-1)}, "
                    f"expected {self.mins.size(-1)}"
                )
            self.mins = X.min(dim=-2, keepdim=True)[0]
            self.ranges = X.max(dim=-2, keepdim=True)[0] - self.mins
        return (X - self.mins) / self.ranges

    def _untransform(self, X: Tensor) -> Tensor:
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

    def equals(self, other: InputTransform) -> bool:
        r"""Check if another input transform is equivalent.

        Args:
            other: Another input transform

        Returns:
            A boolean indicating if the other transform is equivalent.
        """
        return (
            super().equals(other=other)
            and (self._d == other._d)
            and (self.learn_bounds == other.learn_bounds)
        )


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
        transform_on_preprocess: bool = False,
        approximate: bool = True,
        tau: float = 1e-3,
    ) -> None:
        r"""Initialize transform.

        Args:
            indices: The indices of the integer inputs
            transform_on_train: A boolean indicating whether to apply the
                transforms in train() mode. Default: True
            transform_on_eval: A boolean indicating whether to apply the
                transform in eval() mode. Default: True
            transform_on_preprocess: A boolean indicating whether to apply the
                transform when preprocessing inputs. Default: False
            approximate: A boolean indicating whether approximate or exact
                rounding should be used. Default: approximate
            tau: The temperature parameter for approximate rounding
        """
        super().__init__()
        self.transform_on_train = transform_on_train
        self.transform_on_eval = transform_on_eval
        self.transform_on_preprocess = transform_on_preprocess
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
            other: Another input transform

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
        transform_on_preprocess: bool = False,
        reverse: bool = False,
    ) -> None:
        r"""Initialize transform.

        Args:
            indices: The indices of the inputs to log transform
            transform_on_train: A boolean indicating whether to apply the
                transforms in train() mode. Default: True
            transform_on_eval: A boolean indicating whether to apply the
                transform in eval() mode. Default: True
            transform_on_preprocess: A boolean indicating whether to apply the
                transform when preprocessing inputs. Default: False
            reverse: A boolean indicating whether the forward pass should untransform
                the inputs.
        """
        super().__init__()
        self.register_buffer("indices", torch.tensor(indices, dtype=torch.long))
        self.transform_on_train = transform_on_train
        self.transform_on_eval = transform_on_eval
        self.transform_on_preprocess = transform_on_preprocess
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
        transform_on_preprocess: bool = False,
        reverse: bool = False,
        eps: float = 1e-7,
        concentration1_prior: Optional[Prior] = None,
        concentration0_prior: Optional[Prior] = None,
    ) -> None:
        r"""Initialize transform.

        Args:
            indices: The indices of the inputs to warp.
            transform_on_train: A boolean indicating whether to apply the
                transforms in train() mode. Default: True.
            transform_on_eval: A boolean indicating whether to apply the
                transform in eval() mode. Default: True.
            transform_on_preprocess: A boolean indicating whether to apply the
                transform when preprocessing. Default: False.
            reverse: A boolean indicating whether the forward pass should untransform
                the inputs.
            eps: A small value used to clip values to be in the interval (0, 1).
            concentration1_prior: A prior distribution on the concentration1 parameter
                of the Kumaraswamy distribution.
            concentration0_prior: A prior distribution on the concentration0 parameter
                of the Kumaraswamy distribution.


        """
        super().__init__()
        self.eps = eps
        self.register_buffer("indices", torch.tensor(indices, dtype=torch.long))
        self.transform_on_train = transform_on_train
        self.transform_on_eval = transform_on_eval
        self.transform_on_preprocess = transform_on_preprocess
        self.reverse = reverse
        for i in (0, 1):
            p_name = f"concentration{i}"
            self.register_parameter(
                p_name, nn.Parameter(torch.full(self.indices.shape, 1.0))
            )
        if concentration0_prior is not None:
            self.register_prior(
                "concentration0_prior",
                concentration0_prior,
                lambda: self.concentration0,
                lambda v: self._set_concentration(i=0, value=v),
            )
        if concentration1_prior is not None:
            self.register_prior(
                "concentration1_prior",
                concentration1_prior,
                lambda: self.concentration1,
                lambda v: self._set_concentration(i=1, value=v),
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
            X: A `batch_shape x n x d`-dim tensor of inputs.

        Returns:
            A `batch_shape x n x d`-dim tensor of transformed inputs.
        """
        X_tf = X.clone()
        k = Kumaraswamy(
            concentration1=self.concentration1, concentration0=self.concentration0
        )
        X_tf[..., self.indices] = k.cdf(
            X[..., self.indices].clamp(self.eps, 1 - self.eps)
        )
        return X_tf

    def _untransform(self, X: Tensor) -> Tensor:
        X_tf = X.clone()
        k = Kumaraswamy(
            concentration1=self.concentration1, concentration0=self.concentration0
        )
        X_tf[..., self.indices] = k.icdf(X[..., self.indices])
        return X_tf
