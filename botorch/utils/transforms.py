#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Some basic data transformation helpers.
"""

from __future__ import annotations

import warnings
from functools import wraps
from typing import Any, Callable, List, Optional, TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from botorch.acquisition import AcquisitionFunction  # pragma: no cover
    from botorch.model import Model  # pragma: no cover


def squeeze_last_dim(Y: Tensor) -> Tensor:
    r"""Squeeze the last dimension of a Tensor.

    Args:
        Y: A `... x d`-dim Tensor.

    Returns:
        The input tensor with last dimension squeezed.

    Example:
        >>> Y = torch.rand(4, 3)
        >>> Y_squeezed = squeeze_last_dim(Y)
    """
    warnings.warn(
        "`botorch.utils.transforms.squeeze_last_dim` is deprecated. "
        "Simply use `.squeeze(-1)1 instead.",
        DeprecationWarning,
    )
    return Y.squeeze(-1)


def standardize(Y: Tensor) -> Tensor:
    r"""Standardizes (zero mean, unit variance) a tensor by dim=-2.

    If the tensor is single-dimensional, simply standardizes the tensor.
    If for some batch index all elements are equal (or if there is only a single
    data point), this function will return 0 for that batch index.

    Args:
        Y: A `batch_shape x n x m`-dim tensor.

    Returns:
        The standardized `Y`.

    Example:
        >>> Y = torch.rand(4, 3)
        >>> Y_standardized = standardize(Y)
    """
    stddim = -1 if Y.dim() < 2 else -2
    Y_std = Y.std(dim=stddim, keepdim=True)
    Y_std = Y_std.where(Y_std >= 1e-9, torch.full_like(Y_std, 1.0))
    return (Y - Y.mean(dim=stddim, keepdim=True)) / Y_std


def normalize(X: Tensor, bounds: Tensor) -> Tensor:
    r"""Min-max normalize X w.r.t. the provided bounds.

    Args:
        X: `... x d` tensor of data
        bounds: `2 x d` tensor of lower and upper bounds for each of the X's d
            columns.

    Returns:
        A `... x d`-dim tensor of normalized data, given by
            `(X - bounds[0]) / (bounds[1] - bounds[0])`. If all elements of `X`
            are contained within `bounds`, the normalized values will be
            contained within `[0, 1]^d`.

    Example:
        >>> X = torch.rand(4, 3)
        >>> bounds = torch.stack([torch.zeros(3), 0.5 * torch.ones(3)])
        >>> X_normalized = normalize(X, bounds)
    """
    return (X - bounds[0]) / (bounds[1] - bounds[0])


def unnormalize(X: Tensor, bounds: Tensor) -> Tensor:
    r"""Un-normalizes X w.r.t. the provided bounds.

    Args:
        X: `... x d` tensor of data
        bounds: `2 x d` tensor of lower and upper bounds for each of the X's d
            columns.

    Returns:
        A `... x d`-dim tensor of unnormalized data, given by
            `X * (bounds[1] - bounds[0]) + bounds[0]`. If all elements of `X`
            are contained in `[0, 1]^d`, the un-normalized values will be
            contained within `bounds`.

    Example:
        >>> X_normalized = torch.rand(4, 3)
        >>> bounds = torch.stack([torch.zeros(3), 0.5 * torch.ones(3)])
        >>> X = unnormalize(X_normalized, bounds)
    """
    return X * (bounds[1] - bounds[0]) + bounds[0]


def normalize_indices(indices: Optional[List[int]], d: int) -> Optional[List[int]]:
    r"""Normalize a list of indices to ensure that they are positive.

    Args:
        indices: A list of indices (may contain negative indices for indexing
            "from the back").
        d: The dimension of the tensor to index.

    Returns:
        A normalized list of indices such that each index is between `0` and
        `d-1`, or None if indices is None.
    """
    if indices is None:
        return indices
    normalized_indices = []
    for i in indices:
        if i < 0:
            i = i + d
        if i < 0 or i > d - 1:
            raise ValueError(f"Index {i} out of bounds for tensor or length {d}.")
        normalized_indices.append(i)
    return normalized_indices


def _verify_output_shape(acqf: Any, X: Tensor, output: Tensor) -> bool:
    r"""
    Performs the output shape checks for `t_batch_mode_transform`. Output shape checks
    help in catching the errors due to AcquisitionFunction arguments with erroneous
    return shapes before these errors propagate further down the line.

    This method checks that the `output` shape matches either the t-batch shape of X
    or the `batch_shape` of `acqf.model`.

    Args:
        acqf: The AcquisitionFunction object being evaluated.
        X: The `... x q x d`-dim input tensor with an explicit t-batch.
        output: The return value of `acqf.method(X, ...)`.

    Returns:
        True if `output` has the correct shape, False otherwise.
    """
    try:
        return (
            output.shape == X.shape[:-2]
            or (output.shape == torch.Size() and X.shape[:-2] == torch.Size([1]))
            or output.shape == acqf.model.batch_shape
            # for a batched model, we may unsqueeze a batch dimension in X
            # corresponding to the model's batch dim. In that case the
            # acquisition function output should replace the right-most
            # batch dim of X with the model's batch shape.
            or output.shape == X.shape[:-3] + acqf.model.batch_shape
        )
    except (AttributeError, NotImplementedError):
        # acqf does not have model or acqf.model does not define `batch_shape`
        warnings.warn(
            "Output shape checks failed! Expected output shape to match t-batch shape"
            f"of X, but got output with shape {output.shape} for X with shape"
            f"{X.shape}. Make sure that this is the intended behavior!",
            RuntimeWarning,
        )
        return True


def is_fully_bayesian(model: Model) -> bool:
    r"""Check if at least one model is a SaasFullyBayesianSingleTaskGP

    Args:
        model: A BoTorch model (may be a `ModelList` or `ModelListGP`)
        d: The dimension of the tensor to index.

    Returns:
        True if at least one model is a `SaasFullyBayesianSingleTaskGP`
    """
    from botorch.models import ModelList, ModelListGP
    from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
    from botorch.models.fully_bayesian_multitask import SaasFullyBayesianMultiTaskGP

    full_bayesian_model_cls = [
        SaasFullyBayesianSingleTaskGP,
        SaasFullyBayesianMultiTaskGP,
    ]

    if any(isinstance(model, m_cls) for m_cls in full_bayesian_model_cls):
        return True
    elif isinstance(model, ModelList):
        for m in model.models:
            if any(isinstance(m, m_cls) for m_cls in full_bayesian_model_cls):
                return True
            elif isinstance(m, ModelListGP) and any(
                isinstance(m_sub, m_cls)
                for m_sub in m.models
                for m_cls in full_bayesian_model_cls
            ):
                return True
    return False


def t_batch_mode_transform(
    expected_q: Optional[int] = None,
    assert_output_shape: bool = True,
) -> Callable[
    [Callable[[AcquisitionFunction, Any], Any]],
    Callable[[AcquisitionFunction, Any], Any],
]:
    r"""Factory for decorators enabling consistent t-batch behavior.

    This method creates decorators for instance methods to transform an input tensor
    `X` to t-batch mode (i.e. with at least 3 dimensions). This assumes the tensor
    has a q-batch dimension. The decorator also checks the q-batch size if `expected_q`
    is provided, and the output shape if `assert_output_shape` is `True`.

    Args:
        expected_q: The expected q-batch size of `X`. If specified, this will raise an
            AssertionError if `X`'s q-batch size does not equal expected_q.
        assert_output_shape: If `True`, this will raise an AssertionError if the
            output shape does not match either the t-batch shape of `X`,
            or the `acqf.model.batch_shape` for acquisition functions using
            batched models.

    Returns:
        The decorated instance method.

    Example:
        >>> class ExampleClass:
        >>>     @t_batch_mode_transform(expected_q=1)
        >>>     def single_q_method(self, X):
        >>>         ...
        >>>
        >>>     @t_batch_mode_transform()
        >>>     def arbitrary_q_method(self, X):
        >>>         ...
    """

    def decorator(
        method: Callable[[AcquisitionFunction, Any], Any],
    ) -> Callable[[AcquisitionFunction, Any], Any]:
        @wraps(method)
        def decorated(
            acqf: AcquisitionFunction, X: Any, *args: Any, **kwargs: Any
        ) -> Any:

            # Allow using acquisition functions for other inputs (e.g. lists of strings)
            if not isinstance(X, Tensor):
                return method(acqf, X, *args, **kwargs)

            if X.dim() < 2:
                raise ValueError(
                    f"{type(acqf).__name__} requires X to have at least 2 dimensions,"
                    f" but received X with only {X.dim()} dimensions."
                )
            elif expected_q is not None and X.shape[-2] != expected_q:
                raise AssertionError(
                    f"Expected X to be `batch_shape x q={expected_q} x d`, but"
                    f" got X with shape {X.shape}."
                )
            # add t-batch dim
            X = X if X.dim() > 2 else X.unsqueeze(0)
            output = method(acqf, X, *args, **kwargs)
            if hasattr(acqf, "model") and is_fully_bayesian(acqf.model):
                output = output.mean(dim=-1)
            if assert_output_shape and not _verify_output_shape(
                acqf=acqf,
                X=X,
                output=output,
            ):
                raise AssertionError(
                    "Expected the output shape to match either the t-batch shape of "
                    "X, or the `model.batch_shape` in the case of acquisition "
                    "functions using batch models; but got output with shape "
                    f"{output.shape} for X with shape {X.shape}."
                )
            return output

        return decorated

    return decorator


def concatenate_pending_points(
    method: Callable[[Any, Tensor], Any]
) -> Callable[[Any, Tensor], Any]:
    r"""Decorator concatenating X_pending into an acquisition function's argument.

    This decorator works on the `forward` method of acquisition functions taking
    a tensor `X` as the argument. If the acquisition function has an `X_pending`
    attribute (that is not `None`), this is concatenated into the input `X`,
    appropriately expanding the pending points to match the batch shape of `X`.

    Example:
        >>> class ExampleAcquisitionFunction:
        >>>     @concatenate_pending_points
        >>>     @t_batch_mode_transform()
        >>>     def forward(self, X):
        >>>         ...
    """

    @wraps(method)
    def decorated(cls: Any, X: Tensor, **kwargs: Any) -> Any:
        if cls.X_pending is not None:
            X = torch.cat([X, match_batch_shape(cls.X_pending, X)], dim=-2)
        return method(cls, X, **kwargs)

    return decorated


def match_batch_shape(X: Tensor, Y: Tensor) -> Tensor:
    r"""Matches the batch dimension of a tensor to that of another tensor.

    Args:
        X: A `batch_shape_X x q x d` tensor, whose batch dimensions that
            correspond to batch dimensions of `Y` are to be matched to those
            (if compatible).
        Y: A `batch_shape_Y x q' x d` tensor.

    Returns:
        A `batch_shape_Y x q x d` tensor containing the data of `X` expanded to
        the batch dimensions of `Y` (if compatible). For instance, if `X` is
        `b'' x b' x q x d` and `Y` is `b x q x d`, then the returned tensor is
        `b'' x b x q x d`.

    Example:
        >>> X = torch.rand(2, 1, 5, 3)
        >>> Y = torch.rand(2, 6, 4, 3)
        >>> X_matched = match_batch_shape(X, Y)
        >>> X_matched.shape
        torch.Size([2, 6, 5, 3])

    """
    return X.expand(X.shape[: -(Y.dim())] + Y.shape[:-2] + X.shape[-2:])


def convert_to_target_pre_hook(module, *args):
    r"""Pre-hook for automatically calling `.to(X)` on module prior to `forward`"""
    module.to(args[0][0])
