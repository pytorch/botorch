#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Assorted helper methods and objects for working with BoTorch models."""

from __future__ import annotations

import warnings
from contextlib import contextmanager, ExitStack
from typing import List, Optional, Tuple

import torch
from botorch import settings
from botorch.exceptions import InputDataError, InputDataWarning
from botorch.settings import _Flag
from gpytorch import settings as gpt_settings
from gpytorch.module import Module
from torch import Tensor


def _make_X_full(X: Tensor, output_indices: List[int], tf: int) -> Tensor:
    r"""Helper to construct input tensor with task indices.

    Args:
        X: The raw input tensor (without task information).
        output_indices: The output indices to generate (passed in via `posterior`).
        tf: The task feature index.

    Returns:
        Tensor: The full input tensor for the multi-task model, including task
            indices.
    """
    index_shape = X.shape[:-1] + torch.Size([1])
    indexers = (
        torch.full(index_shape, fill_value=i, device=X.device, dtype=X.dtype)
        for i in output_indices
    )
    X_l, X_r = X[..., :tf], X[..., tf:]
    return torch.cat(
        [torch.cat([X_l, indexer, X_r], dim=-1) for indexer in indexers], dim=-2
    )


def multioutput_to_batch_mode_transform(
    train_X: Tensor,
    train_Y: Tensor,
    num_outputs: int,
    train_Yvar: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    r"""Transforms training inputs for a multi-output model.

    Used for multi-output models that internally are represented by a
    batched single output model, where each output is modeled as an
    independent batch.

    Args:
        train_X: A `n x d` or `input_batch_shape x n x d` (batch mode) tensor of
            training features.
        train_Y: A `n x m` or `target_batch_shape x n x m` (batch mode) tensor of
            training observations.
        num_outputs: number of outputs
        train_Yvar: A `n x m` or `target_batch_shape x n x m` tensor of observed
            measurement noise.

    Returns:
        3-element tuple containing

        - A `input_batch_shape x m x n x d` tensor of training features.
        - A `target_batch_shape x m x n` tensor of training observations.
        - A `target_batch_shape x m x n` tensor observed measurement noise.
    """
    # make train_Y `batch_shape x m x n`
    train_Y = train_Y.transpose(-1, -2)
    # expand train_X to `batch_shape x m x n x d`
    train_X = train_X.unsqueeze(-3).expand(
        train_X.shape[:-2] + torch.Size([num_outputs]) + train_X.shape[-2:]
    )
    if train_Yvar is not None:
        # make train_Yvar `batch_shape x m x n`
        train_Yvar = train_Yvar.transpose(-1, -2)
    return train_X, train_Y, train_Yvar


def add_output_dim(X: Tensor, original_batch_shape: torch.Size) -> Tuple[Tensor, int]:
    r"""Insert the output dimension at the correct location.

    The trailing batch dimensions of X must match the original batch dimensions
    of the training inputs, but can also include extra batch dimensions.

    Args:
        X: A `(new_batch_shape) x (original_batch_shape) x n x d` tensor of
            features.
        original_batch_shape: the batch shape of the model's training inputs.

    Returns:
        2-element tuple containing

        - A `(new_batch_shape) x (original_batch_shape) x m x n x d` tensor of
            features.
        - The index corresponding to the output dimension.
    """
    X_batch_shape = X.shape[:-2]
    if len(X_batch_shape) > 0 and len(original_batch_shape) > 0:
        # check that X_batch_shape supports broadcasting or augments
        # original_batch_shape with extra batch dims
        try:
            torch.broadcast_shapes(X_batch_shape, original_batch_shape)
        except RuntimeError:
            raise RuntimeError(
                "The trailing batch dimensions of X must match the trailing "
                "batch dimensions of the training inputs."
            )
    # insert `m` dimension
    X = X.unsqueeze(-3)
    output_dim_idx = max(len(original_batch_shape), len(X_batch_shape))
    return X, output_dim_idx


def check_no_nans(Z: Tensor) -> None:
    r"""Check that tensor does not contain NaN values.

    Raises an InputDataError if `Z` contains NaN values.

    Args:
        Z: The input tensor.
    """
    if torch.any(torch.isnan(Z)).item():
        raise InputDataError("Input data contains NaN values.")


def check_min_max_scaling(
    X: Tensor,
    strict: bool = False,
    atol: float = 1e-2,
    raise_on_fail: bool = False,
    ignore_dims: Optional[List[int]] = None,
) -> None:
    r"""Check that tensor is normalized to the unit cube.

    Args:
        X: A `batch_shape x n x d` input tensor. Typically the training inputs
            of a model.
        strict: If True, require `X` to be scaled to the unit cube (rather than
            just to be contained within the unit cube).
        atol: The tolerance for the boundary check. Only used if `strict=True`.
        raise_on_fail: If True, raise an exception instead of a warning.
        ignore_dims: Subset of dimensions where the min-max scaling check is omitted.
    """
    ignore_dims = ignore_dims or []
    check_dims = list(set(range(X.shape[-1])) - set(ignore_dims))
    if len(check_dims) == 0:
        return None

    with torch.no_grad():
        X_check = X[..., check_dims]
        Xmin = torch.min(X_check, dim=-1).values
        Xmax = torch.max(X_check, dim=-1).values
        msg = None
        if strict and max(torch.abs(Xmin).max(), torch.abs(Xmax - 1).max()) > atol:
            msg = "scaled"
        if torch.any(Xmin < -atol) or torch.any(Xmax > 1 + atol):
            msg = "contained"
        if msg is not None:
            msg = (
                f"Input data is not {msg} to the unit cube. "
                "Please consider min-max scaling the input data."
            )
            if raise_on_fail:
                raise InputDataError(msg)
            warnings.warn(msg, InputDataWarning)


def check_standardization(
    Y: Tensor,
    atol_mean: float = 1e-2,
    atol_std: float = 1e-2,
    raise_on_fail: bool = False,
) -> None:
    r"""Check that tensor is standardized (zero mean, unit variance).

    Args:
        Y: The input tensor of shape `batch_shape x n x m`. Typically the
            train targets of a model. Standardization is checked across the
            `n`-dimension.
        atol_mean: The tolerance for the mean check.
        atol_std: The tolerance for the std check.
        raise_on_fail: If True, raise an exception instead of a warning.
    """
    with torch.no_grad():
        Ymean, Ystd = torch.mean(Y, dim=-2), torch.std(Y, dim=-2)
        if torch.abs(Ymean).max() > atol_mean or torch.abs(Ystd - 1).max() > atol_std:
            msg = (
                "Input data is not standardized. Please consider scaling the "
                "input to zero mean and unit variance."
            )
            if raise_on_fail:
                raise InputDataError(msg)
            warnings.warn(msg, InputDataWarning)


def validate_input_scaling(
    train_X: Tensor,
    train_Y: Tensor,
    train_Yvar: Optional[Tensor] = None,
    raise_on_fail: bool = False,
    ignore_X_dims: Optional[List[int]] = None,
) -> None:
    r"""Helper function to validate input data to models.

    Args:
        train_X: A `n x d` or `batch_shape x n x d` (batch mode) tensor of
            training features.
        train_Y: A `n x m` or `batch_shape x n x m` (batch mode) tensor of
            training observations.
        train_Yvar: A `batch_shape x n x m` or `batch_shape x n x m` (batch mode)
            tensor of observed measurement noise.
        raise_on_fail: If True, raise an error instead of emitting a warning
            (only for normalization/standardization checks, an error is always
            raised if NaN values are present).
        ignore_X_dims: For this subset of dimensions from `{1, ..., d}`, ignore the
            min-max scaling check.

    This function is typically called inside the constructor of standard BoTorch
    models. It validates the following:
    (i) none of the inputs contain NaN values
    (ii) the training data (`train_X`) is normalized to the unit cube for all
    dimensions except those in `ignore_X_dims`.
    (iii) the training targets (`train_Y`) are standardized (zero mean, unit var)
    No checks (other than the NaN check) are performed for observed variances
    (`train_Yvar`) at this point.
    """
    if settings.validate_input_scaling.off():
        return
    check_no_nans(train_X)
    check_no_nans(train_Y)
    if train_Yvar is not None:
        check_no_nans(train_Yvar)
        if torch.any(train_Yvar < 0):
            raise InputDataError("Input data contains negative variances.")
    check_min_max_scaling(
        X=train_X, raise_on_fail=raise_on_fail, ignore_dims=ignore_X_dims
    )
    check_standardization(Y=train_Y, raise_on_fail=raise_on_fail)


def mod_batch_shape(module: Module, names: List[str], b: int) -> None:
    r"""Recursive helper to modify gpytorch modules' batch shape attribute.

    Modifies the module in-place.

    Args:
        module: The module to be modified.
        names: The list of names to access the attribute. If the full name of
            the module is `"module.sub_module.leaf_module"`, this will be
            `["sub_module", "leaf_module"]`.
        b: The new size of the last element of the module's `batch_shape`
            attribute.
    """
    if len(names) == 0:
        return
    m = getattr(module, names[0])
    if len(names) == 1 and hasattr(m, "batch_shape") and len(m.batch_shape) > 0:
        m.batch_shape = m.batch_shape[:-1] + torch.Size([b] if b > 0 else [])
    else:
        mod_batch_shape(module=m, names=names[1:], b=b)


@contextmanager
def gpt_posterior_settings():
    r"""Context manager for settings used for computing model posteriors."""
    with ExitStack() as es:
        if gpt_settings.debug.is_default():
            es.enter_context(gpt_settings.debug(False))
        if gpt_settings.fast_pred_var.is_default():
            es.enter_context(gpt_settings.fast_pred_var())
        es.enter_context(
            gpt_settings.detach_test_caches(settings.propagate_grads.off())
        )
        yield


class fantasize(_Flag):
    r"""A flag denoting whether we are currently in a `fantasize` context."""
    _state: bool = False
