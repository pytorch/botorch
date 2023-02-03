#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Utilities for optimization.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Dict, Generator, Iterable, NamedTuple, Optional, Union

from torch import device as Device, dtype as Dtype, Tensor
from torch.nn import Module


class TensorCheckpoint(NamedTuple):
    values: Tensor
    device: Optional[Device] = None
    dtype: Optional[Dtype] = None


@contextmanager
def delattr_ctx(
    instance: object, *attrs: str, enforce_hasattr: bool = False
) -> Generator[None, None, None]:
    r"""Contextmanager for temporarily deleting attributes."""
    try:
        cache = {}
        for key in attrs:
            if hasattr(instance, key):
                cache[key] = getattr(instance, key)
                delattr(instance, key)
            elif enforce_hasattr:
                raise ValueError(
                    f"Attribute {key} missing from {type(instance)} instance."
                )
        yield
    finally:
        for key, cached_val in cache.items():
            setattr(instance, key, cached_val)


@contextmanager
def requires_grad_ctx(
    module: Module, assignments: Dict[str, bool]
) -> Generator[None, None, None]:
    r"""Contextmanager for temporarily setting the requires_grad field of a module's
    parameters."""
    try:
        cache = {}
        for name, mode in assignments.items():
            parameter = module.get_parameter(name)
            cache[name] = parameter.requires_grad
            parameter.requires_grad_(mode)
        yield
    finally:
        for name, mode in cache.items():
            module.get_parameter(name).requires_grad_(mode)


@contextmanager
def parameter_rollback_ctx(
    parameters: Dict[str, Tensor],
    checkpoint: Optional[Dict[str, TensorCheckpoint]] = None,
    **tkwargs: Any,
) -> Generator[Dict[str, TensorCheckpoint], None, None]:
    r"""Contextmanager that exits by rolling back a module's state_dict.

    Args:
        module: Module instance.
        name_filter: Optional Boolean function used to filter items by name.
        checkpoint: Optional cache of values and tensor metadata specifying the rollback
            state for the module (or some subset thereof).
        **tkwargs: Keyword arguments passed to `torch.Tensor.to` when copying data from
            each tensor in `module.state_dict()` to the internally created checkpoint.
            Only adhered to when the `checkpoint` argument is None.

    Yields:
        A dictionary of TensorCheckpoints for the module's state_dict. Any in-places
        changes to the checkpoint will be observed at rollback time. If the checkpoint
        is cleared, no rollback will occur.
    """
    # Create copies of the orginal values
    if checkpoint is None:
        checkpoint = {
            name: TensorCheckpoint(
                values=param.detach().to(**tkwargs).clone(),
                device=param.device,
                dtype=param.dtype,
            )
            for name, param in parameters.items()
        }

    try:  # yield the checkpoint dictionary to the user
        yield checkpoint
    finally:  # restore original values of tracked parameters
        if checkpoint:
            for name, param in parameters.items():
                if name in checkpoint:
                    values, device, dtype = checkpoint[name]
                    param.data.copy_(values.to(device=device, dtype=dtype))


@contextmanager
def module_rollback_ctx(
    module: Module,
    name_filter: Optional[Callable[[str], bool]] = None,
    checkpoint: Optional[Dict[str, TensorCheckpoint]] = None,
    **tkwargs: Any,
) -> Generator[Dict[str, TensorCheckpoint], None, None]:
    r"""Contextmanager that exits by rolling back a module's state_dict.

    Args:
        module: Module instance.
        name_filter: Optional Boolean function used to filter items by name.
        checkpoint: Optional cache of values and tensor metadata specifying the rollback
            state for the module (or some subset thereof).
        **tkwargs: Keyword arguments passed to `torch.Tensor.to` when copying data from
            each tensor in `module.state_dict()` to the internally created checkpoint.
            Only adhered to when the `checkpoint` argument is None.

    Yields:
        A dictionary of TensorCheckpoints for the module's state_dict. Any in-places
        changes to the checkpoint will be observed at rollback time. If the checkpoint
        is cleared, no rollback will occur.
    """
    # Create copies of the orginal values
    if checkpoint is None:
        checkpoint = {
            name: TensorCheckpoint(
                values=values.detach().to(**tkwargs).clone(),
                device=values.device,
                dtype=values.dtype,
            )
            for name, values in module.state_dict().items()
            if name_filter is None or name_filter(name)
        }

    try:  # yield the checkpoint dictionary to the user
        yield checkpoint
    finally:  # restore original values of tracked parameters
        if checkpoint:
            state_dict = module.state_dict()
            for key, (values, device, dtype) in checkpoint.items():
                tnsr = state_dict.get(key)
                if tnsr is None:
                    state_dict[key] = values.to(device=device, dtype=dtype)
                else:
                    tnsr[...] = values.to(device=device, dtype=dtype)

            module.load_state_dict(state_dict)


@contextmanager
def zero_grad_ctx(
    parameters: Union[Dict[str, Tensor], Iterable[Tensor]],
    zero_on_enter: bool = True,
    zero_on_exit: bool = False,
) -> Generator[None, None, None]:
    def zero_() -> None:
        for param in (
            parameters.values() if isinstance(parameters, dict) else parameters
        ):
            if param.grad is not None:
                param.grad.zero_()

    if zero_on_enter:
        zero_()

    try:
        yield
    finally:
        if zero_on_exit:
            zero_()
