#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Core methods for building closures in torch and interfacing with numpy."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from functools import partial
from typing import Any

import numpy.typing as npt

import torch
from botorch.optim.utils import (
    _handle_numerical_errors,
    get_tensors_as_ndarray_1d,
    set_tensors_from_ndarray_1d,
)
from botorch.optim.utils.numpy_utils import as_ndarray
from botorch.utils.context_managers import zero_grad_ctx
from numpy import float64 as np_float64, full as np_full, zeros as np_zeros
from torch import Tensor


class ForwardBackwardClosure:
    r"""Wrapper for fused forward and backward closures."""

    def __init__(
        self,
        forward: Callable[[], Tensor],
        parameters: dict[str, Tensor],
        backward: Callable[[Tensor], None] = Tensor.backward,
        reducer: Callable[[Tensor], Tensor] | None = torch.sum,
        callback: Callable[[Tensor, Sequence[Tensor | None]], None] | None = None,
        context_manager: Callable = None,  # pyre-ignore [9]
    ) -> None:
        r"""Initializes a ForwardBackwardClosure instance.

        Args:
            closure: Callable that returns a tensor.
            parameters: A dictionary of tensors whose `grad` fields are to be returned.
            backward: Callable that takes the (reduced) output of `forward` and sets the
                `grad` attributes of tensors in `parameters`.
            reducer: Optional callable used to reduce the output of the forward pass.
            callback: Optional callable that takes the reduced output of `forward` and
                the gradients of `parameters` as positional arguments.
            context_manager: A ContextManager used to wrap each forward-backward call.
                When passed as `None`, `context_manager` defaults to a `zero_grad_ctx`
                that zeroes the gradients of `parameters` upon entry.
        """
        if context_manager is None:
            context_manager = partial(zero_grad_ctx, parameters)

        self.forward = forward
        self.backward = backward
        self.parameters = parameters
        self.reducer = reducer
        self.callback = callback
        self.context_manager = context_manager

    def __call__(self, **kwargs: Any) -> tuple[Tensor, tuple[Tensor | None, ...]]:
        with self.context_manager():
            values = self.forward(**kwargs)
            value = values if self.reducer is None else self.reducer(values)
            self.backward(value)

            grads = tuple(param.grad for param in self.parameters.values())
            if self.callback:
                self.callback(value, grads)

            return value, grads


class NdarrayOptimizationClosure:
    r"""Adds stateful behavior and a numpy.ndarray-typed API to a closure with an
    expected return type Tuple[Tensor, Union[Tensor, Sequence[Optional[Tensor]]]]."""

    def __init__(
        self,
        closure: Callable[[], tuple[Tensor, Sequence[Tensor | None]]],
        parameters: dict[str, Tensor],
        as_array: Callable[[Tensor], npt.NDArray] = None,  # pyre-ignore [9]
        get_state: Callable[[], npt.NDArray] = None,  # pyre-ignore [9]
        set_state: Callable[[npt.NDArray], None] = None,  # pyre-ignore [9]
        fill_value: float = 0.0,
        persistent: bool = True,
    ) -> None:
        r"""Initializes a NdarrayOptimizationClosure instance.

        Args:
            closure: A ForwardBackwardClosure instance.
            parameters: A dictionary of tensors representing the closure's state.
                Expected to correspond with the first `len(parameters)` optional
                gradient tensors returned by `closure`.
            as_array: Callable used to convert tensors to ndarrays.
            get_state: Callable that returns the closure's state as an ndarray. When
                passed as `None`, defaults to calling `get_tensors_as_ndarray_1d`
                on `closure.parameters` while passing `as_array` (if given by the user).
            set_state: Callable that takes a 1-dimensional ndarray and sets the
                closure's state. When passed as `None`, `set_state` defaults to
                calling `set_tensors_from_ndarray_1d` with `closure.parameters` and
                a given ndarray.
            fill_value: Fill value for parameters whose gradients are None. In most
                cases, `fill_value` should either be zero or NaN.
            persistent: Boolean specifying whether an ndarray should be retained
                as a persistent buffer for gradients.
        """
        if get_state is None:
            # Note: Numpy supports copying data between ndarrays with different dtypes.
            # Hence, our default behavior need not coerce the ndarray representations
            # of tensors in `parameters` to float64 when copying over data.
            _as_array = as_ndarray if as_array is None else as_array
            get_state = partial(
                get_tensors_as_ndarray_1d,
                tensors=parameters,
                dtype=np_float64,
                as_array=_as_array,
            )

        if as_array is None:  # per the note, do this after resolving `get_state`
            as_array = partial(as_ndarray, dtype=np_float64)

        if set_state is None:
            set_state = partial(set_tensors_from_ndarray_1d, parameters)

        self.closure = closure
        self.parameters = parameters

        self.as_array = as_ndarray
        self._get_state = get_state
        self._set_state = set_state

        self.fill_value = fill_value
        self.persistent = persistent
        self._gradient_ndarray: npt.NDArray | None = None

    def __call__(
        self, state: npt.NDArray | None = None, **kwargs: Any
    ) -> tuple[npt.NDArray, npt.NDArray]:
        if state is not None:
            self.state = state

        try:
            value_tensor, grad_tensors = self.closure(**kwargs)
            value = self.as_array(value_tensor)
            grads = self._get_gradient_ndarray(fill_value=self.fill_value)
            index = 0
            for param, grad in zip(self.parameters.values(), grad_tensors):
                size = param.numel()
                if grad is not None:
                    grads[index : index + size] = self.as_array(grad.view(-1))
                index += size
        except RuntimeError as e:
            value, grads = _handle_numerical_errors(e, x=self.state, dtype=np_float64)

        return value, grads

    @property
    def state(self) -> npt.NDArray:
        return self._get_state()

    @state.setter
    def state(self, state: npt.NDArray) -> None:
        self._set_state(state)

    def _get_gradient_ndarray(self, fill_value: float | None = None) -> npt.NDArray:
        if self.persistent and self._gradient_ndarray is not None:
            if fill_value is not None:
                self._gradient_ndarray.fill(fill_value)
            return self._gradient_ndarray

        size = sum(param.numel() for param in self.parameters.values())
        array = (
            np_zeros(size, dtype=np_float64)
            if fill_value is None or fill_value == 0.0
            else np_full(size, fill_value, dtype=np_float64)
        )
        if self.persistent:
            self._gradient_ndarray = array

        return array
