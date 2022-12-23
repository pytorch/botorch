#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, List, Tuple, TypeVar

from memory_profiler import memory_usage

TReturn = TypeVar("TReturn")


def get_memory_usage_preserving_output(
    f: Callable[..., TReturn]
) -> Callable[..., Tuple[TReturn, List[float]]]:
    """
    Returns a function that returns both the output of the original function and the
    result of calling `memory_usage` on the function.

    Can be used as a decorator.

    Args:
        f: Function to be decorated

    Returns:
        Tuple of (output of f, output of memory_usage)

    Example:
        >>> from math import log
        >>> wrapped = get_memory_usage_preserving_output(log)
        >>> identity_fn_output, memory_output = wrapped(1)
        >>> identity_fn_output
        0.0
        >>> memory_output
        [214.00390625, 214.00390625, 214.00390625]
    """

    def new_fn(*args, **kwargs) -> Tuple[TReturn, List[float]]:
        writable = {}

        def f_with_side_effect(*args, **kwargs) -> None:
            result = f(*args, **kwargs)
            writable["result"] = result

        mem_usage = memory_usage(
            (f_with_side_effect, args, kwargs), include_children=True
        )
        f_output = writable["result"]
        return f_output, mem_usage

    return new_fn
