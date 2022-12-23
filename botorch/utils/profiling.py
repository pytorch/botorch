#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, List, Tuple, TypeVar

from memory_profiler import memory_usage

TReturn = TypeVar("TReturn")


def get_memory_usage_preserving_output(
    f: Callable[..., TReturn], *args, **kwargs
) -> Tuple[TReturn, List[float]]:
    """
    Calls `memory_usage` on a function and returns both the output of the function
    and the output of `memory_usage`.

    Args:
        f: Function passed to `memory_usage`, whose output is returned
        args: args passed to `f` through `memory_usage`
        kwargs: kwargs passed to `f` through `memory_usage`

    Returns:
        Tuple of (output of f, output of memory_usage)
    """
    writable = {}

    def f_with_side_effect(*args, **kwargs) -> None:
        result = f(*args, **kwargs)
        writable["result"] = result

    mem_usage = memory_usage((f_with_side_effect, args, kwargs), include_children=True)
    f_output = writable["result"]

    return f_output, mem_usage
