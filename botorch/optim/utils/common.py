#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""General-purpose optimization utilities."""

from __future__ import annotations

from inspect import signature
from logging import debug as logging_debug
from typing import Any, Callable, Optional, Tuple
from warnings import warn_explicit, WarningMessage

import numpy as np
from linear_operator.utils.errors import NanError, NotPSDError

TNone = type(None)


class _TDefault:
    pass


DEFAULT = _TDefault()


def _filter_kwargs(function: Callable, **kwargs: Any) -> Any:
    r"""Filter out kwargs that are not applicable for a given function.
    Return a copy of given kwargs dict with only the required kwargs."""
    return {k: v for k, v in kwargs.items() if k in signature(function).parameters}


def _handle_numerical_errors(
    error: RuntimeError, x: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(error, NotPSDError):
        raise error
    error_message = error.args[0] if len(error.args) > 0 else ""
    if (
        isinstance(error, NanError)
        or "singular" in error_message  # old pytorch message
        or "input is not positive-definite" in error_message  # since pytorch #63864
    ):
        return np.full((), "nan", dtype=x.dtype), np.full_like(x, "nan")
    raise error  # pragma: nocover


def _warning_handler_template(
    w: WarningMessage,
    debug: Optional[Callable[[WarningMessage], bool]] = None,
    rethrow: Optional[Callable[[WarningMessage], bool]] = None,
) -> bool:
    r"""Helper for making basic warning handlers. Typically used with functools.partial.

    Args:
        w: The WarningMessage to be resolved and filtered out or returned unresolved.
        debug: Optional callable used to specify that a warning should be
            resolved as a logging statement at the DEBUG level.
        rethrow: Optional callable used to specify that a warning should be
            resolved by rethrowing the warning.

    Returns:
        Boolean indicating whether or not the warning message was resolved.
    """
    if debug and debug(w):
        logging_debug(str(w.message))
        return True

    if rethrow and rethrow(w):
        warn_explicit(str(w.message), w.category, w.filename, w.lineno)
        return True

    return False
