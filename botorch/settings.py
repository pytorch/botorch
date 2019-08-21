#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

r"""
BoTorch settings.
"""

import typing  # noqa F401
import warnings

from .exceptions import BotorchWarning


class _Flag:
    r"""Base class for context managers for a binary setting."""

    _state: bool = False

    @classmethod
    def on(cls) -> bool:
        return cls._state

    @classmethod
    def off(cls) -> bool:
        return not cls._state

    @classmethod
    def _set_state(cls, state: bool) -> None:
        cls._state = state

    def __init__(self, state: bool = True) -> None:
        self.prev = self.__class__.on()
        self.state = state

    def __enter__(self) -> None:
        self.__class__._set_state(self.state)

    def __exit__(self, *args) -> None:
        self.__class__._set_state(self.prev)


class propagate_grads(_Flag):
    r"""Flag for propagating gradients to model training inputs / training data.

    When set to `True`, gradients will be propagated to the training inputs.
    This is useful in particular for propating gradients through fantasy models.
    """

    _state: bool = False


def suppress_botorch_warnings(suppress: bool) -> None:
    r"""Set botorch warning filter.

    Args:
        state: A boolean indicating whether warnings should be prints
    """
    warnings.simplefilter("ignore" if suppress else "default", BotorchWarning)


class debug(_Flag):
    r"""Flag for printing verbose BotorchWarnings.

    When set to `True`, verbose `BotorchWarning`s will be printed for debuggability.
    Warnings that are not subclasses of `BotorchWarning` will not be affected by
    this context_manager.
    """

    _state: bool = False
    suppress_botorch_warnings(suppress=not _state)

    @classmethod
    def _set_state(cls, state: bool) -> None:
        cls._state = state
        suppress_botorch_warnings(suppress=not cls._state)
