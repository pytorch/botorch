#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

r"""
BoTorch settings.
"""

import typing  # noqa F401


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
