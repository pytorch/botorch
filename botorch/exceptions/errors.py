#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Botorch Errors.
"""


class BotorchError(Exception):
    r"""Base botorch exception."""

    pass


class CandidateGenerationError(BotorchError):
    r"""Exception raised during generating candidates."""

    pass


class InputDataError(BotorchError):
    r"""Exception raised when input data does not comply with conventions."""

    pass


class UnsupportedError(BotorchError):
    r"""Currently unsupported feature."""

    pass


class BotorchTensorDimensionError(BotorchError):
    r"""Exception raised when a tensor violates a botorch convention."""

    pass
