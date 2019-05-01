#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

r"""
Botorch Errors.
"""


class BotorchError(Exception):
    r"""Base botorch exception."""

    pass


class CandidateGenerationError(BotorchError):
    r"""Exception raised during generating candidates."""

    pass


class UnsupportedError(BotorchError):
    r"""Currently unsupported feature."""

    pass
