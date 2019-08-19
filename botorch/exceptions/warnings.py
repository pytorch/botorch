#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

r"""
Botorch Warnings.
"""


class BotorchWarning(Warning):
    r"""Base botorch warning."""

    pass


class BadInitialCandidatesWarning(BotorchWarning):
    r"""Warning issued if set of initial candidates for optimziation is bad."""

    pass


class InputDataWarning(BotorchWarning):
    r"""Warning raised when input data does not comply with conventions."""

    pass


class OptimizationWarning(BotorchWarning):
    r"""Optimization-releated warnings."""

    pass


class SamplingWarning(BotorchWarning):
    r"""Sampling related warnings."""

    pass


class BotorchTensorDimensionWarning(BotorchWarning):
    r"""Warning raised when a tensor possibly violates a botorch convention."""

    pass
