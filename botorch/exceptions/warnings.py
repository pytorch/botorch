#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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


class CostAwareWarning(BotorchWarning):
    r"""Warning raised in the context of cost-aware acquisition strategies."""

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
