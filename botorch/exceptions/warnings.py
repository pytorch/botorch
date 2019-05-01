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


class SamplingWarning(BotorchWarning):
    r"""Sampling releated warnings."""

    pass
