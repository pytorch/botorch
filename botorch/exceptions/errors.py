#!/usr/bin/env python3

"""
Botorch Errors.
"""


class BotorchError(Exception):
    """Base botorch exception."""

    pass


class CandidateGenerationError(BotorchError):
    """Exception raised during generating candidates."""

    pass


class UnsupportedError(BotorchError):
    """Currently unsupported feature."""

    pass
