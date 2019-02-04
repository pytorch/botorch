#!/usr/bin/env python3


class BotorchWarning(Warning):
    """Base botorch warning"""

    pass


class BadInitialCandidatesWarning(BotorchWarning):
    """Warning issues if set of initial candidates for optimziation is bad."""

    pass
