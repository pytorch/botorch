#!/usr/bin/env python3


class BotorchWarning(Warning):
    """Base botorch warning."""

    pass


class BadInitialCandidatesWarning(BotorchWarning):
    """Warning issued if set of initial candidates for optimziation is bad."""

    pass


class SamplingWarning(BotorchWarning):
    """Sampling releated warnings."""

    pass
