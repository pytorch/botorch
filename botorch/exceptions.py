#!/usr/bin/env python3


class BotorchError(Exception):
    """Base botorch exception"""

    pass


class CandidateGenerationError(BotorchError):
    """Exception raised during generating candidates"""

    pass


class BadInitialCandidatesError(CandidateGenerationError):
    """Exception raised if set of initial candidates for optimziation is bad """

    pass
