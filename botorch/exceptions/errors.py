#!/usr/bin/env python3


class BotorchError(Exception):
    """Base botorch exception"""

    pass


class CandidateGenerationError(BotorchError):
    """Exception raised during generating candidates"""

    pass
