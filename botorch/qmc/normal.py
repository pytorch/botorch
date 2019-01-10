#!/usr/bin/env python3

import math
from typing import Optional

import numpy as np

from .sobol import SobolEngine


class NormalQMCEngine:
    """Engine for drawing qMC samples from the standard Normal

    Args:
        seed: The seed with which to seed the random number generator of the
            underlying SobolEngine.

    This implementation uses Box-Muller transformed Sobol samples.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._sobol_engine = SobolEngine(dimen=2, scramble=True, seed=seed)

    def draw(self, n: int = 1) -> np.ndarray:
        """Draw n qMC samples from the standard Normal."""
        # get base samples
        samples = self._sobol_engine.draw(math.ceil(n / 2))
        # apply Box-Muller transform
        R = np.sqrt(-2 * np.log(samples[:, 0]))
        theta = 2 * math.pi * samples[:, 1]
        Z = np.concatenate([R * np.cos(theta), R * np.sin(theta)])
        np.random.shuffle(Z)
        return Z[:n]
