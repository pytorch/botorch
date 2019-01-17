#!/usr/bin/env python3

import math
from typing import Optional

import numpy as np

from .sobol import SobolEngine


class NormalQMCEngine:
    """Engine for drawing qMC samples from a multivariate normal N(0, I_d)

    Args:
        d: The dimension of the samples
        seed: The seed with which to seed the random number generator of the
            underlying SobolEngine.

    This implementation uses Box-Muller transformed Sobol samples following
    pg. 123 in [1]

    [1] G. Pages. Numerical Probability: An Introduction with Applications to Finance.
        Universitext. Springer International Publishing, 2018.

    """

    def __init__(self, d: int, seed: Optional[int] = None) -> None:
        self._d = d
        self._seed = seed
        # to apply Box-Muller, we need an even number of dimensions
        sobol_dim = 2 * math.ceil(d / 2)
        self._sobol_engine = SobolEngine(dimen=sobol_dim, scramble=True, seed=seed)

    def draw(self, n: int = 1) -> np.ndarray:
        """Draw n qMC samples from the standard Normal."""
        # get base samples
        samples = self._sobol_engine.draw(n)
        # apply Box-Muller transform
        even = np.arange(0, samples.shape[-1], 2)  # note: [1] indexes starting from 1
        Rs = np.sqrt(-2 * np.log(samples[:, even]))
        thetas = 2 * math.pi * samples[:, 1 + even]
        cos = np.cos(thetas)
        sin = np.sin(thetas)
        transf_samples = np.stack([Rs * cos, Rs * sin], -1).reshape(n, -1)
        # make sure we only return the number of dimension requested
        return transf_samples[:, : self._d]
