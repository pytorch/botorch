#!/usr/bin/env python3

import math
from typing import Optional

import numpy as np
from scipy.stats import norm

from .sobol import SobolEngine


class NormalQMCEngine:
    """Engine for drawing qMC samples from a multivariate normal N(0, I_d)

    Args:
        d: The dimension of the samples
        seed: The seed with which to seed the random number generator of the
            underlying SobolEngine.
        inv_transform: If True, use inverse transform instead of Box-Muller

    By default, this implementation uses Box-Muller transformed Sobol samples
    following pg. 123 in [1]. To use the inverse transform instead, set
    inv_transform to True.

    [1] G. Pages. Numerical Probability: An Introduction with Applications to Finance.
        Universitext. Springer International Publishing, 2018.

    """

    def __init__(
        self, d: int, seed: Optional[int] = None, inv_transform: bool = False
    ) -> None:
        self._d = d
        self._seed = seed
        self._inv_transform = inv_transform
        if inv_transform:
            sobol_dim = d
        else:
            # to apply Box-Muller, we need an even number of dimensions
            sobol_dim = 2 * math.ceil(d / 2)
        self._sobol_engine = SobolEngine(dimen=sobol_dim, scramble=True, seed=seed)

    def draw(self, n: int = 1) -> np.ndarray:
        """Draw n qMC samples from the standard Normal."""
        # get base samples
        samples = self._sobol_engine.draw(n)
        if self._inv_transform:
            # apply inverse transform (values to close to 0/1 result in inf values)
            return norm.ppf(0.5 + (1 - 1e-10) * (samples - 0.5))
        else:
            # apply Box-Muller transform (note: [1] indexes starting from 1)
            even = np.arange(0, samples.shape[-1], 2)
            Rs = np.sqrt(-2 * np.log(samples[:, even]))
            thetas = 2 * math.pi * samples[:, 1 + even]
            cos = np.cos(thetas)
            sin = np.sin(thetas)
            transf_samples = np.stack([Rs * cos, Rs * sin], -1).reshape(n, -1)
            # make sure we only return the number of dimension requested
            return transf_samples[:, : self._d]
