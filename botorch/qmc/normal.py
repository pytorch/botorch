#!/usr/bin/env python3

r"""
Quasi Monte-Carlo sampling from Normal distributions.

References:

.. [Pages2018numprob]
    G. Pages. Numerical Probability: An Introduction with Applications to
    Finance. Universitext. Springer International Publishing, 2018.
"""

import math
from typing import List, Optional, Union

import numpy as np
from scipy.stats import norm

from .sobol import SobolEngine


class NormalQMCEngine:
    r"""Engine for qMC sampling from a Multivariate Normal `N(0, I_d)`.

    By default, this implementation uses Box-Muller transformed Sobol samples
    following pg. 123 in [Pages2018numprob]_. To use the inverse transform
    instead, set `inv_transform=True`.

    Example:
        >>> engine = NormalQMCEngine(3, inv_transform=True)
        >>> samples = engine.draw(10)
    """

    def __init__(
        self, d: int, seed: Optional[int] = None, inv_transform: bool = False
    ) -> None:
        r"""Engine for drawing qMC samples from a multivariate normal `N(0, I_d)`.

        Args:
            d: The dimension of the samples.
            seed: The seed with which to seed the random number generator of the
                underlying SobolEngine.
            inv_transform: If True, use inverse transform instead of Box-Muller.
        """
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
        r"""Draw n qMC samples from the standard Normal.

        Args:
            n: The number of samples.

        Returns:
            The samples as a numpy array.
        """
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


class MultivariateNormalQMCEngine:
    r"""Engine for qMC sampling from a multivariate Normal `N(\mu, \Sigma)`.

    By default, this implementation uses Box-Muller transformed Sobol samples
    following pg. 123 in [Pages2018numprob]_. To use the inverse transform
    instead, set `inv_transform=True`.

    Example:
        >>> mean = torch.tensor([1.0, 2.0])
        >>> cov = torch.tensor([[1.0, 0.25]. [0.25, 2.0]])
        >>> engine = MultivariateNormalQMCEngine(mean, cov)
        >>> samples = engine.draw(10)
    """

    def __init__(
        self,
        mean: Union[float, List[float], np.ndarray],
        cov: Union[float, List[List[float]], np.ndarray],
        seed: Optional[int] = None,
        inv_transform: bool = False,
    ) -> None:
        r"""Engine for qMC sampling from a multivariate Normal `N(\mu, \Sigma)`.

        Args:
            mean: The mean vector.
            cov: The covariance matrix.
            seed: The seed with which to seed the random number generator of the
                underlying SobolEngine.
            inv_transform: If True, use inverse transform instead of Box-Muller.
        """
        # check for square/symmetric cov matrix and mean vector has the same d
        mean = np.array(mean, copy=False, ndmin=1)
        cov = np.array(cov, copy=False, ndmin=2)
        if not cov.shape[0] == cov.shape[1]:
            raise ValueError("Covariance matrix is not square.")
        if not mean.shape[0] == cov.shape[0]:
            raise ValueError("Dimension mismatch between mean and covariance.")
        if not np.allclose(cov, cov.transpose()):
            raise ValueError("Covariance matrix is not symmetric.")
        self._mean = mean
        self._normal_engine = NormalQMCEngine(
            d=mean.shape[0], seed=seed, inv_transform=inv_transform
        )
        # compute Cholesky decomp; if it fails, do the eigendecomposition
        try:
            self._corr_matrix = np.linalg.cholesky(cov).transpose()
        except np.linalg.LinAlgError:
            eigval, eigvec = np.linalg.eigh(cov)
            if not np.all(eigval >= -1.0e-8):
                raise ValueError("Covariance matrix not PSD.")
            eigval = np.clip(eigval, 0.0, None)
            self._corr_matrix = (eigvec * np.sqrt(eigval)).transpose()

    def draw(self, n: int = 1) -> np.ndarray:
        r"""Draw n qMC samples from the multivariate Normal.

        Args:
            n: The number of samples.

        Returns:
            The samples as a numpy array.
        """
        base_samples = self._normal_engine.draw(n)
        qmc_samples = base_samples @ self._corr_matrix + self._mean
        return qmc_samples
