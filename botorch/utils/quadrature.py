# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple

import numpy as np
import torch

from torch import Tensor


def clenshaw_curtis_quadrature(
    deg: int,
    a: float = 0.0,
    b: float = 1.0,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Clenshaw-Curtis quadrature.

    This might be useful if we want to use Chebyshev interpolants for the evaluation
    of the component functions. We could even approximate the GP prior as a distribution
    over Chebyshev polynomials.

    Clenshaw-Curtis quadrature uses the same nodes as Chebyshev interpolants but for
    integration.

    Args:
        deg: Number of sample points and weights. Integrates poynomials of degree
            `deg - 1` exactly.
        a: Lower bound of the integration domain.
        b: Upper bound of the integration domain.
        dtype: Desired floating point type of the return Tensors.
        device: Desired device type of the return Tensors.

    Returns:
        A tuple of Clenshaw-Curtis quadrature nodes and weights of length order.
    """
    dtype = dtype if dtype is not None else torch.get_default_dtype()
    x, w = _clenshaw_curtis_quadrature(order=deg - 1)
    x = torch.as_tensor(x, dtype=dtype, device=device)
    w = torch.as_tensor(w, dtype=dtype, device=device)
    if not (a == 0 and b == 1):  # need to normalize for different domain
        x = (b - a) * x + a
        w = w * (b - a)
    return x, w


def higher_dimensional_quadrature(
    xs: Tuple[Tensor, ...], ws: Tuple[Tensor, ...]
) -> Tuple[Tensor, Tensor]:
    """
    Returns:
        A tuple of higher-dimensional quadrature nodes and weights. The nodes are
        `n^d x d`-dimensional, the weights are `n^d`-dimensional.
    """
    x = torch.cartesian_prod(*xs)
    w = torch.cartesian_prod(*ws).prod(-1)
    return x, w


def _clenshaw_curtis_quadrature(order: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Clenshaw-Curtis quadrature on integration domain of [0, 1], modified from ChaosPy.

    Args:
        order: Integrates poynomials of degree order.

    Returns:
        A tuple of Clenshaw-Curtis quadrature nodes and weights of length order + 1.
    """
    if order == 0:
        return np.array([0.5]), np.array([1.0])
    elif order == 1:
        return np.array([0.0, 1.0]), np.array([0.5, 0.5])

    theta = (order - np.arange(order + 1)) * np.pi / order
    abscissas = 0.5 * np.cos(theta) + 0.5

    steps = np.arange(1, order, 2)
    length = len(steps)
    remains = order - length

    beta = np.hstack(
        [2.0 / (steps * (steps - 2)), [1.0 / steps[-1]], np.zeros(remains)]
    )
    beta = -beta[:-1] - beta[:0:-1]

    gamma = -np.ones(order)
    gamma[length] += order
    gamma[remains] += order
    gamma /= order**2 - 1 + (order % 2)

    # original implementation:
    weights = np.fft.ihfft(beta + gamma)
    if max(weights.imag) > 1e-15:
        raise ValueError(
            "Clenshaw-Curtis quadrature weights are not real. Expected imaginary "
            f"values to be <1e-15, got {max(weights.imag)=}"
        )
    weights = weights.real
    weights = np.hstack([weights, weights[len(weights) - 2 + (order % 2) :: -1]]) / 2

    # implementation based on irfft:
    # weights = np.fft.irfft(beta + gamma, order)
    # weights = weights / 2
    # weights = np.hstack((weights, weights[0]))

    return abscissas, weights
