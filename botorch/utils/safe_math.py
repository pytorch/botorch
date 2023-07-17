#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Special implementations of mathematical functions that
solve numerical issues of naive implementations.

.. [Maechler2012accurate]
    M. Mächler. Accurately Computing log (1 - exp (-| a|))
        Assessed by the Rmpfr package. Technical report, 2012.
"""

from __future__ import annotations

import math

from typing import Tuple, Union

import torch
from botorch.exceptions import UnsupportedError
from botorch.utils.constants import get_constants_like
from torch import finfo, Tensor
from torch.nn.functional import softplus

_log2 = math.log(2)
_inv_sqrt_3 = math.sqrt(1 / 3)


# Unary ops
def exp(x: Tensor, **kwargs) -> Tensor:
    info = finfo(x.dtype)
    maxexp = get_constants_like(math.log(info.max) - 1e-4, x)
    return torch.exp(x.clip(max=maxexp), **kwargs)


def log(x: Tensor, **kwargs) -> Tensor:
    info = finfo(x.dtype)
    return torch.log(x.clip(min=info.tiny), **kwargs)


# Binary ops
def add(a: Tensor, b: Tensor, **kwargs) -> Tensor:
    _0 = get_constants_like(0, a)
    case = a.isinf() & b.isinf() & (a != b)
    return torch.where(case, _0, a + b)


def sub(a: Tensor, b: Tensor) -> Tensor:
    _0 = get_constants_like(0, a)
    case = (a.isinf() & b.isinf()) & (a == b)
    return torch.where(case, _0, a - b)


def div(a: Tensor, b: Tensor) -> Tensor:
    _0, _1 = get_constants_like(values=(0, 1), ref=a)
    case = ((a == _0) & (b == _0)) | (a.isinf() & a.isinf())
    return torch.where(case, torch.where(a != b, -_1, _1), a / torch.where(case, _1, b))


def mul(a: Tensor, b: Tensor) -> Tensor:
    _0 = get_constants_like(values=0, ref=a)
    case = (a.isinf() & (b == _0)) | (b.isinf() & (a == _0))
    return torch.where(case, _0, a * torch.where(case, _0, b))


def log1mexp(x: Tensor) -> Tensor:
    """Numerically accurate evaluation of log(1 - exp(x)) for x < 0.
    See [Maechler2012accurate]_ for details.
    """
    log2 = get_constants_like(values=_log2, ref=x)
    is_small = -log2 < x  # x < 0
    return torch.where(
        is_small,
        (-x.expm1()).log(),
        (-x.exp()).log1p(),
    )


def log1pexp(x: Tensor) -> Tensor:
    """Numerically accurate evaluation of log(1 + exp(x)).
    See [Maechler2012accurate]_ for details.
    """
    mask = x <= 18
    return torch.where(
        mask,
        (lambda z: z.exp().log1p())(x.masked_fill(~mask, 0)),
        (lambda z: z + (-z).exp())(x.masked_fill(mask, 0)),
    )


def logexpit(X: Tensor) -> Tensor:
    """Computes the logarithm of the expit (a.k.a. sigmoid) function."""
    return -log1pexp(-X)


def logdiffexp(log_a: Tensor, log_b: Tensor) -> Tensor:
    """Computes log(b - a) accurately given log(a) and log(b).
    Assumes, log_b > log_a, i.e. b > a > 0.

    Args:
        log_a (Tensor): The logarithm of a, assumed to be less than log_b.
        log_b (Tensor): The logarithm of b, assumed to be larger than log_a.

    Returns:
        A Tensor of values corresponding to log(b - a).
    """
    return log_b + log1mexp(log_a - log_b)


def logmeanexp(
    X: Tensor, dim: Union[int, Tuple[int, ...]], keepdim: bool = False
) -> Tensor:
    """Computes `log(mean(exp(X), dim=dim, keepdim=keepdim))`.

    Args:
        X: Values of which to compute the logmeanexp.
        dim: The dimension(s) over which to compute the mean.
        keepdim: If True, keeps the reduced dimensions.

    Returns:
        A Tensor of values corresponding to `log(mean(exp(X), dim=dim))`.
    """
    n = X.shape[dim] if isinstance(dim, int) else math.prod(X.shape[i] for i in dim)
    return torch.logsumexp(X, dim=dim, keepdim=keepdim) - math.log(n)


def log_softplus(x: Tensor, tau: Union[float, Tensor] = 1.0) -> Tensor:
    """Computes the logarithm of the softplus function with high numerical accuracy.

    Args:
        x: Input tensor, should have single or double precision floats.
        tau: Decreasing tau increases the tightness of the
            approximation to ReLU. Non-negative and defaults to 1.0.

    Returns:
        Tensor corresponding to `log(softplus(x))`.
    """
    check_dtype_float32_or_float64(x)
    tau = torch.as_tensor(tau, dtype=x.dtype, device=x.device)
    # cutoff chosen to achieve accuracy to machine epsilon
    upper = 16 if x.dtype == torch.float32 else 32
    lower = -15 if x.dtype == torch.float32 else -35
    mask = x / tau > lower
    return torch.where(
        mask,
        softplus(x.masked_fill(~mask, lower), beta=(1 / tau), threshold=upper).log(),
        x / tau + tau.log(),
    )


def smooth_amax(X: Tensor, tau: Union[float, Tensor] = 1e-3, dim: int = -1) -> Tensor:
    """Computes a smooth approximation to `max(X, dim=dim)`, i.e the maximum value of
    `X` over dimension `dim`, using the logarithm of the `l_(1/tau)` norm of `exp(X)`.
    Note that when `X = log(U)` is the *logarithm* of an acquisition utility `U`,

    `logsumexp(log(U) / tau) * tau = log(sum(U^(1/tau))^tau) = log(norm(U, ord=(1/tau))`

    Args:
        X: A Tensor from which to compute the smoothed amax.
        tau: Temperature parameter controlling the smooth approximation
            to max operator, becomes tighter as tau goes to 0. Needs to be positive.

    Returns:
        A Tensor of smooth approximations to `max(X, dim=dim)`.
    """
    # consider normalizing by log_n = math.log(X.shape[dim]) to reduce error
    return torch.logsumexp(X / tau, dim=dim) * tau  # ~ X.amax(dim=dim)


def check_dtype_float32_or_float64(X: Tensor) -> None:
    if X.dtype != torch.float32 and X.dtype != torch.float64:
        raise UnsupportedError(
            f"Only dtypes float32 and float64 are supported, but received {X.dtype}."
        )


def log_fatplus(x: Tensor, tau: Union[float, Tensor] = 1.0) -> Tensor:
    """Computes the logarithm of the fat-tailed softplus.

    NOTE: Separated out in case the complexity of the `log` implementation increases
    in the future.
    """
    return fatplus(x, tau=tau).log()


def fatplus(x: Tensor, tau: Union[float, Tensor] = 1.0) -> Tensor:
    """Computes a fat-tailed approximation to `ReLU(x) = max(x, 0)` by linearly
    combining a regular softplus function and the density function of a Cauchy
    distribution. The coefficient `alpha` of the Cauchy density is chosen to guarantee
    monotonicity and convexity.

    Args:
        x: A Tensor on whose values to compute the smoothed function.

    Returns:
        A Tensor of values of the fat-tailed softplus.
    """

    def _fatplus(x: Tensor) -> Tensor:
        alpha = 1e-1  # guarantees monotonicity and convexity (TODO: ref + Lemma 4)
        return softplus(x) + alpha * cauchy(x)

    return tau * _fatplus(x / tau)


def fatmax(X: Tensor, dim: int, tau: Union[float, Tensor] = 1.0) -> Tensor:
    """Computes a smooth approximation to amax(X, dim=dim) with a fat tail.

    Args:
        X: A Tensor from which to compute the smoothed amax.
        tau: Temperature parameter controlling the smooth approximation
            to max operator, becomes tighter as tau goes to 0. Needs to be positive.
        standardize: Toggles the temperature standardization of the smoothed function.

    Returns:
        A Tensor of smooth approximations to `max(X, dim=dim)` with a fat tail.
    """
    if X.shape[dim] == 1:
        return X.squeeze(dim)

    M = X.amax(dim=dim, keepdim=True)
    Y = (X - M) / tau  # NOTE: this would cause NaNs when X has Infs.
    M = M.squeeze(dim)
    return M + tau * cauchy(Y).sum(dim=dim).log()  # could change to mean


def log_fatmoid(X: Tensor, tau: Union[float, Tensor] = 1.0) -> Tensor:
    """Computes the logarithm of the fatmoid. Separated out in case the implementation
    of the logarithm becomes more complex in the future to ensure numerical stability.
    """
    return fatmoid(X, tau=tau).log()


def fatmoid(X: Tensor, tau: Union[float, Tensor] = 1.0) -> Tensor:
    """Computes a twice continuously differentiable approximation to the Heaviside
    step function with a fat tail, i.e. `O(1 / x^2)` as `x` goes to -inf.

    Args:
        X: A Tensor from which to compute the smoothed step function.
        tau: Temperature parameter controlling the smoothness of the approximation.

    Returns:
        A tensor of fat-tailed approximations to the Heaviside step function.
    """
    X = X / tau
    m = _inv_sqrt_3  # this defines the inflection point
    return torch.where(
        X < 0,
        2 / 3 * cauchy(X - m),
        1 - 2 / 3 * cauchy(X + m),
    )


def cauchy(x: Tensor) -> Tensor:
    """Computes a Lorentzian, i.e. an un-normalized Cauchy density function."""
    return 1 / (1 + x.square())


def sigmoid(X: Tensor, log: bool = False, fat: bool = False) -> Tensor:
    """A sigmoid function with an optional fat tail and evaluation in log space for
    better numerical behavior. Notably, the fat-tailed sigmoid can be used to remedy
    numerical underflow problems in the value and gradient of the canonical sigmoid.

    Args:
        X: The Tensor on which to evaluate the sigmoid.
        log: Toggles the evaluation of the log sigmoid.
        fat: Toggles the evaluation of the fat-tailed sigmoid.

    Returns:
        A Tensor of (log-)sigmoid values.
    """
    Y = log_fatmoid(X) if fat else logexpit(X)
    return Y if log else Y.exp()
