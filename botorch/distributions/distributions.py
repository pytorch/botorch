#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Probability Distributions.

This is modified from https://github.com/probtorch/pytorch/pull/143 and
https://github.com/tensorflow/probability/blob/v0.11.1/
tensorflow_probability/python/distributions/kumaraswamy.py.

TODO: replace with PyTorch version once the PR is up and landed.
"""

from __future__ import annotations

from typing import Tuple, Union

import torch
from torch import Tensor
from torch.distributions import constraints
from torch.distributions.gumbel import euler_constant
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import AffineTransform, PowerTransform
from torch.distributions.uniform import Uniform
from torch.distributions.utils import broadcast_all


def _weighted_logsumexp(
    logx: Tensor, w: Tensor, dim: int, keepdim: bool = False
) -> Tuple[Tensor, Tensor]:
    log_absw_x = logx + w.abs().log()
    max_log_absw_x = torch.max(log_absw_x, dim=dim, keepdim=True).values
    max_log_absw_x = torch.where(
        torch.isinf(max_log_absw_x),
        torch.zeros([], dtype=max_log_absw_x.dtype),
        max_log_absw_x,
    )
    wx_over_max_absw_x = torch.sign(w) * torch.exp(log_absw_x - max_log_absw_x)
    sum_wx_over_max_absw_x = wx_over_max_absw_x.sum(dim=dim, keepdim=keepdim)
    if not keepdim:
        max_log_absw_x = max_log_absw_x.squeeze(dim)
    sgn = torch.sign(sum_wx_over_max_absw_x)
    lswe = max_log_absw_x + torch.log(sgn * sum_wx_over_max_absw_x)
    return lswe, sgn


def _log_moments(a: Tensor, b: Tensor, n: int) -> Tensor:
    r"""Computes the logarithm of the n-th moment of the Kumaraswamy distribution.
    Args:
        a: 1st concentration parameter of the distribution
                (often referred to as alpha)
        b: 2nd concentration parameter of the distribution
            (often referred to as beta)
        n: The moment number

    Returns:
        The logarithm of the n-th moment of the Kumaraswamy distribution.
    """
    arg1 = 1 + n / a
    log_value = torch.lgamma(arg1) + torch.lgamma(b) - torch.lgamma(arg1 + b)
    return b.log() + log_value


class Kumaraswamy(TransformedDistribution):
    r"""A Kumaraswamy distribution.

    Example::

        >>> m = Kumaraswamy(torch.Tensor([1.0]), torch.Tensor([1.0]))
        >>> m.sample()  # sample from a Kumaraswamy distribution
        tensor([ 0.1729])

    Args:
        concentration1: 1st concentration parameter of the distribution
            (often referred to as alpha)
        concentration0: 2nd concentration parameter of the distribution
            (often referred to as beta)
    """
    arg_constraints = {
        "concentration1": constraints.positive,
        "concentration0": constraints.positive,
    }
    support = constraints.unit_interval
    has_rsample = True

    def __init__(
        self,
        concentration1: Union[float, Tensor],
        concentration0: Union[float, Tensor],
        validate_args: bool = False,
    ):
        self.concentration1, self.concentration0 = broadcast_all(
            concentration1, concentration0
        )
        base_dist = Uniform(
            torch.full_like(self.concentration0, 0.0),
            torch.full_like(self.concentration0, 1.0),
        )
        transforms = [
            AffineTransform(loc=1.0, scale=-1.0),
            PowerTransform(exponent=self.concentration0.reciprocal()),
            AffineTransform(loc=1.0, scale=-1.0),
            PowerTransform(exponent=self.concentration1.reciprocal()),
        ]
        super().__init__(base_dist, transforms, validate_args=validate_args)

    def expand(
        self, batch_shape: torch.Size, _instance: Kumaraswamy = None
    ) -> Kumaraswamy:
        new = self._get_checked_instance(Kumaraswamy, _instance)
        new.concentration1 = self.concentration1.expand(batch_shape)
        new.concentration0 = self.concentration0.expand(batch_shape)
        return super().expand(batch_shape, _instance=new)

    @property
    def mean(self) -> None:
        return _log_moments(a=self.concentration1, b=self.concentration0, n=1).exp()

    @property
    def variance(self) -> None:
        log_moment2 = _log_moments(a=self.concentration1, b=self.concentration0, n=2)
        log_moment1 = _log_moments(a=self.concentration1, b=self.concentration0, n=1)
        lswe, sgn = _weighted_logsumexp(
            logx=torch.stack([log_moment2, 2 * log_moment1], dim=-1),
            w=torch.tensor(
                [1.0, -1.0], dtype=log_moment1.dtype, device=log_moment1.device
            ),
            dim=-1,
        )
        return sgn * lswe.exp()

    def entropy(self) -> None:
        t1 = 1 - self.concentration1.reciprocal()
        t0 = 1 - self.concentration0.reciprocal()
        H0 = torch.digamma(self.concentration0 + 1) + euler_constant
        return (
            t0
            + t1 * H0
            - torch.log(self.concentration1)
            - torch.log(self.concentration0)
        )
