#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Optional

import torch
from gpytorch.constraints import Interval, Positive
from gpytorch.kernels import Kernel
from gpytorch.priors import Prior
from torch import Tensor


class ExpDecayKernel(Kernel):
    r"""
    Computes a covariance matrix based on the exponential decay kernel
    between inputs :math:`\mathbf{x_1}` and :math:`\mathbf{x_2}` (we expect d = 1):

    .. math::
        \begin{equation*}
            k_\text{expdecay}(\mathbf{x_1}, \mathbf{x_2}) = w +
            \frac{\beta^{\alpha}}{(\mathbf{x_1} + \mathbf{x_2} + \beta)^{\alpha}}.
        \end{equation*}

    where

    * :math:`w` is an :attr:`offset` parameter, `\beta` is an :attr:'lenthscale'
            parameter `\alpha` is an :attr:`power` parameter
    Args:
        :attr:`lengthscale_constraint` (Constraint, optional):
            Constraint to place on lengthscale parameter. Default: `Positive`.
        :attr:`lengthscale_prior` (:class:`gpytorch.priors.Prior`):
            Prior over the lengthscale parameter (default `None`).
        :attr:`power_constraint` (Constraint, optional):
            Constraint to place on power parameter. Default: `Positive`.
        :attr:`power_prior` (:class:`gpytorch.priors.Prior`):
            Prior over the power parameter (default `None`).
        :attr:`offset_constraint` (Constraint, optional):
            Constraint to place on offset parameter. Default: `Positive`.
        :attr:`active_dims` (list):
            List of data dimensions to operate on.
            `len(active_dims)` should equal `num_dimensions`.
    """

    def __init__(
        self,
        power_prior: Optional[Prior] = None,
        offset_prior: Optional[Prior] = None,
        power_constraint: Optional[Interval] = None,
        offset_constraint: Optional[Interval] = None,
        **kwargs
    ):
        super().__init__(has_lengthscale=True, **kwargs)

        if power_constraint is None:
            power_constraint = Positive()
        if offset_constraint is None:
            offset_constraint = Positive()

        self.register_parameter(
            name="raw_power",
            parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1)),
        )

        self.register_parameter(
            name="raw_offset",
            parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1)),
        )

        if power_prior is not None:
            self.register_prior(
                "power_prior",
                power_prior,
                lambda: self.power,
                lambda v: self._set_power(v),
            )
        self.register_constraint("raw_power", offset_constraint)

        if offset_prior is not None:
            self.register_prior(
                "offset_prior",
                offset_prior,
                lambda: self.offset,
                lambda v: self._set_offset(v),
            )

        self.register_constraint("raw_offset", offset_constraint)

    @property
    def power(self) -> Tensor:
        return self.raw_power_constraint.transform(self.raw_power)

    @power.setter
    def power(self, value: Tensor) -> None:
        self._set_power(value)

    def _set_power(self, value: Tensor) -> None:
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_power)
        self.initialize(raw_power=self.raw_power_constraint.inverse_transform(value))

    @property
    def offset(self) -> Tensor:
        return self.raw_offset_constraint.transform(self.raw_offset)

    @offset.setter
    def offset(self, value: Tensor) -> None:
        self._set_offset(value)

    def _set_offset(self, value: Tensor) -> None:
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_offset)
        self.initialize(raw_offset=self.raw_offset_constraint.inverse_transform(value))

    def forward(self, x1: Tensor, x2: Tensor, **params) -> Tensor:
        offset = self.offset.view(*self.batch_shape, 1, 1)
        power = self.power.view(*self.batch_shape, 1, 1)
        x1_ = x1.div(self.lengthscale)
        x2_ = x2.div(self.lengthscale)
        diff = self.covar_dist(x1_, -x2_, **params)
        res = offset + (diff + 1).pow(-power)
        return res
