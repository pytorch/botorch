#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from copy import deepcopy
from typing import Any, Optional

import torch
from botorch.exceptions import UnsupportedError
from gpytorch.constraints import Interval, Positive
from gpytorch.kernels import Kernel
from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.priors import Prior
from gpytorch.priors.torch_priors import GammaPrior


class LinearTruncatedFidelityKernel(Kernel):
    r"""GPyTorch Linear Truncated Fidelity Kernel.

    Computes a covariance matrix based on the Linear truncated kernel
    between inputs `x_1` and `x_2`:

        K(x_1, x_2) = k_0 + c_1(x_1, x_1)k_1 + c_2(x_1,x_2)k_2 + c_3(x_2,x_2)k_3

    where

    - `k_i(i=0,1,2,3)` are Matern kernels calculated between `x_1[:-2]`
        and `x_2[:-2]` with different priors.
    - `c_1=(1 - x_1[-1])(1 - x_2[-1]))(1 + x_1[-1] x_2[-1])^p` is the kernel of
        thjebias term, which can be decomposed
        into a determistic part and a polynomial kernel.
    - `c_3` is the same as `c_1` but is calculated from the second last entries
        of `x_1` and `x_2`.
    - `c_2` is the interaction term with four deterministic terms and the
        polynomial kernel between `x_1[-2:]` and `x_2[-2:]`
    - `p` is the order of the polynomial kernel.

    We assume the last two dimensions of input `x` are the fidelity parameters.

    Args:
        dimension: The dimension of `x`. This is not needed if active_dims is
            specified.
        nu: The smoothness parameter fo Matern kernel: either 1/2, 3/2, or 5/2.
        batch_shape: Set this if you want a separate lengthscale for each batch
            of input data. It should be `batch_shape` if x1` is a
            `batch_shape x n x d` tensor.
        active_dims: Set this if you want to compute the covariance of only a
            few input dimensions. The numbers corresponds to the indices of the
            dimensions.
        lengthscale_prior: Set this if you want to apply a prior to the
            lengthscale parameter of Matern kernel `k_0`. Default is
            `Gamma(1.1, 1/20)`.
        lengthscale_constraint: Set this if you want to apply a constraint to
            the lengthscale parameter of the Matern kernel `k_0`. Default is
            `Positive`.
        lengthscale_2_prior: Set this if you want to apply a prior to the
            lengthscale parameter of Matern kernels `k_i(i>0)`. Default is
            `Gamma(5, 1/20)`.
        lengthscale_2_constraint: Set this if you want to apply a constraint to
            the lengthscale parameter of Matern kernels `k_i(i>0)`. Default is
            `Positive`.
        power_prior: Set this if you want to apply a prior to the power parameter
            of the polynomial kernel. Default is `None`.
        power_constraint: Set this if you want to apply a constraint to the power
            parameter of the polynomial kernel. Default is `Positive`.
        train_iteration_fidelity: Set this to True if your data contains
            iteration fidelity parameter.
        train_data_fidelity: Set this to True if your data contains training
            data fidelity parameter.
        covar_module_1: Set this if you want a different kernel for the
            unbiased part. Default is `MaternKernel`.
        covar_module_2: Set this if you want a different kernel for the biased
            part. Default is `MaternKernel`.

    Example:
        >>> x = torch.randn(10, 5)
        >>> # Non-batch: Simple option
        >>> covar_module = LinearTruncatedFidelityKernel()
        >>> covar = covar_module(x)  # Output: LazyVariable of size (10 x 10)
        >>>
        >>> batch_x = torch.randn(2, 10, 5)
        >>> # Batch: Simple option
        >>> covar_module = LinearTruncatedFidelityKernel(batch_shape = torch.Size([2]))
        >>> covar = covar_module(x)  # Output: LazyVariable of size (2 x 10 x 10)
    """

    def __init__(
        self,
        dimension: int = 3,
        nu: float = 2.5,
        train_iteration_fidelity: bool = True,
        train_data_fidelity: bool = True,
        lengthscale_prior: Optional[Prior] = None,
        power_prior: Optional[Prior] = None,
        power_constraint: Optional[Interval] = None,
        lengthscale_2_prior: Optional[Prior] = None,
        lengthscale_2_constraint: Optional[Interval] = None,
        lengthscale_constraint: Optional[Interval] = None,
        covar_module_1: Optional[Kernel] = None,
        covar_module_2: Optional[Kernel] = None,
        **kwargs: Any,
    ):
        if not train_iteration_fidelity and not train_data_fidelity:
            raise UnsupportedError("You should have at least one fidelity parameter.")
        if nu not in {0.5, 1.5, 2.5}:
            raise ValueError("nu expected to be 0.5, 1.5, or 2.5")
        super().__init__(**kwargs)
        self.train_iteration_fidelity = train_iteration_fidelity
        self.train_data_fidelity = train_data_fidelity
        if power_constraint is None:
            power_constraint = Positive()

        if lengthscale_prior is None:
            lengthscale_prior = GammaPrior(3, 6)

        if lengthscale_2_prior is None:
            lengthscale_2_prior = GammaPrior(6, 2)

        if lengthscale_constraint is None:
            lengthscale_constraint = Positive()

        if lengthscale_2_constraint is None:
            lengthscale_2_constraint = Positive()

        self.register_parameter(
            name="raw_power",
            parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1)),
        )

        if power_prior is not None:
            self.register_prior(
                "power_prior",
                power_prior,
                lambda: self.power,
                lambda v: self._set_power(v),
            )
        self.register_constraint("raw_power", power_constraint)

        m = self.train_iteration_fidelity + self.train_data_fidelity

        if self.active_dims is not None:
            dimension = len(self.active_dims)

        if covar_module_1 is None:
            self.covar_module_1 = MaternKernel(
                nu=nu,
                batch_shape=self.batch_shape,
                lengthscale_prior=lengthscale_prior,
                ard_num_dims=dimension - m,
                lengthscale_constraint=lengthscale_constraint,
            )
        else:
            self.covar_module_1 = covar_module_1

        if covar_module_2 is None:
            self.covar_module_2 = MaternKernel(
                nu=nu,
                batch_shape=self.batch_shape,
                lengthscale_prior=lengthscale_2_prior,
                ard_num_dims=dimension - m,
                lengthscale_constraint=lengthscale_2_constraint,
            )
        else:
            self.covar_module_2 = covar_module_2

    @property
    def power(self) -> torch.Tensor:
        return self.raw_power_constraint.transform(self.raw_power)

    @power.setter
    def power(self, value: torch.Tensor) -> None:
        self._set_power(value)

    def _set_power(self, value: torch.Tensor) -> None:
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_power)
        self.initialize(raw_power=self.raw_power_constraint.inverse_transform(value))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, **params) -> torch.Tensor:
        r""""""
        power = self.power.view(*self.batch_shape, 1, 1)

        m = self.train_iteration_fidelity + self.train_data_fidelity
        active_dimsM = list(range(x1.shape[-1] - m))
        x1_ = x1[..., active_dimsM]
        x2_ = x2[..., active_dimsM]
        covar_0 = self.covar_module_1(x1_, x2_)
        covar_1 = self.covar_module_2(x1_, x2_)
        x11_ = x1[..., -1].unsqueeze(-1)
        x21t_ = x2[..., -1].unsqueeze(-1).transpose(-1, -2)
        if self.train_iteration_fidelity and self.train_data_fidelity:
            covar_2 = self.covar_module_2(x1_, x2_)
            covar_3 = self.covar_module_2(x1_, x2_)
            x12_ = x1[..., -2].unsqueeze(-1)
            x22t_ = x2[..., -2].unsqueeze(-1).transpose(-1, -2)
            res = (
                covar_0
                + (1 - x12_) * (1 - x22t_) * (1 + x12_ * x22t_).pow(power) * covar_1
                + (1 - x12_)
                * (1 - x22t_)
                * (1 - x11_)
                * (1 - x21t_)
                * (1 + torch.matmul(x1[..., -2:], x2[..., -2:].transpose(-1, -2))).pow(
                    power
                )
                * covar_2
                + (1 - x11_) * (1 - x21t_) * (1 + x11_ * x21t_).pow(power) * covar_3
            )
        else:
            res = (
                covar_0
                + (1 - x11_) * (1 - x21t_) * (1 + x11_ * x21t_).pow(power) * covar_1
            )
        return res

    def __getitem__(self, index):
        new_kernel = deepcopy(self)
        new_kernel.covar_module_1 = self.covar_module_1[index]
        new_kernel.covar_module_2 = self.covar_module_2[index]
        return new_kernel
