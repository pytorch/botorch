#! /usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Gaussian Process Regression models based on GPyTorch models.
"""

from typing import Optional

import torch
from botorch.exceptions import UnsupportedError
from botorch.models.fidelity_kernels.downsampling import DownsamplingKernel
from botorch.models.fidelity_kernels.exponential_decay import ExponentialDecayKernel
from botorch.models.fidelity_kernels.linear_truncated_fidelity import (
    LinearTruncatedFidelityKernel,
)
from gpytorch.kernels.rbf_kernel import RBFKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.priors.torch_priors import GammaPrior
from torch import Tensor

from ..gp_regression import SingleTaskGP


class SingleTaskMultiFidelityGP(SingleTaskGP):
    r"""A single task multi-fidelity GP model.

    A sub-class of SingleTaskGP model. By default the last two input dimensions
    are the fidelity parameters: training iterations, training data points.
    The kernel comes from this paper https://arxiv.org/abs/1903.04703

    Args:
        train_X: A `batch_shape x n x (d + s) ` tensor of training features,
            where `s` is the dimension of the fidelity parameters.
        train_Y: A `batch_shape x n x m` tensor of training observations.
        train_iteration_fidelity: An indicator of whether we have the training
            iteration fidelity variable.
        train_data_fidelity: An indicator of whether we have the downsampling
            fidelity variable. If train_iteration_fidelity and train_data_fidelity
            are both True, the last and second last columns are treated as the
            training data points fidelity parameter and training iteration
            number fidelity parameter respectively. Otherwise the last column of
            `train_X` is treated as the fidelity parameter with True indicator.
            We assume that `train_X` has at least one fidelity parameter.
        likelihood: A likelihood. If omitted, use a standard GaussianLikelihood
            with inferred noise level.

    Example:
        >>> train_X = torch.rand(20, 4)
        >>> train_Y = train_X.pow(2).sum(dim=-1, keepdim=True)
        >>> model = SingleTaskMultiFidelityGP(train_X, train_Y)
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_iteration_fidelity: bool = True,
        train_data_fidelity: bool = True,
        likelihood: Optional[Likelihood] = None,
    ) -> None:
        self._set_dimensions(train_X=train_X, train_Y=train_Y)
        num_fidelity = train_iteration_fidelity + train_data_fidelity
        ard_num_dims = train_X.shape[-1] - num_fidelity
        active_dimsX = list(range(train_X.shape[-1] - num_fidelity))
        rbf_kernel = RBFKernel(
            ard_num_dims=ard_num_dims,
            batch_shape=self._aug_batch_shape,
            lengthscale_prior=GammaPrior(3.0, 6.0),
            active_dims=active_dimsX,
        )
        exp_kernel = ExponentialDecayKernel(
            batch_shape=self._aug_batch_shape,
            lengthscale_prior=GammaPrior(3.0, 6.0),
            offset_prior=GammaPrior(3.0, 6.0),
            power_prior=GammaPrior(3.0, 6.0),
        )
        ds_kernel = DownsamplingKernel(
            batch_shape=self._aug_batch_shape,
            offset_prior=GammaPrior(3.0, 6.0),
            power_prior=GammaPrior(3.0, 6.0),
        )
        if train_iteration_fidelity and train_data_fidelity:
            active_dimsS1 = [train_X.shape[-1] - 1]
            active_dimsS2 = [train_X.shape[-1] - 2]
            exp_kernel.active_dims = torch.tensor(active_dimsS1)
            ds_kernel.active_dims = torch.tensor(active_dimsS2)
            kernel = rbf_kernel * exp_kernel * ds_kernel
        elif train_iteration_fidelity or train_data_fidelity:
            active_dimsS = [train_X.shape[-1] - 1]
            if train_iteration_fidelity:
                exp_kernel.active_dims = torch.tensor(active_dimsS)
                kernel = rbf_kernel * exp_kernel
            else:
                ds_kernel.active_dims = torch.tensor(active_dimsS)
                kernel = rbf_kernel * ds_kernel
        else:
            raise UnsupportedError("You should have at least one fidelity parameter.")
        covar_module = ScaleKernel(
            kernel,
            batch_shape=self._aug_batch_shape,
            outputscale_prior=GammaPrior(2.0, 0.15),
        )
        super().__init__(train_X=train_X, train_Y=train_Y, covar_module=covar_module)
        self.to(train_X)


class SingleTaskGPLTKernel(SingleTaskGP):
    r"""A single task multi-fidelity GP model wiht Linear Truncated kernel.

    A sub-class of SingleTaskGP model. By default the last two input dimensions
    are the fidelity parameters: training iterations, training data points.

    Args:
        train_X: A `batch_shape x n x (d + s) ` tensor of training features,
            where `s` is the dimension of the fidelity parameters.
        train_Y: A `batch_shape x n x m` tensor of training observations.
        dimension: The dimension of `x`.
        nu: The smoothness parameter fo Matern kernel: either 1/2, 3/2, or 5/2.
            Default: '2.5'
        train_iteration_fidelity: An indicator of whether we have the training
            iteration fidelity variable.
        train_data_fidelity: An indicator of whether we have the downsampling
            fidelity variable. If train_iteration_fidelity and train_data_fidelity
            are both True, the last and second last columns are treated as the
            training data points fidelity parameter and training iteration
            number fidelity parameter respectively. Otherwise the last column of
            train_X is treated as the fidelity parameter with True indicator.
            We assume that `train_X` has at least one fidelity parameter.
        likelihood: A likelihood. If omitted, use a standard
            GaussianLikelihood with inferred noise level.

    Example:
        >>> train_X = torch.rand(20, 4)
        >>> train_Y = train_X.pow(2).sum(dim=-1, keepdim=True)
        >>> model = SingleTaskGPLTKernel(train_X, train_Y)
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        nu: float = 2.5,
        train_iteration_fidelity: bool = True,
        train_data_fidelity: bool = True,
        likelihood: Optional[Likelihood] = None,
    ) -> None:
        if not train_iteration_fidelity and not train_data_fidelity:
            raise UnsupportedError("You should have at least one fidelity parameter.")
        self._set_dimensions(train_X=train_X, train_Y=train_Y)
        kernel = LinearTruncatedFidelityKernel(
            nu=nu,
            dimension=train_X.shape[-1],
            train_iteration_fidelity=train_iteration_fidelity,
            train_data_fidelity=train_data_fidelity,
            batch_shape=self._aug_batch_shape,
            power_prior=GammaPrior(3.0, 3.0),
        )
        covar_module = ScaleKernel(
            kernel,
            batch_shape=self._aug_batch_shape,
            outputscale_prior=GammaPrior(2.0, 0.15),
        )
        super().__init__(train_X=train_X, train_Y=train_Y, covar_module=covar_module)
        self.to(train_X)
