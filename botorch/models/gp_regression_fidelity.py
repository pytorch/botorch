#! /usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Gaussian Process Regression models based on GPyTorch models.

.. [Wu2019mf]
    J. Wu, S. Toscano-Palmerin, P. I. Frazier, and A. G. Wilson. Practical
    multi-fidelity bayesian optimization for hyperparameter tuning. ArXiv 2019.
"""

from typing import Optional

from gpytorch.kernels.kernel import ProductKernel
from gpytorch.kernels.rbf_kernel import RBFKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.priors.torch_priors import GammaPrior
from torch import Tensor

from ..exceptions.errors import UnsupportedError
from .gp_regression import SingleTaskGP
from .kernels.downsampling import DownsamplingKernel
from .kernels.exponential_decay import ExponentialDecayKernel
from .kernels.linear_truncated_fidelity import LinearTruncatedFidelityKernel
from .transforms.outcome import OutcomeTransform


class SingleTaskMultiFidelityGP(SingleTaskGP):
    r"""A single task multi-fidelity GP model.

    A SingleTaskGP model using a DownsamplingKernel for the data fidelity
    parameter (if present) and an ExponentialDecayKernel for the iteration
    fidelity parameter (if present).

    This kernel is described in [Wu2019mf]_.

    Args:
        train_X: A `batch_shape x n x (d + s)` tensor of training features,
            where `s` is the dimension of the fidelity parameters (either one
            or two).
        train_Y: A `batch_shape x n x m` tensor of training observations.
        iteration_fidelity: The column index for the training iteration fidelity
            parameter (optional).
        data_fidelity: The column index for the downsampling fidelity parameter
            (optional).
        linear_truncated: If True, use a `LinearTruncatedFidelityKernel` instead
            of the default kernel.
        nu: The smoothness parameter for the Matern kernel: either 1/2, 3/2, or
            5/2. Only used when `linear_truncated=True`.
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
        iteration_fidelity: Optional[int] = None,
        data_fidelity: Optional[int] = None,
        linear_truncated: bool = True,
        nu: float = 2.5,
        likelihood: Optional[Likelihood] = None,
        outcome_transform: Optional[OutcomeTransform] = None,
    ) -> None:
        if iteration_fidelity is None and data_fidelity is None:
            raise UnsupportedError(
                "SingleTaskMultiFidelityGP requires at least one fidelity parameter."
            )
        if iteration_fidelity is not None and iteration_fidelity < 0:
            iteration_fidelity = train_X.size(-1) + iteration_fidelity
        if data_fidelity is not None and data_fidelity < 0:
            data_fidelity = train_X.size(-1) + data_fidelity
        self._set_dimensions(train_X=train_X, train_Y=train_Y)
        if linear_truncated:
            fidelity_dims = [
                i for i in (iteration_fidelity, data_fidelity) if i is not None
            ]
            kernel = LinearTruncatedFidelityKernel(
                fidelity_dims=fidelity_dims,
                dimension=train_X.size(-1),
                nu=nu,
                batch_shape=self._aug_batch_shape,
                power_prior=GammaPrior(3.0, 3.0),
            )
        else:
            active_dimsX = [
                i
                for i in range(train_X.size(-1))
                if i not in {iteration_fidelity, data_fidelity}
            ]
            kernel = RBFKernel(
                ard_num_dims=len(active_dimsX),
                batch_shape=self._aug_batch_shape,
                lengthscale_prior=GammaPrior(3.0, 6.0),
                active_dims=active_dimsX,
            )
            additional_kernels = []
            if iteration_fidelity is not None:
                exp_kernel = ExponentialDecayKernel(
                    batch_shape=self._aug_batch_shape,
                    lengthscale_prior=GammaPrior(3.0, 6.0),
                    offset_prior=GammaPrior(3.0, 6.0),
                    power_prior=GammaPrior(3.0, 6.0),
                    active_dims=[iteration_fidelity],
                )
                additional_kernels.append(exp_kernel)
            if data_fidelity is not None:
                ds_kernel = DownsamplingKernel(
                    batch_shape=self._aug_batch_shape,
                    offset_prior=GammaPrior(3.0, 6.0),
                    power_prior=GammaPrior(3.0, 6.0),
                    active_dims=[data_fidelity],
                )
                additional_kernels.append(ds_kernel)
            kernel = ProductKernel(kernel, *additional_kernels)

        covar_module = ScaleKernel(
            kernel,
            batch_shape=self._aug_batch_shape,
            outputscale_prior=GammaPrior(2.0, 0.15),
        )
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            covar_module=covar_module,
            outcome_transform=outcome_transform,
        )
        self.to(train_X)
