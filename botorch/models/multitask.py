#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Multi-Task GP models.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from botorch.models.gpytorch import MultiTaskGPyTorchModel
from botorch.models.utils import validate_input_scaling
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.kernels.index_kernel import IndexKernel
from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.likelihoods.gaussian_likelihood import (
    FixedNoiseGaussianLikelihood,
    GaussianLikelihood,
)
from gpytorch.means.constant_mean import ConstantMean
from gpytorch.models.exact_gp import ExactGP
from gpytorch.priors.torch_priors import GammaPrior
from torch import Tensor


class MultiTaskGP(ExactGP, MultiTaskGPyTorchModel):
    r"""Multi-Task GP model using an ICM kernel, inferring observation noise.

    Multi-task exact GP that uses a simple ICM kernel. Can be single-output or
    multi-output. This model uses relatively strong priors on the base Kernel
    hyperparameters, which work best when covariates are normalized to the unit
    cube and outcomes are standardized (zero mean, unit variance).

    This model infers the noise level. WARNING: It currently does not support
    different noise levels for the different tasks. If you have known observation
    noise, please use `FixedNoiseMultiTaskGP` instead.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        task_feature: int,
        output_tasks: Optional[List[int]] = None,
        rank: Optional[int] = None,
    ) -> None:
        r"""Multi-Task GP model using an ICM kernel, inferring observation noise.

        Args:
            train_X: A `n x (d + 1)` or `b x n x (d + 1)` (batch mode) tensor
                of training data. One of the columns should contain the task
                features (see `task_feature` argument).
            train_Y: A `n` or `b x n` (batch mode) tensor of training
                observations.
            task_feature: The index of the task feature
                (`-d <= task_feature <= d`).
            output_tasks: A list of task indices for which to compute model
                outputs for. If omitted, return outputs for all task indices.
            rank: The rank to be used for the index kernel. If omitted, use a
                full rank (i.e. number of tasks) kernel.

        Example:
            >>> X1, X2 = torch.rand(10, 2), torch.rand(20, 2)
            >>> i1, i2 = torch.zeros(10, 1), torch.ones(20, 1)
            >>> train_X = torch.stack([
            >>>     torch.cat([X1, i1], -1), torch.cat([X2, i2], -1),
            >>> ])
            >>> train_Y = torch.cat(f1(X1), f2(X2))
            >>> model = MultiTaskGP(train_X, train_Y, task_feature=-1)
        """
        self._validate_tensor_args(X=train_X, Y=train_Y)
        validate_input_scaling(train_X=train_X, train_Y=train_Y)
        if train_X.ndim != 2:
            # Currently, batch mode MTGPs are blocked upstream in GPyTorch
            raise ValueError(f"Unsupported shape {train_X.shape} for train_X.")
        # squeeze output dim
        train_Y = train_Y.squeeze(-1)
        d = train_X.shape[-1] - 1
        if not (-d <= task_feature <= d):
            raise ValueError(f"Must have that -{d} <= task_feature <= {d}")
        all_tasks = train_X[:, task_feature].unique().to(dtype=torch.long).tolist()
        if output_tasks is None:
            output_tasks = all_tasks
        else:
            if any(t not in all_tasks for t in output_tasks):
                raise RuntimeError("All output tasks must be present in input data.")
        self._output_tasks = output_tasks
        self._num_outputs = len(output_tasks)

        # TODO (T41270962): Support task-specific noise levels in likelihood
        likelihood = GaussianLikelihood(noise_prior=GammaPrior(1.1, 0.05))

        # construct indexer to be used in forward
        self._task_feature = task_feature
        self._base_idxr = torch.arange(d)
        self._base_idxr[task_feature:] += 1  # exclude task feature

        super().__init__(
            train_inputs=train_X, train_targets=train_Y, likelihood=likelihood
        )
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(
            base_kernel=MaternKernel(
                nu=2.5, ard_num_dims=d, lengthscale_prior=GammaPrior(3.0, 6.0)
            ),
            outputscale_prior=GammaPrior(2.0, 0.15),
        )
        num_tasks = len(all_tasks)
        self._rank = rank if rank is not None else num_tasks
        # TODO: Add LKJ prior for the index kernel
        self.task_covar_module = IndexKernel(num_tasks=num_tasks, rank=self._rank)
        self.to(train_X)

    def _split_inputs(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        r"""Extracts base features and task indices from input data.

        Args:
            x: The full input tensor with trailing dimension of size `d + 1`.
                Should be of float/double data type.

        Returns:
            2-element tuple containin

            - A `q x d` or `b x q x d` (batch mode) tensor with trailing
            dimension made up of the `d` non-task-index columns of `x`, arranged
            in the order as specified by the indexer generated during model
            instantiation.
            - A `q` or `b x q` (batch mode) tensor of long data type containing
            the task indices.
        """
        batch_shape, d = x.shape[:-2], x.shape[-1]
        x_basic = x[..., self._base_idxr].view(batch_shape + torch.Size([-1, d - 1]))
        task_idcs = (
            x[..., self._task_feature]
            .view(batch_shape + torch.Size([-1, 1]))
            .to(dtype=torch.long)
        )
        return x_basic, task_idcs

    def forward(self, x: Tensor) -> MultivariateNormal:
        x_basic, task_idcs = self._split_inputs(x)
        # Compute base mean and covariance
        mean_x = self.mean_module(x_basic)
        covar_x = self.covar_module(x_basic)
        # Compute task covariances
        covar_i = self.task_covar_module(task_idcs)
        # Combine the two in an ICM fashion
        covar = covar_x.mul(covar_i)
        return MultivariateNormal(mean_x, covar)


class FixedNoiseMultiTaskGP(MultiTaskGP):
    r"""Multi-Task GP model using an ICM kernel, with known observation noise.

    Multi-task exact GP that uses a simple ICM kernel. Can be single-output or
    multi-output. This model uses relatively strong priors on the base Kernel
    hyperparameters, which work best when covariates are normalized to the unit
    cube and outcomes are standardized (zero mean, unit variance).

    This model requires observation noise data (specified in `train_Yvar`).
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor,
        task_feature: int,
        output_tasks: Optional[List[int]] = None,
        rank: Optional[int] = None,
    ) -> None:
        r"""Multi-Task GP model using an ICM kernel and known observatioon noise.

        Args:
            train_X: A `n x (d + 1)` or `b x n x (d + 1)` (batch mode) tensor
                of training data. One of the columns should contain the task
                features (see `task_feature` argument).
            train_Y: A `n` or `b x n` (batch mode) tensor of training
                observations.
            train_Yvar: A `n` or `b x n` (batch mode) tensor of observation
                noise standard errors.
            task_feature: The index of the task feature
                (`-d <= task_feature <= d`).
            output_tasks: A list of task indices for which to compute model
                outputs for. If omitted, return outputs for all task indices.
            rank: The rank to be used for the index kernel. If omitted, use a
                full rank (i.e. number of tasks) kernel.

        Example:
            >>> X1, X2 = torch.rand(10, 2), torch.rand(20, 2)
            >>> i1, i2 = torch.zeros(10, 1), torch.ones(20, 1)
            >>> train_X = torch.cat([
            >>>     torch.cat([X1, i1], -1), torch.cat([X2, i2], -1),
            >>> ], dim=0)
            >>> train_Y = torch.cat(f1(X1), f2(X2))
            >>> train_Yvar = 0.1 + 0.1 * torch.rand_like(train_Y)
            >>> model = FixedNoiseMultiTaskGP(train_X, train_Y, train_Yvar, -1)
        """
        self._validate_tensor_args(X=train_X, Y=train_Y, Yvar=train_Yvar)
        # We'll instatiate a MultiTaskGP and simply override the likelihood
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            task_feature=task_feature,
            output_tasks=output_tasks,
            rank=rank,
        )
        self.likelihood = FixedNoiseGaussianLikelihood(noise=train_Yvar.squeeze(-1))
        self.to(train_X)
