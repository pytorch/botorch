#! /usr/bin/env python3

r"""
Multi-Task GP models.
"""

from typing import List, Optional, Tuple

import torch
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

from .gpytorch import MultiTaskGPyTorchModel


class MultiTaskGP(ExactGP, MultiTaskGPyTorchModel):
    r"""Multi-Task GP model using an ICM kernel, inferring observation noise.

    Multi-task exact GP that uses a simple ICM kernel. Can be single-output or
    multi-output. This model uses relatively strong priors on the base Kernel
    hyperparameters, which work best when covariates are normalized to the unit
    cube and outcomes are standardized (zero mean, unit variance).

    WARNING: This model currently does not support inferring different noise
    levels for the different tasks (work ongoing). If you have known observation
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
                (`-(d+1) <= task_feature <= d+1`).
            output_tasks: A list of task indices for which to compute model
                outputs for. If omitted, return outputs for all task indices.
            rank: The rank to be used for the index kernel. If omitted, use a
                full rank (i.e. number of tasks) kernel.
        """
        if train_X.ndimension() != 2:
            # Currently, batch mode MTGPs are blocked upstream in GPyTorch
            raise ValueError(f"Unsupported shape {train_X.shape} for train_X.")
        d = train_X.shape[-1] - 1
        if not (-(d + 1) <= task_feature <= d + 1):
            raise ValueError(f"Must have that -({d+1}) <= task_feature <= {d+1}")
        all_tasks = train_X[:, task_feature].unique().to(dtype=torch.long).tolist()
        if output_tasks is None:
            output_tasks = all_tasks
        else:
            if any(t not in all_tasks for t in output_tasks):
                raise RuntimeError("All output tasks must be present in input data.")
        self._output_tasks = output_tasks

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

    @property
    def num_outputs(self) -> int:
        return len(self._output_tasks)

    def _split_inputs(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        r"""Extracts base features and task indices from input data.

        Args:
            x: The full input tensor with trailing dimension of size `d + 1`.
                Should be of float/double data type.

        Returns:
            Tensor: A `q x d` or `b x q x d` (batch mode) tensor with trailing
                dimension made up of the `d` non-task-index columns of `x`,
                arranged in the order as specified by the indexer generated
                during model instantiation.
            Tensor: A `q` or `b x q` (batch mode) tensor of long data type
                containing the task indices.
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

    def reinitialize(
        self, train_X: Tensor, train_Y: Tensor, keep_params: bool = True
    ) -> None:
        r"""Reinitialize model and the likelihood given new data.

        Args:
            train_X: A tensor of new training data
            train_Y: A tensor of new training observations
            keep_params: If True, keep the model's hyperarameter values (speeds
                up refitting on similar data)

        This does not refit the model.
        If device/dtype of the new training data are different from that of the
        model, then the model is moved to the new device/dtype.
        """
        if keep_params:
            self.set_train_data(inputs=train_X, targets=train_Y, strict=False)
        else:
            self.__init__(
                train_X=train_X,
                train_Y=train_Y,
                task_feature=self._task_feature,
                output_tasks=self._output_tasks,
                rank=self._rank,
            )
        # move to new device / dtype if necessary
        self.to(train_X)


class FixedNoiseMultiTaskGP(MultiTaskGP):
    r"""Basic Multi-Task GP model using an ICM kernel.

    Multi-task exact GP using a simple ICM kernel. This
    model uses relatively strong priors on the base Kernel hyperparameters, which
    work best when covariates are normalized to the unit cube and outcomes are
    standardized (zero mean, unit variance).
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
        r"""Multi-Task GP model using an ICM kernel and known ovservatioon noise.

        Args:
            train_X: A `n x (d + 1)` or `b x n x (d + 1)` (batch mode) tensor
                of training data. One of the columns should contain the task
                features (see `task_feature` argument).
            train_Y: A `n` or `b x n` (batch mode) tensor of training
                observations.
            train_Yvar: A `n` or `b x n` (batch mode) tensor of observation
                noise standard errors.
            task_feature: The index of the task feature
                (`-(d+1) <= task_feature <= d+1`).
            output_tasks: A list of task indices for which to compute model
                outputs for. If omitted, return outputs for all task indices.
            rank: The rank to be used for the index kernel. If omitted, use a
                full rank (i.e. number of tasks) kernel.
        """
        if train_X.ndimension() != 2:
            # Currently, batch mode MTGPs are blocked upstream in GPyTorch
            raise ValueError(f"Unsupported shape {train_X.shape} for train_X.")
        d = train_X.shape[-1] - 1
        if not (-(d + 1) <= task_feature <= d + 1):
            raise ValueError(f"Must have that -({d+1}) <= task_feature <= {d+1}")
        all_tasks = train_X[:, task_feature].unique().tolist()
        if output_tasks is None:
            output_tasks = all_tasks
        self._output_tasks = output_tasks

        likelihood = FixedNoiseGaussianLikelihood(noise=train_Yvar)

        # construct indexer to be used in forward
        self._task_feature = task_feature
        self._base_idxr = torch.arange(d)
        self._base_idxr[task_feature:] += 1

        super(MultiTaskGP, self).__init__(
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

    def reinitialize(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor,
        keep_params: bool = True,
    ) -> None:
        r"""Reinitialize model and the likelihood given new data.

        Args:
            train_X: A tensor of new training data
            train_Y: A tensor of new training observations
            train_Yvar: A tensor of new observation noises
            keep_params: If True, keep the model's hyperparameter values (speeds
                up refitting on similar data)

        This does not refit the model.
        If device/dtype of the new training data are different from that of the
        model, then the model is moved to the new device/dtype.
        """
        if keep_params:
            self.set_train_data(inputs=train_X, targets=train_Y, strict=False)
            self.likelihood.noise = train_Yvar
        else:
            self.__init__(
                train_X=train_X,
                train_Y=train_Y,
                train_Yvar=train_Yvar,
                task_feature=self._task_feature,
                output_tasks=self._output_tasks,
                rank=self._rank,
            )
        # move to new device / dtype if necessary
        self.to(train_X)
