#! /usr/bin/env python3

r"""
Gaussian Process Regression models based on GPyTorch models.
"""

from copy import deepcopy
from typing import Optional

import torch
from gpytorch.constraints.constraints import GreaterThan
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.likelihoods.gaussian_likelihood import (
    FixedNoiseGaussianLikelihood,
    GaussianLikelihood,
    _GaussianLikelihoodBase,
)
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.likelihoods.noise_models import HeteroskedasticNoise
from gpytorch.means.constant_mean import ConstantMean
from gpytorch.models.exact_gp import ExactGP
from gpytorch.priors.smoothed_box_prior import SmoothedBoxPrior
from gpytorch.priors.torch_priors import GammaPrior
from torch import Tensor

from .gpytorch import BatchedMultiOutputGPyTorchModel
from .utils import multioutput_to_batch_mode_transform


MIN_INFERRED_NOISE_LEVEL = 1e-6


class SingleTaskGP(ExactGP, BatchedMultiOutputGPyTorchModel):
    r"""A single-task exact GP model.

    A single-task exact GP using relatively strong priors on the Kernel
    hyperparameters, which work best when covariates are normalized to the unit
    cube and outcomes are standardized (zero mean, unit variance).

    This model works in batch mode (each batch having its own hyperparameters).
    When the training observations include multiple outputs, this model will use
    batching to model outputs independently.

    Use this model when you have independent output(s) and all outputs use the same
    training data. If outputs are independent and outputs have different training
    data, use the ModelListGP. When modeling correlations between outputs, use
    the MultiTaskGP.
    """

    def __init__(
        self, train_X: Tensor, train_Y: Tensor, likelihood: Optional[Likelihood] = None
    ) -> None:
        r"""A single-task exact GP model.

        Args:
            train_X: A `n x d` or `batch_shape x n x d` (batch mode) tensor of training
                features.
            train_Y: A `n x (o)` or `batch_shape x n x (o)` (batch mode) tensor of
                training observations.
            likelihood: A likelihood. If omitted, use a standard
                GaussianLikelihood with inferred noise level.

        Example:
            >>> train_X = torch.rand(20, 2)
            >>> train_Y = torch.sin(train_X[:, 0]]) + torch.cos(train_X[:, 1])
            >>> model = SingleTaskGP(train_X, train_Y)
        """
        ard_num_dims = train_X.shape[-1]
        self._set_dimensions(train_X=train_X, train_Y=train_Y)
        train_X, train_Y, _ = multioutput_to_batch_mode_transform(
            train_X=train_X, train_Y=train_Y, num_outputs=self._num_outputs
        )
        if likelihood is None:
            noise_prior = GammaPrior(1.1, 0.05)
            noise_prior_mode = (noise_prior.concentration - 1) / noise_prior.rate
            likelihood = GaussianLikelihood(
                noise_prior=noise_prior,
                batch_shape=self._aug_batch_shape,
                noise_constraint=GreaterThan(
                    MIN_INFERRED_NOISE_LEVEL,
                    transform=None,
                    initial_value=noise_prior_mode,
                ),
            )
        else:
            self._likelihood_state_dict = deepcopy(likelihood.state_dict())
        super().__init__(train_X, train_Y, likelihood)
        self.mean_module = ConstantMean(batch_shape=self._aug_batch_shape)
        self.covar_module = ScaleKernel(
            MaternKernel(
                nu=2.5,
                ard_num_dims=ard_num_dims,
                batch_shape=self._aug_batch_shape,
                lengthscale_prior=GammaPrior(3.0, 6.0),
            ),
            batch_shape=self._aug_batch_shape,
            outputscale_prior=GammaPrior(2.0, 0.15),
        )
        self.to(train_X)

    def forward(self, x: Tensor) -> MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class FixedNoiseGP(ExactGP, BatchedMultiOutputGPyTorchModel):
    r"""A single-task exact GP model using fixed noise levels.

    A single-task exact GP that uses fixed observation noise levels. This model
    also uses relatively strong priors on the Kernel hyperparameters, which work
    best when covariates are normalized to the unit cube and outcomes are
    standardized (zero mean, unit variance).

    This model works in batch mode (each batch having its own hyperparameters).
    """

    def __init__(self, train_X: Tensor, train_Y: Tensor, train_Yvar: Tensor) -> None:
        r"""A single-task exact GP model using fixed noise levels.

        Args:
            train_X: A `n x d` or `batch_shape x n x d` (batch mode) tensor of training
                features.
            train_Y: A `n x (o)` or `batch_shape x n x (o)` (batch mode) tensor of
                training observations.
            train_Yvar: A `batch_shape x n x (t)` or `batch_shape x n x (t)`
                (batch mode) tensor of observed measurement noise.

        Example:
            >>> train_X = torch.rand(20, 2)
            >>> train_Y = torch.sin(train_X[:, 0]]) + torch.cos(train_X[:, 1])
            >>> train_Yvar = torch.full_like(train_Y, 0.2)
            >>> model = FixedNoiseGP(train_X, train_Y, train_Yvar)
        """
        ard_num_dims = train_X.shape[-1]
        self._set_dimensions(train_X=train_X, train_Y=train_Y)
        train_X, train_Y, train_Yvar = multioutput_to_batch_mode_transform(
            train_X=train_X,
            train_Y=train_Y,
            num_outputs=self._num_outputs,
            train_Yvar=train_Yvar,
        )
        likelihood = FixedNoiseGaussianLikelihood(
            noise=train_Yvar, batch_shape=self._aug_batch_shape
        )
        super().__init__(
            train_inputs=train_X, train_targets=train_Y, likelihood=likelihood
        )
        self.mean_module = ConstantMean(batch_shape=self._aug_batch_shape)
        self.covar_module = ScaleKernel(
            base_kernel=MaternKernel(
                nu=2.5,
                ard_num_dims=ard_num_dims,
                batch_shape=self._aug_batch_shape,
                lengthscale_prior=GammaPrior(3.0, 6.0),
            ),
            batch_shape=self._aug_batch_shape,
            outputscale_prior=GammaPrior(2.0, 0.15),
        )
        self.to(train_X)

    def forward(self, x: Tensor) -> MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class HeteroskedasticSingleTaskGP(SingleTaskGP):
    r"""A single-task exact GP model using a heteroskeastic noise model.

    This model internally wraps another GP (a SingleTaskGP) to model the observation
    noise. This allows the likelihood to make out-of-sample predictions for the
    observation noise levels.
    """

    def __init__(self, train_X: Tensor, train_Y: Tensor, train_Yvar: Tensor) -> None:
        r"""A single-task exact GP model using a heteroskedastic noise model.

        Args:
            train_X: A `n x d` or `batch_shape x n x d` (batch mode) tensor of training
                features.
            train_Y: A `n x (o)` or `batch_shape x n x (o)` (batch mode) tensor of
                training observations.
            train_Yvar: A `batch_shape x n x (o)` or `batch_shape x n x (o)`
                (batch mode) tensor of observed measurement noise..

        Example:
            >>> train_X = torch.rand(20, 2)
            >>> train_Y = torch.sin(train_X[:, 0]]) + torch.cos(train_X[:, 1])
            >>> se = torch.norm(train_X - 0.5, dim=-1)
            >>> train_Yvar = 0.1 + se * torch.rand_like(train_Y)
            >>> model = HeteroskedasticSingleTaskGP(train_X, train_Y, train_Yvar)
        """
        self._set_dimensions(train_X=train_X, train_Y=train_Y)
        train_Y_log_var = torch.log(train_Yvar)
        noise_likelihood = GaussianLikelihood(
            noise_prior=SmoothedBoxPrior(-3, 5, 0.5, transform=torch.log),
            batch_shape=self._aug_batch_shape,
            noise_constraint=GreaterThan(MIN_INFERRED_NOISE_LEVEL, transform=None),
        )
        noise_model = SingleTaskGP(
            train_X=train_X, train_Y=train_Y_log_var, likelihood=noise_likelihood
        )

        likelihood = _GaussianLikelihoodBase(HeteroskedasticNoise(noise_model))
        super().__init__(train_X=train_X, train_Y=train_Y, likelihood=likelihood)
        self.to(train_X)
