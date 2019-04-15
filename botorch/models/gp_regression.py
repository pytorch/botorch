#! /usr/bin/env python3

r"""
Basic GP Regression models based on GPyTorch GP models.
"""

from copy import deepcopy
from typing import Any, Optional

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

from .gpytorch import GPyTorchModel


MIN_INFERRED_NOISE_LEVEL = 1e-6


class SingleTaskGP(ExactGP, GPyTorchModel):
    r"""A single-task Exact GP model.

    Class implementing a single task exact GP using relatively strong priors on
    the Kernel hyperparameters, which work best when covariates are normalized
    to the unit cube and outcomes are standardized (zero mean, unit variance).

    This model works in batch mode (each batch having its own hyperparameters).
    """

    def __init__(
        self, train_X: Tensor, train_Y: Tensor, likelihood: Optional[Likelihood] = None
    ) -> None:
        r"""A single-task Exact GP model.

        Args:
            train_X: A `n x d` or `b x n x d` (batch mode) tensor of training
                features.
            train_Y: A `n` or `b x n` (batch mode) tensor of training
                observations.
            likelihood: A likelihood. If omitted, use a standard
                GaussianLikelihood with inferred noise level.
        """
        batch_shape = train_X.shape[:-2]
        ard_num_dims = train_X.shape[-1] if train_X.dim() > 1 else None
        if likelihood is None:
            likelihood = GaussianLikelihood(
                noise_prior=GammaPrior(1.1, 0.05),
                batch_shape=batch_shape,
                noise_constraint=GreaterThan(MIN_INFERRED_NOISE_LEVEL, transform=None),
            )
        else:
            self._likelihood_state_dict = deepcopy(likelihood.state_dict())
        super().__init__(train_X, train_Y, likelihood)
        self.mean_module = ConstantMean(batch_shape=batch_shape)
        self.covar_module = ScaleKernel(
            MaternKernel(
                nu=2.5,
                ard_num_dims=ard_num_dims,
                batch_shape=batch_shape,
                lengthscale_prior=GammaPrior(3.0, 6.0),
            ),
            batch_shape=batch_shape,
            outputscale_prior=GammaPrior(2.0, 0.15),
        )
        self.to(train_X)

    def forward(self, x: Tensor) -> MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def reinitialize(
        self, train_X: Tensor, train_Y: Tensor, keep_params: bool = True, **kwargs: Any
    ) -> None:
        r"""Reinitialize model and the likelihood given new training data.

        This does not refit the model.
        If device/dtype of the new training data are different from that of the
        model, then the model is moved to the new device/dtype.

        Args:
            train_X: A tensor of new training features.
            train_Y: A tensor of new training observations.
            keep_params: If True, keep the parameter values (speeds up refitting
                on similar data).
        """
        if keep_params:
            self.set_train_data(inputs=train_X, targets=train_Y, strict=False)
        elif hasattr(self, "_likelihood_state_dict"):
            self.likelihood.load_state_dict(self._likelihood_state_dict)
            self.__init__(train_X=train_X, train_Y=train_Y, likelihood=self.likelihood)
            self.to(train_X)
        else:
            self.__init__(train_X=train_X, train_Y=train_Y)


class FixedNoiseGP(ExactGP, GPyTorchModel):
    """A model using fixed noise levels."""

    def __init__(self, train_X: Tensor, train_Y: Tensor, train_Yvar: Tensor) -> None:
        r"""A model using fixed noise levels.

        Args:
            train_X: A `n x d` or `b x n x d` (batch mode) tensor of training
                inputs.
            train_Y: A `n` or `b x n` (batch mode) tensor of training
                observations.
            train_Yvar: A `n` or `b x n` (batch mode) tensor of observed
                measurement noise.
        """
        batch_shape = train_X.shape[:-2]
        ard_num_dims = train_X.shape[-1] if train_X.dim() > 1 else None
        likelihood = FixedNoiseGaussianLikelihood(noise=train_Yvar)
        super().__init__(
            train_inputs=train_X, train_targets=train_Y, likelihood=likelihood
        )
        self.mean_module = ConstantMean(batch_shape=batch_shape)
        self.covar_module = ScaleKernel(
            base_kernel=MaternKernel(
                nu=2.5,
                ard_num_dims=ard_num_dims,
                batch_shape=batch_shape,
                lengthscale_prior=GammaPrior(3.0, 6.0),
            ),
            batch_shape=batch_shape,
            outputscale_prior=GammaPrior(2.0, 0.15),
        )
        self.to(train_X)

    def forward(self, x: Tensor) -> MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

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
            train_Yvar: A tensor of new training noise observations
            keep_params: If True, keep the model's hyperparameter values (speeds
                up refitting on similar data)

        This does not refit the model.
        If device/dtype of the new training data are different from that of the
        model, then the model is moved to the new device/dtype.
        """
        if keep_params:
            self.set_train_data(inputs=train_X, targets=train_Y, strict=False)
            self.likelihood.noise_covar.register_buffer("noise", train_Yvar)
        else:
            self.__init__(train_X=train_X, train_Y=train_Y, train_Yvar=train_Yvar)
        # move to new device / dtype if necessary
        self.to(train_X)


class HeteroskedasticSingleTaskGP(SingleTaskGP):
    def __init__(self, train_X: Tensor, train_Y: Tensor, train_Yvar: Tensor) -> None:
        train_Y_log_var = torch.log(train_Yvar)
        noise_likelihood = GaussianLikelihood(
            noise_prior=SmoothedBoxPrior(-3, 5, 0.5, transform=torch.log),
            noise_constraint=GreaterThan(MIN_INFERRED_NOISE_LEVEL, transform=None),
        )
        noise_model = SingleTaskGP(
            train_X=train_X, train_Y=train_Y_log_var, likelihood=noise_likelihood
        )
        likelihood = _GaussianLikelihoodBase(HeteroskedasticNoise(noise_model))
        super().__init__(train_X=train_X, train_Y=train_Y, likelihood=likelihood)
        self.to(train_X)

    def reinitialize(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor,
        keep_params: bool = True,
        **kwargs,
    ) -> None:
        r"""Reinitialize model and the likelihood given new data.

        This does not refit the model.
        If device/dtype of the new training data are different from that of the
        model, then the model is moved to the new device/dtype.

        Args:
            train_X: A tensor of new training features
            train_Y: A tensor of new training observations
            train_Yvar: A tensor of new training noise observations
            keep_params: If True, keep the parameter values (speeds up refitting
                on similar data)
        """
        if keep_params:
            train_Y_log_var = torch.log(train_Yvar)
            self.likelihood.noise_covar.noise_model.reinitialize(
                train_X=train_X, train_Y=train_Y_log_var, keep_params=True
            )
            self.set_train_data(inputs=train_X, targets=train_Y, strict=False)
            self.to(train_X)
        else:
            self.__init__(train_X=train_X, train_Y=train_Y, train_Yvar=train_Yvar)
