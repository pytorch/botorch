#! /usr/bin/env python3

from copy import deepcopy
from typing import Optional

import torch
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.likelihoods.gaussian_likelihood import (
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


class SingleTaskGP(ExactGP, GPyTorchModel):
    """A single-task Exact GP model.

    Class implementing a single task exact GP using relatively strong priors on
    the Kernel hyperparameters, which work best when covariates are normalized
    to the unit cube and outcomes are standardized (zero mean, unit variance).
    """

    def __init__(
        self, train_X: Tensor, train_Y: Tensor, likelihood: Optional[Likelihood] = None
    ) -> None:
        """A single-task Exact GP model.
        Args:
            train_X: A `n x d` or `b x n x d` (batch mode) tensor of training data
            train_Y: A `n` or `b x n` (batch mode) tensor of training observations
            likelihood: A likelihood. If omitted, use a standard GaussianLikelihood
                with inferred noise level.
        """
        if train_X.ndimension() == 1:
            batch_size, ard_num_dims = 1, None
        elif train_X.ndimension() == 2:
            batch_size, ard_num_dims = 1, train_X.shape[-1]
        elif train_X.ndimension() == 3:
            batch_size, ard_num_dims = train_X.shape[0], train_X.shape[-1]
        else:
            raise ValueError(f"Unsupported shape {train_X.shape} for train_X.")
        if likelihood is None:
            likelihood = GaussianLikelihood(
                noise_prior=GammaPrior(0.1, 0.01), batch_size=batch_size
            )
            likelihood.parameter_bounds = {"noise_covar.raw_noise": (-15, None)}
        else:
            self._likelihood_state_dict = deepcopy(likelihood.state_dict())
        super().__init__(train_X, train_Y, likelihood)
        self.mean_module = ConstantMean(batch_size=batch_size)
        self.covar_module = ScaleKernel(
            MaternKernel(
                nu=2.5,
                ard_num_dims=ard_num_dims,
                lengthscale_prior=GammaPrior(3.0, 6.0),
            ),
            batch_size=batch_size,
            outputscale_prior=GammaPrior(2.0, 0.15),
        )

    def forward(self, x: Tensor) -> MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def reinitialize(
        self, train_X: Tensor, train_Y: Tensor, keep_params: bool = True, **kwargs
    ) -> None:
        """Reinitialize model and the likelihood.

        Args:
            train_X: A tensor of new training data
            train_Y: A tensor of new training observations
            keep_params: If True, keep the parameter values (speeds up refitting
                on similar data)

        Note: this does not refit the model.
        """
        if keep_params:
            self.set_train_data(inputs=train_X, targets=train_Y, strict=False)
        elif hasattr(self, "_likelihood_state_dict"):
            self.likelihood.load_state_dict(self._likelihood_state_dict)
            self.__init__(train_X=train_X, train_Y=train_Y, likelihood=self.likelihood)
        else:
            self.__init__(train_X=train_X, train_Y=train_Y)


class HeteroskedasticSingleTaskGP(SingleTaskGP):
    def __init__(self, train_X: Tensor, train_Y: Tensor, train_Y_se: Tensor) -> None:
        train_Y_log_var = 2 * torch.log(train_Y_se)
        noise_likelihood = GaussianLikelihood(
            noise_prior=SmoothedBoxPrior(-3, 5, 0.5, transform=torch.log)
        )
        noise_model = SingleTaskGP(
            train_X=train_X, train_Y=train_Y_log_var, likelihood=noise_likelihood
        )
        likelihood = _GaussianLikelihoodBase(HeteroskedasticNoise(noise_model))
        likelihood.parameter_bounds = {
            "noise_covar.noise_model.likelihood.noise_covar.raw_noise": (-15, None)
        }
        super().__init__(train_X=train_X, train_Y=train_Y, likelihood=likelihood)

    def reinitialize(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Y_se: Optional[Tensor] = None,
        keep_params: bool = True,
    ) -> None:
        """Reinitialize model and the likelihood.

        Args:
            train_X: A tensor of new training data
            train_Y: A tensor of new training observations
            train_y_se: A tensor of new training noise observations
            keep_params: If True, keep the parameter values (speeds up refitting
                on similar data)

        Note: this does not refit the model.
        """
        if train_Y_se is None:
            raise RuntimeError("HeteroskedasticSingleTaskGP requires observation noise")
        if keep_params:
            train_Y_log_var = 2 * torch.log(train_Y_se)
            self.likelihood.noise_covar.noise_model.reinitialize(
                train_X=train_X, train_Y=train_Y_log_var, keep_params=True
            )
            self.set_train_data(inputs=train_X, targets=train_Y, strict=False)
        else:
            self.__init__(train_X=train_X, train_Y=train_Y, train_Y_se=train_Y_se)
