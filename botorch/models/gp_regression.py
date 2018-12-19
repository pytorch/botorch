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
from torch.nn.functional import softplus

from .fantasy_utils import _get_fantasy_state, _load_fantasy_state_dict
from .gpytorch import GPyTorchModel


class SingleTaskGP(ExactGP, GPyTorchModel):
    """
    Class implementing a single task exact GP using relatively strong priors on
    the Kernel hyperparameters, which work best when covariates are normalized
    to the unit cube and outcomes are standardized (zero mean, unit variance).
    """

    def __init__(
        self, train_X: Tensor, train_Y: Tensor, likelihood: Likelihood
    ) -> None:
        super().__init__(train_X, train_Y, likelihood)
        if train_X.ndimension() == 1:
            batch_size, ard_num_dims = 1, None
        elif train_X.ndimension() == 2:
            batch_size, ard_num_dims = 1, train_X.shape[-1]
        elif train_X.ndimension() == 3:
            batch_size, ard_num_dims = train_X.shape[0], train_X.shape[-1]
        else:
            raise ValueError(f"Unsupported shape {train_X.shape} for train_X.")
        self.mean_module = ConstantMean(batch_size=batch_size)
        self.covar_module = ScaleKernel(
            MaternKernel(
                nu=2.5,
                ard_num_dims=ard_num_dims,
                lengthscale_prior=GammaPrior(2.0, 5.0),
                param_transform=softplus,
            ),
            batch_size=batch_size,
            param_transform=softplus,
            outputscale_prior=GammaPrior(1.1, 0.05),
        )
        self._likelihood_state_dict = deepcopy(self.likelihood.state_dict())

    def forward(self, x: Tensor) -> MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def fantasize(
        self, X: Tensor, num_samples: int, base_samples: Optional[Tensor] = None
    ) -> "SingleTaskGP":
        state_dict, train_X, train_Y = _get_fantasy_state(
            model=self, X=X, num_samples=num_samples, base_samples=base_samples
        )
        fantasy_model = SingleTaskGP(
            train_X=train_X, train_Y=train_Y, likelihood=deepcopy(self.likelihood)
        )
        return _load_fantasy_state_dict(model=fantasy_model, state_dict=state_dict)

    def reinitialize(
        self, train_X: Tensor, train_Y: Tensor, train_Y_se: Optional[Tensor] = None
    ) -> None:
        """
        Reinitialize model and the likelihood.

        Note: this does not refit the model.
        Note: train_Y_se is not used by SingleTaskGP
        """
        self.likelihood.load_state_dict(self._likelihood_state_dict)
        self.__init__(train_X, train_Y, self.likelihood)


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
        super().__init__(train_X=train_X, train_Y=train_Y, likelihood=likelihood)

    def fantasize(
        self, X: Tensor, num_samples: int, base_samples: Optional[Tensor] = None
    ) -> "HeteroskedasticSingleTaskGP":
        state_dict, train_X, train_Y = _get_fantasy_state(
            model=self, X=X, num_samples=num_samples, base_samples=base_samples
        )
        noise_covar = self.likelihood.noise_covar
        train_Y_log_var = noise_covar.noise_model.train_targets
        # for now use the mean predictions for all noise "samples"
        # TODO: Sample from noise posterior
        noise_fantasies = noise_covar(X).diag()
        tYlv = torch.cat(
            [
                train_Y_log_var.expand(num_samples, -1),
                noise_fantasies.expand(num_samples, -1),
            ],
            dim=1,
        )
        train_Y_se = torch.exp(0.5 * tYlv)
        fantasy_model = self.__class__(
            train_X=train_X, train_Y=train_Y, train_Y_se=train_Y_se
        )
        return _load_fantasy_state_dict(model=fantasy_model, state_dict=state_dict)

    def reinitialize(
        self, train_X: Tensor, train_Y: Tensor, train_Y_se: Optional[Tensor] = None
    ) -> None:
        """
        Reinitialize model and the likelihood.

        Note: this does not refit the model.
        """
        assert train_Y_se is not None
        self.__init__(train_X=train_X, train_Y=train_Y, train_Y_se=train_Y_se)
