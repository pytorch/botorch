#! /usr/bin/env python3

import math
from copy import deepcopy
from typing import Optional

import torch
from gpytorch.distributions.multitask_multivariate_normal import (
    MultitaskMultivariateNormal,
)
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.kernels.multitask_kernel import MultitaskKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.likelihoods.gaussian_likelihood import (
    GaussianLikelihood,
    _GaussianLikelihoodBase,
)
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.likelihoods.noise_models import HeteroskedasticNoise
from gpytorch.means.constant_mean import ConstantMean
from gpytorch.means.multitask_mean import MultitaskMean
from gpytorch.models.exact_gp import ExactGP
from gpytorch.priors.lkj_prior import LKJCovariancePrior
from gpytorch.priors.smoothed_box_prior import SmoothedBoxPrior
from gpytorch.priors.torch_priors import GammaPrior
from torch import Tensor
from torch.nn.functional import softplus

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

    def reinitialize(
        self, train_X: Tensor, train_Y: Tensor, train_Y_se: Optional[Tensor] = None
    ) -> None:
        """
        Reinitialize model and the likelihood.

        Note: this does not refit the model.
        Note: train_Y_se is not used by SingleTaskGP
        """
        self.likelihood.load_state_dict(self._likelihood_state_dict)
        self.__init__(train_X=train_X, train_Y=train_Y, likelihood=self.likelihood)


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

    def reinitialize(
        self, train_X: Tensor, train_Y: Tensor, train_Y_se: Optional[Tensor] = None
    ) -> None:
        """
        Reinitialize model and the likelihood.

        Note: this does not refit the model.
        """
        assert train_Y_se is not None
        self.__init__(train_X=train_X, train_Y=train_Y, train_Y_se=train_Y_se)


class BlockMultiTaskGP(ExactGP, GPyTorchModel):
    """
    Class implementing a multi-task exact GP with Kronecker structure, using a
    simple ICM kernel. Requires that all targets are observed for every feature.

    Args:
        train_X: A `(b) x n x d` tensor of features (assumed normalized to [0, 1]^d)
        train_Y: A `(b) x n x t` tensor of observed outcomes (assumed standardized,
            i.e., zero mean and unit variance)
        likelihood: A MultitaskGaussianLikelihood
        rank (optional): The rank of the ICM kernel
        eta (optional): The parameter on the LKJ correlation prior. A value of 1.0
            is uninformative, values <1.0 favor stronger correlations (in magnitude),
            correlations vanish as eta -> inf.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        likelihood: Likelihood,
        rank: Optional[int] = None,
        eta: float = 2.0,
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
        num_tasks = train_Y.shape[-1]
        if rank is None:
            rank = num_tasks
        self.mean_module = MultitaskMean(
            ConstantMean(batch_size=batch_size), num_tasks=num_tasks
        )
        self.covar_module = MultitaskKernel(
            MaternKernel(
                nu=2.5,
                ard_num_dims=ard_num_dims,
                lengthscale_prior=GammaPrior(2.0, 5.0),
                param_transform=softplus,
                batch_size=batch_size,
            ),
            num_tasks=num_tasks,
            rank=rank,
            batch_size=batch_size,
            task_covar_prior=LKJCovariancePrior(
                n=num_tasks,
                eta=eta,
                sd_prior=SmoothedBoxPrior(math.exp(-6), math.exp(3), 0.05),
            ),
        )
        self._rank = rank
        self._eta = eta
        self._likelihood_state_dict = deepcopy(self.likelihood.state_dict())

    def forward(self, x: Tensor) -> MultitaskMultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultitaskMultivariateNormal(mean_x, covar_x)

    def reinitialize(
        self, train_X: Tensor, train_Y: Tensor, train_Y_se: Optional[Tensor] = None
    ) -> None:
        """
        Reinitialize model and the likelihood.

        Note: this does not refit the model.
        """
        if train_Y_se is not None:
            raise ValueError("Cannot re-initialize BlockMultiTaskGP with train_Y_se")
        self.likelihood.load_state_dict(self._likelihood_state_dict)
        self.__init__(
            train_X=train_X,
            train_Y=train_Y,
            likelihood=self.likelihood,
            rank=self._rank,
            eta=self._eta,
        )
