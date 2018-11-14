#! /usr/bin/env python3

from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from gpytorch.likelihoods import Likelihood
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP
from gpytorch.priors import GammaPrior
from torch import Tensor


class GPRegressionModel(ExactGP):
    def __init__(
        self, train_x: Tensor, train_y: Tensor, likelihood: Likelihood
    ) -> None:
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        if train_x.ndimension() == 1:
            batch_size, ard_num_dims = 1, None
        elif train_x.ndimension() == 2:
            batch_size, ard_num_dims = 1, train_x.shape[-1]
        elif train_x.ndimension() == 3:
            batch_size, ard_num_dims = train_x.shape[0], train_x.shape[-1]
        else:
            raise ValueError(f"Unsupported shape {train_x.shape} for train_x.")
        self.mean_module = ConstantMean(batch_size=batch_size)
        self.covar_module = ScaleKernel(
            RBFKernel(ard_num_dims=ard_num_dims, batch_size=batch_size),
            batch_size=batch_size,
        )

    def forward(self, x: Tensor) -> MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class SingleTaskGP(ExactGP):
    """
    Class implementing a single task exact GP using relatively strong priors on
    the Kernel hyperparameters, which work best when covariates are normalized
    to the unit cube and outcomes are standardized (zero mean, unit variance).
    """

    def __init__(
        self, train_X: Tensor, train_Y: Tensor, likelihood: Likelihood
    ) -> None:
        super().__init__(train_X, train_Y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(
            MaternKernel(
                nu=2.5,
                ard_num_dims=train_X.shape[1] if train_X.ndimension() > 1 else None,
                log_lengthscale_prior=GammaPrior(2.0, 5.0, log_transform=True),
            ),
            log_outputscale_prior=GammaPrior(1.1, 0.05, log_transform=True),
        )

    def forward(self, x: Tensor) -> MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
