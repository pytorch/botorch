# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from torch import Tensor


class FullyBayesianPosterior(GPyTorchPosterior):
    r"""A posterior for a fully Bayesian model."""

    def __init__(
        self, mvn: MultivariateNormal, marginalize_over_mcmc_samples: bool = False
    ) -> None:
        r"""A posterior for a fully Bayesian model.

        The MCMC batch dimension is -3.

        Args:
            mvn: A GPyTorch MultivariateNormal (single-output case)
            marginalize_over_mcmc_samples: If true, use the law of total variance to
                marginalize over the hyperparameter samples. This should always be
                false when computing acquisition functions.
        """
        super().__init__(mvn=mvn)
        self._mean = mvn.mean.unsqueeze(-1)
        self._variance = mvn.covariance_matrix.diagonal(dim1=-2, dim2=-1).unsqueeze(-1)
        if marginalize_over_mcmc_samples:
            num_mcmc_samples = self._variance.shape[-3]
            t1 = self._variance.sum(dim=-3) / num_mcmc_samples
            t2 = self._mean.pow(2).sum(dim=-3) / num_mcmc_samples
            t3 = -(self._mean.sum(dim=-3) / num_mcmc_samples).pow(2)
            self._variance = t1 + t2 + t3
            self._mean = self._mean.mean(dim=-3)
            self.mvn = None

    @property
    def mean(self) -> Tensor:
        r"""The posterior mean."""
        return self._mean

    @property
    def variance(self) -> Tensor:
        r"""The posterior variance."""
        return self._variance
