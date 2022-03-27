# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


r"""Gaussian Process Regression models with fully Bayesian inference.

We use a lightweight PyTorch implementation of a Matern-5/2 kernel as there are some
performance issues with running NUTS on top of standard GPyTorch models. The resulting
hyperparameter samples are loaded into a batched GPyTorch model after fitting.

References:

.. [Eriksson2021saasbo]
    D. Eriksson, M. Jankowiak. High-Dimensional Bayesian Optimization
    with Sparse Axis-Aligned Subspaces. Proceedings of the Thirty-
    Seventh Conference on Uncertainty in Artificial Intelligence, 2021.
"""


import math
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Union

import pyro
import torch
from botorch.acquisition.objective import PosteriorTransform
from botorch.models.gp_regression import SingleTaskGP, FixedNoiseGP
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.models.utils import validate_input_scaling
from botorch.posteriors.fully_bayesian import FullyBayesianPosterior
from botorch.sampling.samplers import MCSampler
from botorch.utils.containers import TrainingData
from gpytorch.constraints import GreaterThan
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.kernels.kernel import Distance
from gpytorch.likelihoods.gaussian_likelihood import (
    FixedNoiseGaussianLikelihood,
    GaussianLikelihood,
)
from gpytorch.means.constant_mean import ConstantMean
from torch import Tensor

MIN_INFERRED_NOISE_LEVEL = 1e-6


def matern52_kernel(X: Tensor, lengthscale: Tensor) -> Tensor:
    """Matern-5/2 kernel."""
    nu = 5 / 2
    dist = compute_dists(X=X, lengthscale=lengthscale)
    exp_component = torch.exp(-math.sqrt(nu * 2) * dist)
    constant_component = (math.sqrt(5) * dist).add(1).add(5.0 / 3.0 * (dist ** 2))
    return constant_component * exp_component


def compute_dists(X: Tensor, lengthscale: Tensor) -> Tensor:
    """Compute kernel distances."""
    return Distance()._dist(
        X / lengthscale, X / lengthscale, postprocess=False, x1_eq_x2=True
    )


class PyroModel:
    r"""Abstract base class for a Pyro model."""

    @abstractmethod
    def sample(self) -> None:
        r"""Sample from the model."""
        pass  # pragma: no cover

    @abstractmethod
    def postprocess_mcmc_samples(
        self, mcmc_samples: Dict[str, Tensor], **kwargs: Any
    ) -> Dict[str, Tensor]:
        """Post-process the final MCMC samples."""
        pass  # pragma: no cover


class SaasPyroModel(PyroModel):
    r"""Implementation of the sparse axis-aligned subspace priors (SAAS) model.

    The SAAS model uses sparsity-inducing priors to identift the most important
    parameters. This model is suitable for high-dimensional BO with potentially
    hundreds of tunable parameters. See [Eriksson2021saasbo]_ for more details.
    """

    def __init__(
        self, train_X: Tensor, train_Y: Tensor, train_Yvar: Optional[Tensor] = None
    ):
        r"""Initialize the SAAS model.

        Args:
            train_X: Training inputs (n x d)
            train_Y: Training targets (n x 1)
            train_Yvar: Observed noise variance (n x 1). Inferred if None.
        """
        self.train_X = train_X
        self.train_Y = train_Y
        self.train_Yvar = train_Yvar

    def sample(self) -> None:
        r"""Sample from the SAAS model.

        This samples the mean, noise variance, outputscale, and lengthscales according
        to the SAAS prior.
        """
        tkwargs = {"dtype": self.train_X.dtype, "device": self.train_X.device}
        outputscale = self.sample_outputscale(concentration=2.0, rate=0.15, **tkwargs)
        mean = self.sample_mean(**tkwargs)
        noise = self.sample_noise(**tkwargs)
        lengthscale = self.sample_lengthscale(dim=self.train_X.shape[-1], **tkwargs)
        k = matern52_kernel(X=self.train_X, lengthscale=lengthscale)
        k = outputscale * k + noise * torch.eye(self.train_X.shape[0], **tkwargs)
        pyro.sample(
            "Y",
            pyro.distributions.MultivariateNormal(
                loc=mean.view(-1).expand(self.train_X.shape[0]),
                covariance_matrix=k,
            ),
            obs=self.train_Y.squeeze(-1),
        )

    def sample_outputscale(
        self, concentration: float = 2.0, rate: float = 0.15, **tkwargs: Any
    ) -> Tensor:
        r"""Sample the outputscale."""
        return pyro.sample(
            "outputscale",
            pyro.distributions.Gamma(
                torch.tensor(concentration, **tkwargs),
                torch.tensor(rate, **tkwargs),
            ),
        )

    def sample_mean(self, **tkwargs: Any) -> Tensor:
        r"""Sample the mean constant."""
        return pyro.sample(
            "mean",
            pyro.distributions.Normal(
                torch.tensor(0.0, **tkwargs),
                torch.tensor(1.0, **tkwargs),
            ),
        )

    def sample_noise(self, **tkwargs: Any) -> Tensor:
        r"""Sample the noise variance."""
        if self.train_Yvar is None:
            return pyro.sample(
                "noise",
                pyro.distributions.Gamma(
                    torch.tensor(0.9, **tkwargs),
                    torch.tensor(10.0, **tkwargs),
                ),
            )
        else:
            return self.train_Yvar

    def sample_lengthscale(
        self, dim: int, alpha: float = 0.1, **tkwargs: Any
    ) -> Tensor:
        r"""Sample the lengthscale."""
        tausq = pyro.sample(
            "kernel_tausq",
            pyro.distributions.HalfCauchy(torch.tensor(alpha, **tkwargs)),
        )
        inv_length_sq = pyro.sample(
            "_kernel_inv_length_sq",
            pyro.distributions.HalfCauchy(torch.ones(dim, **tkwargs)),
        )
        inv_length_sq = pyro.deterministic(
            "kernel_inv_length_sq", tausq * inv_length_sq
        )
        lengthscale = pyro.deterministic(
            "lengthscale",
            (1.0 / inv_length_sq).sqrt(),
        )
        return lengthscale

    def postprocess_mcmc_samples(
        self, mcmc_samples: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        r"""Post-process the MCMC samples.

        This computes the true lengthscales and removes the inverse lengthscales and
        tausq (global shrinkage).
        """
        inv_length_sq = (
            mcmc_samples["kernel_tausq"].unsqueeze(-1)
            * mcmc_samples["_kernel_inv_length_sq"]
        )
        mcmc_samples["lengthscale"] = (1.0 / inv_length_sq).sqrt()
        # Delete `kernel_tausq` and `_kernel_inv_length_sq` since they aren't loaded
        # into the final model.
        del mcmc_samples["kernel_tausq"], mcmc_samples["_kernel_inv_length_sq"]
        return mcmc_samples


class SaasFullyBayesianSingleTaskGP(SingleTaskGP):
    r"""A fully Bayesian single-task GP model with the SAAS prior.

    This model assumes that the inputs have been normalized to [0, 1]^d and that
    the output has been standardizes to have zero mean and unit variance. The SAAS
    model [Eriksson2021saasbo]_ with a Matern-5/2 is used by default.

    You are expected to use `fit_fully_bayesian_model_nuts` to fit this model as it
    isn't compatible with `fit_gpytorch_model`.

    Example:
    >>> saas_gp = SaasFullyBayesianSingleTaskGP(train_X, train_Y)
    >>> fit_fully_bayesian_model_nuts(saas_gp)
    >>> posterior = saas_gp.posterior(test_X, marginalize_over_mcmc_samples=True)
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Optional[Tensor] = None,
        outcome_transform: Optional[OutcomeTransform] = None,
        input_transform: Optional[InputTransform] = None,
    ) -> None:
        r"""Initialize the fully Bayesian single-task GP model.

        Args:
            train_X: Training inputs (n x d)
            train_Y: Training targets (n x 1)
            train_Yvar: Observed noise variance (n x 1). Inferred if None.
        """
        if not (
            train_X.ndim == train_Y.ndim == 2
            and len(train_X) == len(train_Y)
            and train_Y.shape[-1] == 1
        ):
            raise ValueError(
                "Expected train_X to have shape n x d and train_Y to have shape n x 1"
            )
        if train_Yvar is not None:
            if train_Y.shape != train_Yvar.shape:
                raise ValueError(
                    "Expected train_Yvar to be None or have the same shape as train_Y"
                )
        with torch.no_grad():
            transformed_X = self.transform_inputs(
                X=train_X, input_transform=input_transform
            )
        if outcome_transform is not None:
            train_Y, train_Yvar = outcome_transform(train_Y, train_Yvar)
        self._validate_tensor_args(X=transformed_X, Y=train_Y)
        validate_input_scaling(
            train_X=transformed_X, train_Y=train_Y, train_Yvar=train_Yvar
        )
        self._set_dimensions(train_X=train_X, train_Y=train_Y)
        if train_Yvar is not None:  # Clamp after transforming
            train_Yvar = train_Yvar.clamp(MIN_INFERRED_NOISE_LEVEL)

        super().__init__(train_X, train_Y)
        self.train_X = train_X
        self.train_Y = train_Y
        self.train_Yvar = train_Yvar
        self.mean_module = None
        self.covar_module = None
        self.likelihood = None
        self.pyro_model = SaasPyroModel(
            train_X=transformed_X, train_Y=train_Y, train_Yvar=train_Yvar
        )
        if outcome_transform is not None:
            self.outcome_transform = outcome_transform
        if input_transform is not None:
            self.input_transform = input_transform

    def _check_if_fitted(self):
        r"""Raise an exception if the model hasn't been fitted."""
        if self.covar_module is None:
            raise RuntimeError(
                "Model has not been fitted. You need to call "
                "`fit_fully_bayesian_model_nuts` to fit the model."
            )

    @property
    def median_lengthscale(self) -> Tensor:
        r"""Median lengthscales across the MCMC samples."""
        self._check_if_fitted()
        lengthscale = self.covar_module.base_kernel.lengthscale.clone()
        return lengthscale.median(0).values.squeeze(0)

    @property
    def num_mcmc_samples(self) -> int:
        r"""Number of MCMC samples in the model."""
        self._check_if_fitted()
        return len(self.covar_module.outputscale)

    def fantasize(
        self,
        X: Tensor,
        sampler: MCSampler,
        observation_noise: Union[bool, Tensor] = True,
        **kwargs: Any,
    ) -> FixedNoiseGP:
        raise NotImplementedError("Fantasize is not implemented!")

    def train(self, mode: bool = True) -> None:
        r"""Puts the model in `train` mode."""
        super().train(mode=mode)
        if mode and self.train_X.ndim == 3:
            self.train_X = self.train_X[0, ...].squeeze(0)
            self.mean_module = None
            self.covar_module = None
            self.likelihood = None

    def load_mcmc_samples(self, mcmc_samples: Dict[str, Tensor]) -> None:
        r"""Load the MCMC hyperparameter samples into the model.

        This method will be called by `fit_fully_bayesian_model_nuts` when the model
        has been fitted in order to create a batched SingleTaskGP model.
        """
        tkwargs = {"device": self.train_X.device, "dtype": self.train_X.dtype}
        num_mcmc_samples = len(mcmc_samples["mean"])
        batch_shape = torch.Size([num_mcmc_samples])

        self.train_X = self.train_X.unsqueeze(0).expand(
            num_mcmc_samples, self.train_X.shape[0], -1
        )
        self.mean_module = ConstantMean(batch_shape=batch_shape).to(**tkwargs)
        self.covar_module = ScaleKernel(
            base_kernel=MaternKernel(
                ard_num_dims=self.train_X.shape[-1],
                batch_shape=batch_shape,
            ),
            batch_shape=batch_shape,
        ).to(**tkwargs)
        if self.train_Yvar is not None:
            self.likelihood = FixedNoiseGaussianLikelihood(
                noise=self.train_Yvar, batch_shape=batch_shape
            ).to(**tkwargs)
        else:
            self.likelihood = GaussianLikelihood(
                batch_shape=batch_shape,
                noise_constraint=GreaterThan(MIN_INFERRED_NOISE_LEVEL),
            ).to(**tkwargs)
            self.likelihood.noise_covar.noise = (
                mcmc_samples["noise"]
                .detach()
                .clone()
                .view(self.likelihood.noise_covar.noise.shape)
                .clamp_min(MIN_INFERRED_NOISE_LEVEL)
                .to(**tkwargs)
            )

        self.covar_module.base_kernel.lengthscale = (
            mcmc_samples["lengthscale"]
            .detach()
            .clone()
            .view(self.covar_module.base_kernel.lengthscale.shape)
            .to(**tkwargs)
        )
        self.covar_module.outputscale = (
            mcmc_samples["outputscale"]
            .detach()
            .clone()
            .view(self.covar_module.outputscale.shape)
            .to(**tkwargs)
        )
        self.mean_module.constant.data = (
            mcmc_samples["mean"]
            .detach()
            .clone()
            .view(self.mean_module.constant.shape)
            .to(**tkwargs)
        )

    def forward(self, X: Tensor) -> MultivariateNormal:
        self._check_if_fitted()
        return super().forward(X.unsqueeze(-3))

    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: bool = False,
        posterior_transform: Optional[PosteriorTransform] = None,
        marginalize_over_mcmc_samples: bool = False,
        **kwargs: Any,
    ) -> FullyBayesianPosterior:
        r"""Computes the posterior over model outputs at the provided points.

        Args:
            X: A `(batch_shape) x q x d`-dim Tensor, where `d` is the dimension
                of the feature space and `q` is the number of points considered
                jointly.
            output_indices: A list of indices, corresponding to the outputs over
                which to compute the posterior (if the model is multi-output).
                Can be used to speed up computation if only a subset of the
                model's outputs are required for optimization. If omitted,
                computes the posterior over all model outputs.
            observation_noise: If True, add the observation noise from the
                likelihood to the posterior. If a Tensor, use it directly as the
                observation noise (must be of shape `(batch_shape) x q x m`).
            posterior_transform: An optional PosteriorTransform.
            marginalize_over_mcmc_samples: If True, we will use the law of total
                variance to marginalize over the MCMC samples. This is useful when
                making test predictions, but shouldn't be used when computing
                acquisition functions values.

        Returns:
            A `FullyBayesianPosterior` object. Includes observation noise if specified.
        """
        self._check_if_fitted()
        posterior = super().posterior(
            X=X,
            output_indices=output_indices,
            observation_noise=observation_noise,
            posterior_transform=posterior_transform,
            **kwargs,
        )
        posterior = FullyBayesianPosterior(
            mvn=posterior.mvn,
            marginalize_over_mcmc_samples=marginalize_over_mcmc_samples,
        )
        return posterior

    @classmethod
    def construct_inputs(
        cls, training_data: TrainingData, **kwargs: Any
    ) -> Dict[str, Any]:
        r"""Construct kwargs for the `Model` from `TrainingData` and other options.

        Args:
            training_data: `TrainingData` container with data for single outcome
                or for multiple outcomes for batched multi-output case.
            **kwargs: None expected for this class.
        """
        inputs = {"train_X": training_data.X, "train_Y": training_data.Y}
        if training_data.Yvar is not None:
            inputs["train_Yvar"] = training_data.Yvar
        return inputs
