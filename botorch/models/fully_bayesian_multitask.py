# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


r"""Multi-task Gaussian Process Regression models with fully Bayesian inference.
"""


from typing import Any, Dict, List, Optional, Tuple, Union

import pyro
import torch
from botorch.acquisition.objective import PosteriorTransform
from botorch.models.fully_bayesian import (
    matern52_kernel,
    MIN_INFERRED_NOISE_LEVEL,
    PyroModel,
    reshape_and_detach,
    SaasPyroModel,
)
from botorch.models.multitask import FixedNoiseMultiTaskGP
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.posteriors.fully_bayesian import FullyBayesianPosterior, MCMC_DIM
from botorch.sampling.samplers import MCSampler
from botorch.utils.datasets import SupervisedDataset
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.kernels import MaternKernel
from gpytorch.kernels.kernel import Kernel
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.means.mean import Mean
from torch import Tensor
from torch.nn.parameter import Parameter


class MultitaskSaasPyroModel(SaasPyroModel):
    r"""
    Implementation of the multi-task sparse axis-aligned subspace priors (SAAS) model.

    The multi-task model uses an ICM kernel. The data kernel is same as the single task
    SAAS model in order to handle high-dimensional parameter spaces. The task kernel
    is a Matern-5/2 kernel using learned task embeddings as the input.
    """

    def set_inputs(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor,
        task_feature: int,
        task_rank: Optional[int] = None,
    ):
        """Set the training data.

        Args:
            train_X: Training inputs (n x (d + 1))
            train_Y: Training targets (n x 1)
            train_Yvar: Observed noise variance (n x 1).
            task_feature: The index of the task feature (`-d <= task_feature <= d`).
            task_rank: The num of learned task embeddings to be used in the task kernel.
                If omitted, set it to be 1.
        """
        if (train_Yvar is None) or (torch.isnan(train_Yvar).all()):
            # TODO: revisit inferred noise for batched MTGP
            raise NotImplementedError(
                "Currently do not support inferred noise for multitask GP with MCMC!"
            )
        super().set_inputs(train_X, train_Y, train_Yvar)
        # obtain a list of task indicies
        all_tasks = train_X[:, task_feature].unique().to(dtype=torch.long).tolist()
        self.task_feature = task_feature
        self.num_tasks = len(all_tasks)
        self.task_rank = task_rank or 1
        # assume there is one column for task feature
        self.ard_num_dims = self.train_X.shape[-1] - 1

    def sample(self) -> None:
        r"""Sample from the SAAS model.

        This samples the mean, noise variance, outputscale, and lengthscales according
        to the SAAS prior.
        """
        tkwargs = {"dtype": self.train_X.dtype, "device": self.train_X.device}
        base_idxr = torch.arange(self.ard_num_dims, **{"device": tkwargs["device"]})
        base_idxr[self.task_feature :] += 1  # exclude task feature
        task_indices = self.train_X[..., self.task_feature].to(
            device=tkwargs["device"], dtype=torch.long
        )

        outputscale = self.sample_outputscale(concentration=2.0, rate=0.15, **tkwargs)
        mean = self.sample_mean(**tkwargs)
        noise = self.sample_noise(**tkwargs)

        lengthscale = self.sample_lengthscale(dim=self.ard_num_dims, **tkwargs)
        k = matern52_kernel(X=self.train_X[..., base_idxr], lengthscale=lengthscale)

        # compute task covar matrix
        task_latent_features = self.sample_latent_features(**tkwargs)[task_indices]
        task_lengthscale = self.sample_task_lengthscale(**tkwargs)
        task_covar = matern52_kernel(
            X=task_latent_features, lengthscale=task_lengthscale
        )
        k = k.mul(task_covar)
        k = outputscale * k + noise * torch.eye(self.train_X.shape[0], **tkwargs)
        pyro.sample(
            "Y",
            pyro.distributions.MultivariateNormal(
                loc=mean.view(-1).expand(self.train_X.shape[0]),
                covariance_matrix=k,
            ),
            obs=self.train_Y.squeeze(-1),
        )

    def sample_latent_features(self, **tkwargs: Any):
        return pyro.sample(
            "latent_features",
            pyro.distributions.Normal(
                torch.tensor(0.0, **tkwargs),
                torch.tensor(1.0, **tkwargs),
            ).expand(torch.Size([self.num_tasks, self.task_rank])),
        )

    def sample_task_lengthscale(
        self, concentration: float = 6.0, rate: float = 3.0, **tkwargs: Any
    ):
        return pyro.sample(
            "task_lengthscale",
            pyro.distributions.Gamma(
                torch.tensor(concentration, **tkwargs),
                torch.tensor(rate, **tkwargs),
            ).expand(torch.Size([self.task_rank])),
        )

    def load_mcmc_samples(
        self, mcmc_samples: Dict[str, Tensor]
    ) -> Tuple[Mean, Kernel, Likelihood, Kernel, Parameter]:
        r"""Load the MCMC samples into the mean_module, covar_module, and likelihood."""
        tkwargs = {"device": self.train_X.device, "dtype": self.train_X.dtype}
        num_mcmc_samples = len(mcmc_samples["mean"])
        batch_shape = torch.Size([num_mcmc_samples])

        mean_module, covar_module, likelihood = super().load_mcmc_samples(
            mcmc_samples=mcmc_samples
        )

        task_covar_module = MaternKernel(
            nu=2.5,
            ard_num_dims=self.task_rank,
            batch_shape=batch_shape,
        ).to(**tkwargs)
        task_covar_module.lengthscale = reshape_and_detach(
            target=task_covar_module.lengthscale,
            new_value=mcmc_samples["task_lengthscale"],
        )
        latent_features = Parameter(
            torch.rand(
                batch_shape + torch.Size([self.num_tasks, self.task_rank]),
                requires_grad=True,
                **tkwargs,
            )
        )
        latent_features = reshape_and_detach(
            target=latent_features,
            new_value=mcmc_samples["latent_features"],
        )
        return mean_module, covar_module, likelihood, task_covar_module, latent_features


class SaasFullyBayesianMultiTaskGP(FixedNoiseMultiTaskGP):
    r"""A fully Bayesian multi-task GP model with the SAAS prior.

    This model assumes that the inputs have been normalized to [0, 1]^d and that the
    output has been stratified standardized to have zero mean and unit variance for
    each task.The SAAS model [Eriksson2021saasbo]_ with a Matern-5/2 is used as data
    kernel by default.

    You are expected to use `fit_fully_bayesian_model_nuts` to fit this model as it
    isn't compatible with `fit_gpytorch_model`.

    Example:
        >>> X1, X2 = torch.rand(10, 2), torch.rand(20, 2)
        >>> i1, i2 = torch.zeros(10, 1), torch.ones(20, 1)
        >>> train_X = torch.cat([
        >>>     torch.cat([X1, i1], -1), torch.cat([X2, i2], -1),
        >>> ])
        >>> train_Y = torch.cat(f1(X1), f2(X2)).unsqueeze(-1)
        >>> train_Yvar = 0.01 * torch.ones_like(train_Y)
        >>> mtsaas_gp = SaasFullyBayesianFixedNoiseMultiTaskGP(
        >>>     train_X, train_Y, train_Yvar, task_feature=-1,
        >>> )
        >>> fit_fully_bayesian_model_nuts(mtsaas_gp)
        >>> posterior = mtsaas_gp.posterior(test_X)
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor,
        task_feature: int,
        output_tasks: Optional[List[int]] = None,
        rank: Optional[int] = None,
        outcome_transform: Optional[OutcomeTransform] = None,
        input_transform: Optional[InputTransform] = None,
        pyro_model: Optional[PyroModel] = None,
    ) -> None:
        r"""Initialize the fully Bayesian multi-task GP model.

        Args:
            train_X: Training inputs (n x (d + 1))
            train_Y: Training targets (n x 1)
            train_Yvar: Observed noise variance (n x 1).
            task_feature: The index of the task feature (`-d <= task_feature <= d`).
            rank: The num of learned task embeddings to be used in the task kernel.
                If omitted, set it to be 1.
        """
        if not (
            train_X.ndim == train_Y.ndim == 2
            and len(train_X) == len(train_Y)
            and train_Y.shape[-1] == 1
        ):
            raise ValueError(
                "Expected train_X to have shape n x d and train_Y to have shape n x 1"
            )
        if train_Yvar is None:
            raise NotImplementedError(
                "Inferred Noise is not supported in multitask SAAS GP."
            )
        else:
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

        train_Yvar = train_Yvar.clamp(MIN_INFERRED_NOISE_LEVEL)

        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            task_feature=task_feature,
            output_tasks=output_tasks,
        )
        self.to(train_X)

        self.mean_module = None
        self.covar_module = None
        self.likelihood = None
        self.task_covar_module = None
        if pyro_model is None:
            pyro_model = MultitaskSaasPyroModel()
        pyro_model.set_inputs(
            train_X=transformed_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            task_feature=task_feature,
            task_rank=rank,
        )
        self.pyro_model = pyro_model
        if outcome_transform is not None:
            self.outcome_transform = outcome_transform
        if input_transform is not None:
            self.input_transform = input_transform

    def train(self, mode: bool = True) -> None:
        r"""Puts the model in `train` mode."""
        super().train(mode=mode)
        if mode:
            self.mean_module = None
            self.covar_module = None
            self.likelihood = None
            self.task_covar_module = None

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
    ) -> FixedNoiseMultiTaskGP:
        raise NotImplementedError("Fantasize is not implemented!")

    def _check_if_fitted(self):
        r"""Raise an exception if the model hasn't been fitted."""
        if self.covar_module is None:
            raise RuntimeError(
                "Model has not been fitted. You need to call "
                "`fit_fully_bayesian_model_nuts` to fit the model."
            )

    def load_mcmc_samples(self, mcmc_samples: Dict[str, Tensor]) -> None:
        r"""Load the MCMC hyperparameter samples into the model.

        This method will be called by `fit_fully_bayesian_model_nuts` when the model
        has been fitted in order to create a batched SingleTaskGP model.
        """
        (
            self.mean_module,
            self.covar_module,
            self.likelihood,
            self.task_covar_module,
            self.latent_features,
        ) = self.pyro_model.load_mcmc_samples(mcmc_samples=mcmc_samples)

    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: bool = False,
        posterior_transform: Optional[PosteriorTransform] = None,
        **kwargs: Any,
    ) -> FullyBayesianPosterior:
        r"""Computes the posterior over model outputs at the provided points.

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
        posterior = FullyBayesianPosterior(mvn=posterior.mvn)
        return posterior

    def forward(self, X: Tensor) -> MultivariateNormal:
        self._check_if_fitted()
        X = X.unsqueeze(MCMC_DIM)

        x_basic, task_idcs = self._split_inputs(X)

        mean_x = self.mean_module(x_basic)
        covar_x = self.covar_module(x_basic)

        tsub_idcs = task_idcs.squeeze(-3).squeeze(-1)
        latent_features = self.latent_features[:, tsub_idcs, :]

        if X.ndim > 3:
            # batch eval mode
            # for X (batch_shape x num_samples x q x d), task_idcs[:,i,:,] are the same
            # reshape X to (batch_shape x num_samples x q x d)
            latent_features = latent_features.permute(
                [-i for i in range(X.ndim - 1, 2, -1)]
                + [0]
                + [-i for i in range(2, 0, -1)]
            )

        # Combine the two in an ICM fashion
        covar_i = self.task_covar_module(latent_features)
        covar = covar_x.mul(covar_i)
        return MultivariateNormal(mean_x, covar)

    @classmethod
    def construct_inputs(
        cls,
        training_data: Dict[str, SupervisedDataset],
        task_feature: int,
        rank: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        r"""Construct `Model` keyword arguments from dictionary of `SupervisedDataset`.

        Args:
            training_data: Dictionary of `SupervisedDataset`.
            task_feature: Column index of embedded task indicator features. For details,
                see `parse_training_data`.
            rank: The rank of the cross-task covariance matrix.
        """

        inputs = super().construct_inputs(
            training_data=training_data, task_feature=task_feature, rank=rank, **kwargs
        )
        inputs.pop("task_covar_prior")
        return inputs
