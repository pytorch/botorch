#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Gaussian Process Regression models based on GPyTorch models.

These models are often a good starting point and are further documented in the
tutorials.

`SingleTaskGP`, `FixedNoiseGP`, and `HeteroskedasticSingleTaskGP` are all
single-task exact GP models, differing in how they treat noise. They use
relatively strong priors on the Kernel hyperparameters, which work best when
covariates are normalized to the unit cube and outcomes are standardized (zero
mean, unit variance).

These models all work in batch mode (each batch having its own hyperparameters).
When the training observations include multiple outputs, these models use
batching to model outputs independently.

These models all support multiple outputs. However, as single-task models,
`SingleTaskGP`, `FixedNoiseGP`, and `HeteroskedasticSingleTaskGP` should be
used only when the outputs are independent and all use the same training data.
If outputs are independent and outputs have different training data, use the
`ModelListGP`. When modeling correlations between outputs, use a multi-task
model like `MultiTaskGP`.
"""

from __future__ import annotations

import warnings
from typing import NoReturn, Optional

import torch
from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel
from botorch.models.model import FantasizeMixin
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import Log, OutcomeTransform
from botorch.models.utils import validate_input_scaling
from botorch.models.utils.gpytorch_modules import (
    get_gaussian_likelihood_with_gamma_prior,
    get_matern_kernel_with_gamma_prior,
    MIN_INFERRED_NOISE_LEVEL,
)
from gpytorch.constraints.constraints import GreaterThan
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.likelihoods.gaussian_likelihood import (
    _GaussianLikelihoodBase,
    FixedNoiseGaussianLikelihood,
    GaussianLikelihood,
)
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.likelihoods.noise_models import HeteroskedasticNoise
from gpytorch.means.constant_mean import ConstantMean
from gpytorch.means.mean import Mean
from gpytorch.mlls.noise_model_added_loss_term import NoiseModelAddedLossTerm
from gpytorch.models.exact_gp import ExactGP
from gpytorch.module import Module
from gpytorch.priors.smoothed_box_prior import SmoothedBoxPrior
from torch import Tensor


class SingleTaskGP(BatchedMultiOutputGPyTorchModel, ExactGP, FantasizeMixin):
    r"""A single-task exact GP model, supporting both known and inferred noise levels.

    A single-task exact GP using relatively strong priors on the Kernel
    hyperparameters, which work best when covariates are normalized to the unit
    cube and outcomes are standardized (zero mean, unit variance).

    This model works in batch mode (each batch having its own hyperparameters).
    When the training observations include multiple outputs, this model will use
    batching to model outputs independently.

    Use this model when you have independent output(s) and all outputs use the
    same training data. If outputs are independent and outputs have different
    training data, use the ModelListGP. When modeling correlations between
    outputs, use the MultiTaskGP.

    An example of a case in which noise levels are known is online
    experimentation, where noise can be measured using the variability of
    different observations from the same arm, or provided by outside software.
    Another use case is simulation optimization, where the evaluation can
    provide variance estimates, perhaps from bootstrapping. In any case, these
    noise levels can be provided to `SingleTaskGP` as `train_Yvar`.

    `SingleTaskGP` can also be used when the observations are known to be
    noise-free. Noise-free observations can be modeled using arbitrarily small
    noise values, such as `train_Yvar=torch.full_like(train_Y, 1e-6)`.

    Example:
        Model with inferred noise levels:

        >>> import torch
        >>> from botorch.models.gp_regression import SingleTaskGP
        >>> from botorch.models.transforms.outcome import Standardize
        >>>
        >>> train_X = torch.rand(20, 2, dtype=torch.float64)
        >>> train_Y = torch.sin(train_X).sum(dim=1, keepdim=True)
        >>> outcome_transform = Standardize(m=1)
        >>> inferred_noise_model = SingleTaskGP(
        ...     train_X, train_Y, outcome_transform=outcome_transform,
        ... )

        Model with a known observation variance of 0.2:

        >>> train_Yvar = torch.full_like(train_Y, 0.2)
        >>> observed_noise_model = SingleTaskGP(
        ...     train_X, train_Y, train_Yvar,
        ...     outcome_transform=outcome_transform,
        ... )

        With noise-free observations:

        >>> train_Yvar = torch.full_like(train_Y, 1e-6)
        >>> noise_free_model = SingleTaskGP(
        ...     train_X, train_Y, train_Yvar,
        ...     outcome_transform=outcome_transform,
        ... )
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Optional[Tensor] = None,
        likelihood: Optional[Likelihood] = None,
        covar_module: Optional[Module] = None,
        mean_module: Optional[Mean] = None,
        outcome_transform: Optional[OutcomeTransform] = None,
        input_transform: Optional[InputTransform] = None,
    ) -> None:
        r"""
        Args:
            train_X: A `batch_shape x n x d` tensor of training features.
            train_Y: A `batch_shape x n x m` tensor of training observations.
            train_Yvar: An optional `batch_shape x n x m` tensor of observed
                measurement noise.
            likelihood: A likelihood. If omitted, use a standard
                `GaussianLikelihood` with inferred noise level if `train_Yvar`
                is None, and a `FixedNoiseGaussianLikelihood` with the given
                noise observations if `train_Yvar` is not None.
            covar_module: The module computing the covariance (Kernel) matrix.
                If omitted, use a `MaternKernel`.
            mean_module: The mean function to be used. If omitted, use a
                `ConstantMean`.
            outcome_transform: An outcome transform that is applied to the
                training data during instantiation and to the posterior during
                inference (that is, the `Posterior` obtained by calling
                `.posterior` on the model will be on the original scale).
            input_transform: An input transform that is applied in the model's
                forward pass.
        """
        with torch.no_grad():
            transformed_X = self.transform_inputs(
                X=train_X, input_transform=input_transform
            )
        if outcome_transform is not None:
            train_Y, train_Yvar = outcome_transform(train_Y, train_Yvar)
        self._validate_tensor_args(X=transformed_X, Y=train_Y, Yvar=train_Yvar)
        ignore_X_dims = getattr(self, "_ignore_X_dims_scaling_check", None)
        validate_input_scaling(
            train_X=transformed_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            ignore_X_dims=ignore_X_dims,
        )
        self._set_dimensions(train_X=train_X, train_Y=train_Y)
        train_X, train_Y, train_Yvar = self._transform_tensor_args(
            X=train_X, Y=train_Y, Yvar=train_Yvar
        )
        if likelihood is None:
            if train_Yvar is None:
                likelihood = get_gaussian_likelihood_with_gamma_prior(
                    batch_shape=self._aug_batch_shape
                )
            else:
                likelihood = FixedNoiseGaussianLikelihood(
                    noise=train_Yvar, batch_shape=self._aug_batch_shape
                )
        else:
            self._is_custom_likelihood = True
        ExactGP.__init__(
            self, train_inputs=train_X, train_targets=train_Y, likelihood=likelihood
        )
        if mean_module is None:
            mean_module = ConstantMean(batch_shape=self._aug_batch_shape)
        self.mean_module = mean_module
        if covar_module is None:
            covar_module = get_matern_kernel_with_gamma_prior(
                ard_num_dims=transformed_X.shape[-1],
                batch_shape=self._aug_batch_shape,
            )
            self._subset_batch_dict = {
                "mean_module.raw_constant": -1,
                "covar_module.raw_outputscale": -1,
                "covar_module.base_kernel.raw_lengthscale": -3,
            }
            if train_Yvar is None:
                self._subset_batch_dict["likelihood.noise_covar.raw_noise"] = -2
        self.covar_module = covar_module
        # TODO: Allow subsetting of other covar modules
        if outcome_transform is not None:
            self.outcome_transform = outcome_transform
        if input_transform is not None:
            self.input_transform = input_transform
        self.to(train_X)

    def forward(self, x: Tensor) -> MultivariateNormal:
        if self.training:
            x = self.transform_inputs(x)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class FixedNoiseGP(SingleTaskGP):
    r"""A single-task exact GP model using fixed noise levels.

    DEPRECATED: `FixedNoiseGP` has been merged into `SingleTaskGP`. Please use
    `SingleTaskGP` with `train_Yvar` instead.
    Will be removed in a future release (~v0.12).
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor,
        covar_module: Optional[Module] = None,
        mean_module: Optional[Mean] = None,
        outcome_transform: Optional[OutcomeTransform] = None,
        input_transform: Optional[InputTransform] = None,
    ) -> None:
        r"""DEPRECATED. See SingleTaskGP."""
        warnings.warn(
            "`FixedNoiseGP` has been merged into `SingleTaskGP`. "
            "Please use `SingleTaskGP` with `train_Yvar` instead.",
            DeprecationWarning,
        )
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            covar_module=covar_module,
            mean_module=mean_module,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
        )


class HeteroskedasticSingleTaskGP(BatchedMultiOutputGPyTorchModel, ExactGP):
    r"""A single-task exact GP model using a heteroskedastic noise model.

    This model differs from `SingleTaskGP` in that noise levels are provided
    rather than inferred, and differs from `FixedNoiseGP` in that it can
    predict noise levels out of sample, because it internally wraps another
    GP (a SingleTaskGP) to model the observation noise.
    Noise levels must be provided to `HeteroskedasticSingleTaskGP` as `train_Yvar`.

    Examples of cases in which noise levels are known include online
    experimentation and simulation optimization.

    Example:
        >>> train_X = torch.rand(20, 2)
        >>> train_Y = torch.sin(train_X).sum(dim=1, keepdim=True)
        >>> se = torch.linalg.norm(train_X, dim=1, keepdim=True)
        >>> train_Yvar = 0.1 + se * torch.rand_like(train_Y)
        >>> model = HeteroskedasticSingleTaskGP(train_X, train_Y, train_Yvar)
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor,
        outcome_transform: Optional[OutcomeTransform] = None,
        input_transform: Optional[InputTransform] = None,
    ) -> None:
        r"""
        Args:
            train_X: A `batch_shape x n x d` tensor of training features.
            train_Y: A `batch_shape x n x m` tensor of training observations.
            train_Yvar: A `batch_shape x n x m` tensor of observed measurement
                noise.
            outcome_transform: An outcome transform that is applied to the
                training data during instantiation and to the posterior during
                inference (that is, the `Posterior` obtained by calling
                `.posterior` on the model will be on the original scale).
                Note that the noise model internally log-transforms the
                variances, which will happen after this transform is applied.
            input_transform: An input transfrom that is applied in the model's
                forward pass.
        """
        if outcome_transform is not None:
            train_Y, train_Yvar = outcome_transform(train_Y, train_Yvar)
        self._validate_tensor_args(X=train_X, Y=train_Y, Yvar=train_Yvar)
        validate_input_scaling(train_X=train_X, train_Y=train_Y, train_Yvar=train_Yvar)
        self._set_dimensions(train_X=train_X, train_Y=train_Y)
        noise_likelihood = GaussianLikelihood(
            noise_prior=SmoothedBoxPrior(-3, 5, 0.5, transform=torch.log),
            batch_shape=self._aug_batch_shape,
            noise_constraint=GreaterThan(
                MIN_INFERRED_NOISE_LEVEL, transform=None, initial_value=1.0
            ),
        )
        noise_model = SingleTaskGP(
            train_X=train_X,
            train_Y=train_Yvar,
            likelihood=noise_likelihood,
            outcome_transform=Log(),
            input_transform=input_transform,
        )
        likelihood = _GaussianLikelihoodBase(HeteroskedasticNoise(noise_model))
        # This is hacky -- this class used to inherit from SingleTaskGP, but it
        # shouldn't so this is a quick fix to enable getting rid of that
        # inheritance
        SingleTaskGP.__init__(
            # pyre-fixme[6]: Incompatible parameter type
            self,
            train_X=train_X,
            train_Y=train_Y,
            likelihood=likelihood,
            input_transform=input_transform,
        )
        self.register_added_loss_term("noise_added_loss")
        self.update_added_loss_term(
            "noise_added_loss", NoiseModelAddedLossTerm(noise_model)
        )
        if outcome_transform is not None:
            self.outcome_transform = outcome_transform
        self.to(train_X)

    # pyre-fixme[15]: Inconsistent override
    def condition_on_observations(self, *_, **__) -> NoReturn:
        raise NotImplementedError

    # pyre-fixme[15]: Inconsistent override
    def subset_output(self, idcs) -> NoReturn:
        raise NotImplementedError

    def forward(self, x: Tensor) -> MultivariateNormal:
        if self.training:
            x = self.transform_inputs(x)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
