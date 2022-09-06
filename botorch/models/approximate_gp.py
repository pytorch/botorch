#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
References

.. [burt2020svgp]
    David R. Burt and Carl Edward Rasmussen and Mark van der Wilk,
    Convergence of Sparse Variational Inference in Gaussian Process Regression,
    Journal of Machine Learning Research, 2020,
    http://jmlr.org/papers/v21/19-1015.html.

.. [chen2018dpp]
    Laming Chen and Guoxin Zhang and Hanning Zhou, Fast greedy MAP inference
    for determinantal point process to improve recommendation diversity,
    Proceedings of the 32nd International Conference on Neural Information
    Processing Systems, 2018, https://arxiv.org/abs/1709.05135.

.. [hensman2013svgp]
    James Hensman and Nicolo Fusi and Neil D. Lawrence, Gaussian Processes
    for Big Data, Proceedings of the 29th Conference on Uncertainty in
    Artificial Intelligence, 2013, https://arxiv.org/abs/1309.6835.

"""

from __future__ import annotations

import copy
from typing import Optional, Type, Union

import torch
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.models.utils import validate_input_scaling
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.sampling import MCSampler
from gpytorch.constraints import GreaterThan
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import Kernel, MaternKernel, ScaleKernel
from gpytorch.likelihoods import (
    GaussianLikelihood,
    Likelihood,
    MultitaskGaussianLikelihood,
)
from gpytorch.means import ConstantMean, Mean
from gpytorch.models import ApproximateGP
from gpytorch.module import Module
from gpytorch.priors import GammaPrior
from gpytorch.utils.memoize import clear_cache_hook
from gpytorch.variational import (
    _VariationalDistribution,
    _VariationalStrategy,
    CholeskyVariationalDistribution,
    IndependentMultitaskVariationalStrategy,
    VariationalStrategy,
)
from linear_operator.operators import LinearOperator
from torch import Tensor


MIN_INFERRED_NOISE_LEVEL = 1e-4
NEG_INF = -(torch.tensor(float("inf")))


class ApproximateGPyTorchModel(GPyTorchModel):
    r"""
    Botorch wrapper class for various (variational) approximate GP models in
    GPyTorch.

    This can either include stochastic variational GPs (SVGPs) or
    variational implementations of weight space approximate GPs.
    """

    def __init__(
        self,
        model: Optional[ApproximateGP] = None,
        likelihood: Optional[Likelihood] = None,
        num_outputs: int = 1,
        *args,
        **kwargs,
    ) -> None:
        r"""
        Args:
            model: Instance of gpytorch.approximate GP models. If omitted,
                constructs a `_SingleTaskVariationalGP`.
            likelihood: Instance of a GPyTorch likelihood. If omitted, uses a
                either a `GaussianLikelihood` (if `num_outputs=1`) or a
                `MultitaskGaussianLikelihood`(if `num_outputs>1`).
            num_outputs: Number of outputs expected for the GP model.
            args: Optional positional arguments passed to the
                `_SingleTaskVariationalGP` constructor if no model is provided.
            kwargs: Optional keyword arguments passed to the
                `_SingleTaskVariationalGP` constructor if no model is provided.
        """
        super().__init__()

        if model is None:
            model = _SingleTaskVariationalGP(num_outputs=num_outputs, *args, **kwargs)

        if likelihood is None:
            if num_outputs == 1:
                likelihood = GaussianLikelihood()
            else:
                likelihood = MultitaskGaussianLikelihood(num_tasks=num_outputs)

        self.model = model
        self.likelihood = likelihood
        self._desired_num_outputs = num_outputs

    @property
    def num_outputs(self):
        return self._desired_num_outputs

    def posterior(
        self, X, output_indices=None, observation_noise=False, *args, **kwargs
    ) -> GPyTorchPosterior:
        self.eval()  # make sure model is in eval mode

        # input transforms are applied at `posterior` in `eval` mode, and at
        # `model.forward()` at the training time
        X = self.transform_inputs(X)

        # check for the multi-batch case for multi-outputs b/c this will throw
        # warnings
        X_ndim = X.ndim
        if self.num_outputs > 1 and X_ndim > 2:
            X = X.unsqueeze(-3).repeat(*[1] * (X_ndim - 2), self.num_outputs, 1, 1)
        dist = self.model(X)
        if observation_noise:
            dist = self.likelihood(dist, *args, **kwargs)

        posterior = GPyTorchPosterior(mvn=dist)
        if hasattr(self, "outcome_transform"):
            posterior = self.outcome_transform.untransform_posterior(posterior)
        return posterior

    def forward(self, X, *args, **kwargs) -> MultivariateNormal:
        if self.training:
            X = self.transform_inputs(X)
        return self.model(X)

    def fantasize(self, X, sampler=MCSampler, observation_noise=True, *args, **kwargs):
        raise NotImplementedError(
            "Fantasization of approximate GPs has not been implemented yet."
        )


class _SingleTaskVariationalGP(ApproximateGP):
    """
    Base class wrapper for a stochastic variational Gaussian Process (SVGP)
    model [hensman2013svgp]_.

    Uses pivoted Cholesky initialization for the inducing points.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Optional[Tensor] = None,
        num_outputs: int = 1,
        learn_inducing_points=True,
        covar_module: Optional[Kernel] = None,
        mean_module: Optional[Mean] = None,
        variational_distribution: Optional[_VariationalDistribution] = None,
        variational_strategy: Type[_VariationalStrategy] = VariationalStrategy,
        inducing_points: Optional[Union[Tensor, int]] = None,
    ) -> None:
        r"""
        Args:
            train_X: Training inputs (due to the ability of the SVGP to sub-sample
                this does not have to be all of the training inputs).
            train_Y: Training targets (optional).
            num_outputs: Number of output responses per input.
            covar_module: Kernel function. If omitted, uses a `MaternKernel`.
            mean_module: Mean of GP model. If omitted, uses a `ConstantMean`.
            variational_distribution: Type of variational distribution to use
                (default: CholeskyVariationalDistribution), the properties of the
                variational distribution will encourage scalability or ease of
                optimization.
            variational_strategy: Type of variational strategy to use (default:
                VariationalStrategy). The default setting uses "whitening" of the
                variational distribution to make training easier.
            inducing_points: The number or specific locations of the inducing points.
        """
        # We use the model subclass wrapper to deal with input / outcome transforms.
        # The number of outputs will be correct here due to the check in
        # SingleTaskVariationalGP.
        input_batch_shape = train_X.shape[:-2]
        aug_batch_shape = copy.deepcopy(input_batch_shape)
        if num_outputs > 1:
            aug_batch_shape += torch.Size((num_outputs,))
        self._aug_batch_shape = aug_batch_shape

        if mean_module is None:
            mean_module = ConstantMean(batch_shape=self._aug_batch_shape).to(train_X)

        if covar_module is None:
            covar_module = ScaleKernel(
                base_kernel=MaternKernel(
                    nu=2.5,
                    ard_num_dims=train_X.shape[-1],
                    batch_shape=self._aug_batch_shape,
                    lengthscale_prior=GammaPrior(3.0, 6.0),
                ),
                batch_shape=self._aug_batch_shape,
                outputscale_prior=GammaPrior(2.0, 0.15),
            ).to(train_X)
            self._subset_batch_dict = {
                "mean_module.constant": -2,
                "covar_module.raw_outputscale": -1,
                "covar_module.base_kernel.raw_lengthscale": -3,
            }

        # initialize inducing points with a pivoted cholesky init if they are not given
        if not isinstance(inducing_points, Tensor):
            if inducing_points is None:
                # number of inducing points is 25% the number of data points
                # as a heuristic
                inducing_points = int(0.25 * train_X.shape[-2])

            inducing_points = _select_inducing_points(
                inputs=train_X,
                covar_module=covar_module,
                num_inducing=inducing_points,
                input_batch_shape=input_batch_shape,
            )

        if variational_distribution is None:
            variational_distribution = CholeskyVariationalDistribution(
                num_inducing_points=inducing_points.shape[-2],
                batch_shape=self._aug_batch_shape,
            )

        variational_strategy = variational_strategy(
            self,
            inducing_points=inducing_points,
            variational_distribution=variational_distribution,
            learn_inducing_locations=learn_inducing_points,
        )

        # wrap variational models in independent multi-task variational strategy
        if num_outputs > 1:
            variational_strategy = IndependentMultitaskVariationalStrategy(
                base_variational_strategy=variational_strategy,
                num_tasks=num_outputs,
                task_dim=-1,
            )
        super().__init__(variational_strategy=variational_strategy)
        self.mean_module = mean_module
        self.covar_module = covar_module

    def forward(self, X) -> MultivariateNormal:
        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X)
        latent_dist = MultivariateNormal(mean_x, covar_x)
        return latent_dist


class SingleTaskVariationalGP(ApproximateGPyTorchModel):
    r"""A single-task variational GP model following [hensman2013svgp]_ with pivoted
    Cholesky initialization following [chen2018dpp]_ and [burt2020svgp]_.

    A single-task variational GP using relatively strong priors on the Kernel
    hyperparameters, which work best when covariates are normalized to the unit
    cube and outcomes are standardized (zero mean, unit variance).

    This model works in batch mode (each batch having its own hyperparameters).
    When the training observations include multiple outputs, this model will use
    batching to model outputs independently. However, batches of multi-output models
    are not supported at this time, if you need to use those, please use a
    ModelListGP.

    Use this model if you have a lot of data or if your responses are non-Gaussian.

    To train this model, you should use gpytorch.mlls.VariationalELBO and not
    the exact marginal log likelihood.

    Example:
        >>> import torch
        >>> from botorch.models import SingleTaskVariationalGP
        >>> from gpytorch.mlls import VariationalELBO
        >>>
        >>> train_X = torch.rand(20, 2)
        >>> model = SingleTaskVariationalGP(train_X)
        >>> mll = VariationalELBO(
        >>>     model.likelihood, model.model, num_data=train_X.shape[-2]
        >>> )
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Optional[Tensor] = None,
        likelihood: Optional[Likelihood] = None,
        num_outputs: int = 1,
        learn_inducing_points: bool = True,
        covar_module: Optional[Kernel] = None,
        mean_module: Optional[Mean] = None,
        variational_distribution: Optional[_VariationalDistribution] = None,
        variational_strategy: Type[_VariationalStrategy] = VariationalStrategy,
        inducing_points: Optional[Union[Tensor, int]] = None,
        outcome_transform: Optional[OutcomeTransform] = None,
        input_transform: Optional[InputTransform] = None,
    ) -> None:
        r"""
        Args:
            train_X: Training inputs (due to the ability of the SVGP to sub-sample
                this does not have to be all of the training inputs).
            train_Y: Training targets (optional).
            likelihood: Instance of a GPyTorch likelihood. If omitted, uses a
                either a `GaussianLikelihood` (if `num_outputs=1`) or a
                `MultitaskGaussianLikelihood`(if `num_outputs>1`).
            num_outputs: Number of output responses per input (default: 1).
            covar_module: Kernel function. If omitted, uses a `MaternKernel`.
            mean_module: Mean of GP model. If omitted, uses a `ConstantMean`.
            variational_distribution: Type of variational distribution to use
                (default: CholeskyVariationalDistribution), the properties of the
                variational distribution will encourage scalability or ease of
                optimization.
            variational_strategy: Type of variational strategy to use (default:
                VariationalStrategy). The default setting uses "whitening" of the
                variational distribution to make training easier.
            inducing_points: The number or specific locations of the inducing points.
        """
        with torch.no_grad():
            transformed_X = self.transform_inputs(
                X=train_X, input_transform=input_transform
            )
        if train_Y is not None:
            if outcome_transform is not None:
                train_Y, _ = outcome_transform(train_Y)
            self._validate_tensor_args(X=transformed_X, Y=train_Y)
            validate_input_scaling(train_X=transformed_X, train_Y=train_Y)
            if train_Y.shape[-1] != num_outputs:
                num_outputs = train_Y.shape[-1]

        self._num_outputs = num_outputs
        self._input_batch_shape = train_X.shape[:-2]
        aug_batch_shape = copy.deepcopy(self._input_batch_shape)
        if num_outputs > 1:
            aug_batch_shape += torch.Size([num_outputs])
        self._aug_batch_shape = aug_batch_shape

        if likelihood is None:
            if num_outputs == 1:
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
                likelihood = MultitaskGaussianLikelihood(num_tasks=num_outputs)
        else:
            self._is_custom_likelihood = True

        model = _SingleTaskVariationalGP(
            train_X=transformed_X,
            train_Y=train_Y,
            num_outputs=num_outputs,
            learn_inducing_points=learn_inducing_points,
            covar_module=covar_module,
            mean_module=mean_module,
            variational_distribution=variational_distribution,
            variational_strategy=variational_strategy,
            inducing_points=inducing_points,
        )

        super().__init__(model=model, likelihood=likelihood, num_outputs=num_outputs)

        if outcome_transform is not None:
            self.outcome_transform = outcome_transform
        if input_transform is not None:
            self.input_transform = input_transform

        # for model fitting utilities
        # TODO: make this a flag?
        self.model.train_inputs = [transformed_X]
        if train_Y is not None:
            self.model.train_targets = train_Y.squeeze(-1)

        self.to(train_X)

    def init_inducing_points(
        self,
        inputs: Tensor,
    ) -> Tensor:
        r"""
        Reinitialize the inducing point locations in-place with the current kernel
        applied to `inputs`.
        The variational distribution and variational strategy caches are reset.

        Args:
            inputs: (\*batch_shape, n, d)-dim input data tensor.

        Returns:
            (\*batch_shape, m, d)-dim tensor of selected inducing point locations.
        """
        var_strat = self.model.variational_strategy
        clear_cache_hook(var_strat)
        if hasattr(var_strat, "base_variational_strategy"):
            var_strat = var_strat.base_variational_strategy
            clear_cache_hook(var_strat)

        with torch.no_grad():
            num_inducing = var_strat.inducing_points.size(-2)
            inducing_points = _select_inducing_points(
                inputs=inputs,
                covar_module=self.model.covar_module,
                num_inducing=num_inducing,
                input_batch_shape=self._input_batch_shape,
            )
            var_strat.inducing_points.copy_(inducing_points)
            var_strat.variational_params_initialized.fill_(0)

        return inducing_points


def _select_inducing_points(
    inputs: Tensor,
    covar_module: Module,
    num_inducing: int,
    input_batch_shape: torch.Size,
) -> Tensor:
    r"""
    Utility function that evaluates a kernel at given inputs and selects inducing point
    locations based on the pivoted Cholesky heuristic.

    Args:
        inputs: A (*batch_shape, n, d)-dim input data tensor.
        covar_module: GPyTorch Module returning a LinearOperator kernel matrix.
        num_inducing: The maximun number (m) of inducing points (m <= n).
        input_batch_shape: The non-task-related batch shape.

    Returns:
        A (*batch_shape, m, d)-dim tensor of inducing point locations.
    """

    train_train_kernel = covar_module(inputs).evaluate_kernel()

    # base case
    if train_train_kernel.ndimension() == 2:
        inducing_points = _pivoted_cholesky_init(
            train_inputs=inputs,
            kernel_matrix=train_train_kernel,
            max_length=num_inducing,
        )
    # multi-task case
    elif train_train_kernel.ndimension() == 3 and len(input_batch_shape) == 0:
        input_element = inputs[0] if inputs.ndimension() == 3 else inputs
        kernel_element = train_train_kernel[0]
        inducing_points = _pivoted_cholesky_init(
            train_inputs=input_element,
            kernel_matrix=kernel_element,
            max_length=num_inducing,
        )
    # batched input cases
    else:
        batched_inputs = (
            inputs.expand(*input_batch_shape, -1, -1)
            if inputs.ndimension() == 2
            else inputs
        )
        reshaped_inputs = batched_inputs.flatten(end_dim=-3)
        inducing_points = []
        for input_element in reshaped_inputs:
            # the extra kernel evals are a little wasteful but make it
            # easier to infer the task batch size
            kernel_element = covar_module(input_element).evaluate_kernel()
            # handle extra task batch dimension
            kernel_element = (
                kernel_element[0]
                if kernel_element.ndimension() == 3
                else kernel_element
            )
            inducing_points.append(
                _pivoted_cholesky_init(
                    train_inputs=input_element,
                    kernel_matrix=kernel_element,
                    max_length=num_inducing,
                )
            )
        inducing_points = torch.stack(inducing_points).view(
            *input_batch_shape, num_inducing, -1
        )

    return inducing_points


def _pivoted_cholesky_init(
    train_inputs: Tensor,
    kernel_matrix: Union[Tensor, LinearOperator],
    max_length: int,
    epsilon: float = 1e-6,
) -> Tensor:
    r"""
    A pivoted cholesky initialization method for the inducing points,
    originally proposed in [burt2020svgp]_ with the algorithm itself coming from
    [chen2018dpp]_. Code is a PyTorch version from [chen2018dpp]_, copied from
    https://github.com/laming-chen/fast-map-dpp/blob/master/dpp.py.

    Args:
        train_inputs: training inputs (of shape n x d)
        kernel_matrix: kernel matrix on the training
            inputs
        max_length: number of inducing points to initialize
        epsilon: numerical jitter for stability.

    Returns:
        max_length x d tensor of the training inputs corresponding to the top
        max_length pivots of the training kernel matrix
    """

    # this is numerically equivalent to iteratively performing a pivoted cholesky
    # while storing the diagonal pivots at each iteration
    # TODO: use gpytorch's pivoted cholesky instead once that gets an exposed list
    # TODO: ensure this works in batch mode, which it does not currently.

    item_size = kernel_matrix.shape[-2]
    cis = torch.zeros(
        (max_length, item_size), device=kernel_matrix.device, dtype=kernel_matrix.dtype
    )
    di2s = kernel_matrix.diag()
    selected_items = []
    selected_item = torch.argmax(di2s)
    selected_items.append(selected_item)

    while len(selected_items) < max_length:
        k = len(selected_items) - 1
        ci_optimal = cis[:k, selected_item]
        di_optimal = torch.sqrt(di2s[selected_item])
        elements = kernel_matrix[..., selected_item, :]
        eis = (elements - torch.matmul(ci_optimal, cis[:k, :])) / di_optimal
        cis[k, :] = eis
        di2s = di2s - eis.pow(2.0)
        di2s[selected_item] = NEG_INF
        selected_item = torch.argmax(di2s)
        if di2s[selected_item] < epsilon:
            break
        selected_items.append(selected_item)

    ind_points = train_inputs[torch.stack(selected_items)]

    return ind_points
