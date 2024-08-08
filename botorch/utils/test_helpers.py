#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Dummy classes and other helpers that are used in multiple test files
should be defined here to avoid relative imports.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from botorch.acquisition.objective import PosteriorTransform
from botorch.exceptions.errors import UnsupportedError
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.model import FantasizeMixin, Model
from botorch.models.transforms.outcome import Standardize
from botorch.models.utils import add_output_dim
from botorch.models.utils.assorted import fantasize
from botorch.posteriors.posterior import Posterior
from botorch.utils.datasets import MultiTaskDataset, SupervisedDataset
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods.gaussian_likelihood import (
    FixedNoiseGaussianLikelihood,
    GaussianLikelihood,
)
from gpytorch.means import ConstantMean
from gpytorch.models.exact_gp import ExactGP
from torch import Size, Tensor
from torch.nn.functional import pad


def get_sample_moments(samples: Tensor, sample_shape: Size) -> tuple[Tensor, Tensor]:
    """Computes the mean and covariance of a set of samples.

    Args:
        samples: A tensor of shape `sample_shape x batch_shape x q`.
        sample_shape: The sample_shape input used while generating the samples using
            the pathwise sampling API.
    """
    sample_dim = len(sample_shape)
    samples = samples.view(-1, *samples.shape[sample_dim:])
    loc = samples.mean(dim=0)
    residuals = (samples - loc).permute(*range(1, samples.ndim), 0)
    return loc, (residuals @ residuals.transpose(-2, -1)) / sample_shape.numel()


def standardize_moments(
    transform: Standardize,
    loc: Tensor,
    covariance_matrix: Tensor,
) -> tuple[Tensor, Tensor]:
    """Standardizes the loc and covariance_matrix using the mean and standard
    deviations from a Standardize transform.
    """
    m = transform.means.squeeze().unsqueeze(-1)
    s = transform.stdvs.squeeze().reciprocal().unsqueeze(-1)
    loc = s * (loc - m)
    correlation_matrix = s.unsqueeze(-1) * covariance_matrix * s.unsqueeze(-2)
    return loc, correlation_matrix


def gen_multi_task_dataset(
    yvar: Optional[float] = None,
    task_values: Optional[list[int]] = None,
    skip_task_features_in_datasets: bool = False,
    **tkwargs,
) -> tuple[MultiTaskDataset, tuple[Tensor, Tensor, Optional[Tensor]]]:
    """Constructs a multi-task dataset with two tasks, each with 10 data points.

    Args:
        yvar: The noise level to use for `train_Yvar`. If None, uses `train_Yvar=None`.
        task_values: The values of the task features. If None, uses [0, 1].
        skip_task_features_in_datasets: If True, the task features are not included in
            Xs of the datasets used to construct the datasets. This is useful for
            testing `MultiTaskDataset`.
    """
    if task_values is not None and skip_task_features_in_datasets:
        raise UnsupportedError(  # pragma: no cover
            "`task_values` and `skip_task_features_in_datasets` can't be used together."
        )
    X = torch.linspace(0, 0.95, 10, **tkwargs) + 0.05 * torch.rand(10, **tkwargs)
    X = X.unsqueeze(dim=-1)
    Y1 = torch.sin(X * (2 * math.pi)) + torch.randn_like(X) * 0.2
    Y2 = torch.cos(X * (2 * math.pi)) + torch.randn_like(X) * 0.2
    if task_values is None:
        task_values = [0, 1]
    train_X = torch.cat([pad(X, (1, 0), value=i) for i in task_values])
    train_Y = torch.cat([Y1, Y2])

    Yvar1 = None if yvar is None else torch.full_like(Y1, yvar)
    Yvar2 = None if yvar is None else torch.full_like(Y2, 2 * yvar)
    train_Yvar = None if yvar is None else torch.cat([Yvar1, Yvar2])
    Y3 = torch.tan(X * (2 * math.pi)) + torch.randn_like(X) * 0.2
    Yvar3 = None if yvar is None else torch.full_like(Y3, yvar)
    if len(task_values) == 3:
        train_Y = torch.cat([train_Y, Y3])
        if train_Yvar is not None:
            train_Yvar = torch.cat([train_Yvar, Yvar3])
    feature_slice = slice(1, None) if skip_task_features_in_datasets else slice(None)
    datasets = [
        SupervisedDataset(
            X=train_X[:10, feature_slice],
            Y=Y1,
            Yvar=Yvar1,
            feature_names=["task", "X"][feature_slice],
            outcome_names=["y"],
        ),
        SupervisedDataset(
            X=train_X[10:20, feature_slice],
            Y=Y2,
            Yvar=Yvar2,
            feature_names=["task", "X"][feature_slice],
            outcome_names=["y1"],
        ),
    ]
    if len(task_values) == 3:
        datasets.append(
            SupervisedDataset(
                X=train_X[20:, feature_slice],
                Y=Y3,
                Yvar=Yvar3,
                feature_names=["task", "X"][feature_slice],
                outcome_names=["y2"],
            )
        )
    dataset = MultiTaskDataset(
        datasets=datasets,
        target_outcome_name="y",
        task_feature_index=None if skip_task_features_in_datasets else 0,
    )
    return dataset, (train_X, train_Y, train_Yvar)


def get_pvar_expected(posterior: Posterior, model: Model, X: Tensor, m: int) -> Tensor:
    """Computes the expected variance of a posterior after adding the
    predictive noise from the likelihood.
    """
    X = model.transform_inputs(X)
    lh_kwargs = {}
    if isinstance(model.likelihood, FixedNoiseGaussianLikelihood):
        lh_kwargs["noise"] = model.likelihood.noise.mean().expand(X.shape[:-1])
    if m == 1:
        return model.likelihood(
            posterior.distribution, X, **lh_kwargs
        ).variance.unsqueeze(-1)
    X_, odi = add_output_dim(X=X, original_batch_shape=model._input_batch_shape)
    pvar_exp = model.likelihood(model(X_), X_, **lh_kwargs).variance
    return torch.stack([pvar_exp.select(dim=odi, index=i) for i in range(m)], dim=-1)


class DummyNonScalarizingPosteriorTransform(PosteriorTransform):
    scalarize = False

    def evaluate(self, Y):
        pass  # pragma: no cover

    def forward(self, posterior):
        pass  # pragma: no cover


class SimpleGPyTorchModel(GPyTorchModel, ExactGP, FantasizeMixin):
    last_fantasize_flag: bool = False

    def __init__(self, train_X, train_Y, outcome_transform=None, input_transform=None):
        r"""
        Args:
            train_X: A tensor of inputs, passed to self.transform_inputs.
            train_Y: Passed to outcome_transform.
            outcome_transform: Transform applied to train_Y.
            input_transform: A Module that performs the input transformation, passed to
                self.transform_inputs.
        """
        with torch.no_grad():
            transformed_X = self.transform_inputs(
                X=train_X, input_transform=input_transform
            )
        if outcome_transform is not None:
            train_Y, _ = outcome_transform(train_Y)
        self._validate_tensor_args(transformed_X, train_Y)
        train_Y = train_Y.squeeze(-1)
        likelihood = GaussianLikelihood()
        super().__init__(train_X, train_Y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel())
        if outcome_transform is not None:
            self.outcome_transform = outcome_transform
        if input_transform is not None:
            self.input_transform = input_transform
        self._num_outputs = 1
        self.to(train_X)
        self.transformed_call_args = []

    def forward(self, x):
        self.last_fantasize_flag = fantasize.on()
        if self.training:
            x = self.transform_inputs(x)
        self.transformed_call_args.append(x)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
