#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import replace
from itertools import product
from unittest.mock import MagicMock

import torch
from botorch import models
from botorch.models import ModelListGP, SingleTaskGP, SingleTaskVariationalGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.sampling.pathwise import (
    draw_kernel_feature_paths,
    GeneralizedLinearPath,
    PathList,
)
from botorch.sampling.pathwise.utils import get_train_inputs, is_finite_dimensional
from botorch.utils.test_helpers import get_sample_moments, standardize_moments
from botorch.utils.testing import BotorchTestCase
from gpytorch.distributions import MultitaskMultivariateNormal
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from torch import Size
from torch.nn.functional import pad

from .helpers import gen_module, gen_random_inputs, TestCaseConfig


class TestPriorSamplers(BotorchTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.models = defaultdict(list)
        self.num_features = 1024

        seed = 0
        for kernel in (
            MaternKernel(nu=2.5, ard_num_dims=2, batch_shape=Size([])),
            ScaleKernel(RBFKernel(ard_num_dims=2, batch_shape=Size([2]))),
        ):
            with torch.random.fork_rng():
                torch.manual_seed(seed)
                tkwargs = {"device": self.device, "dtype": torch.float64}

                base = kernel.base_kernel if isinstance(kernel, ScaleKernel) else kernel
                base.lengthscale = 0.1 + 0.3 * torch.rand_like(base.lengthscale)
                kernel.to(**tkwargs)

                uppers = 1 + 9 * torch.rand(base.lengthscale.shape[-1], **tkwargs)
                bounds = pad(uppers.unsqueeze(0), (0, 0, 1, 0))

                X = uppers * torch.rand(4, base.lengthscale.shape[-1], **tkwargs)
                Y = 10 * kernel(X).cholesky() @ torch.randn(4, 1, **tkwargs)
                if kernel.batch_shape:
                    Y = Y.squeeze(-1).transpose(0, 1)  # n x m

                input_transform = Normalize(d=X.shape[-1], bounds=bounds)
                outcome_transform = Standardize(m=Y.shape[-1])

                # SingleTaskGP w/ inferred noise in eval mode
                self.models["inferred"].append(
                    SingleTaskGP(
                        train_X=X,
                        train_Y=Y,
                        covar_module=deepcopy(kernel),
                        input_transform=deepcopy(input_transform),
                        outcome_transform=deepcopy(outcome_transform),
                    )
                    .to(**tkwargs)
                    .eval()
                )

                # SingleTaskGP w/ observed noise in train mode
                self.models["observed"].append(
                    SingleTaskGP(
                        train_X=X,
                        train_Y=Y,
                        train_Yvar=0.01 * torch.rand_like(Y),
                        covar_module=kernel,
                        input_transform=input_transform,
                        outcome_transform=outcome_transform,
                    ).to(**tkwargs)
                )

                # SingleTaskVariationalGP in train mode
                # When batched, uses a multitask format which break the tests below
                if not kernel.batch_shape:
                    self.models["variational"].append(
                        SingleTaskVariationalGP(
                            train_X=X,
                            train_Y=Y,
                            covar_module=kernel,
                            input_transform=input_transform,
                            outcome_transform=outcome_transform,
                        ).to(**tkwargs)
                    )

            seed += 1

    def test_draw_kernel_feature_paths(self):
        for seed, model_group in enumerate(self.models.values()):
            for model, sample_shape in product(
                model_group, [Size([1024]), Size([2, 512])]
            ):
                with torch.random.fork_rng():
                    torch.random.manual_seed(seed)
                    paths = draw_kernel_feature_paths(
                        model=model,
                        sample_shape=sample_shape,
                        num_features=self.num_features,
                    )
                    self.assertIsInstance(paths, GeneralizedLinearPath)
                    self._test_draw_kernel_feature_paths(model, paths, sample_shape)

        with self.subTest("test_model_list"):
            model_list = ModelListGP(
                self.models["inferred"][0], self.models["observed"][0]
            )
            path_list = draw_kernel_feature_paths(
                model=model_list,
                sample_shape=sample_shape,
                num_features=self.num_features,
            )
            (train_X,) = get_train_inputs(model_list.models[0], transformed=False)
            X = torch.zeros(
                4, train_X.shape[-1], dtype=train_X.dtype, device=self.device
            )
            sample_list = path_list(X)
            self.assertIsInstance(path_list, PathList)
            self.assertIsInstance(sample_list, list)
            self.assertEqual(len(sample_list), len(path_list._paths_list))

        with self.subTest("test_initialization"):
            model = self.models["inferred"][0]
            sample_shape = torch.Size([16])
            expected_weight_shape = (
                sample_shape + model.covar_module.batch_shape + (self.num_features,)
            )
            weight_generator = MagicMock(
                side_effect=lambda _: torch.rand(expected_weight_shape)
            )
            draw_kernel_feature_paths(
                model=model,
                sample_shape=sample_shape,
                num_features=self.num_features,
                weight_generator=weight_generator,
            )
            weight_generator.assert_called_once_with(expected_weight_shape)

    def _test_draw_kernel_feature_paths(self, model, paths, sample_shape, atol=3):
        (train_X,) = get_train_inputs(model, transformed=False)
        X = torch.rand(16, train_X.shape[-1], dtype=train_X.dtype, device=self.device)

        # Evaluate sample paths
        samples = paths(X)
        batch_shape = (
            model.model.covar_module.batch_shape
            if isinstance(model, SingleTaskVariationalGP)
            else model.covar_module.batch_shape
        )
        self.assertEqual(samples.shape, sample_shape + batch_shape + X.shape[-2:-1])

        # Calculate sample statistics
        sample_moments = get_sample_moments(samples, sample_shape)
        if hasattr(model, "outcome_transform"):
            # Do this instead of untransforming exact moments
            sample_moments = standardize_moments(
                model.outcome_transform, *sample_moments
            )

        # Compute prior distribution
        prior = model.forward(X if model.training else model.input_transform(X))
        exact_moments = (prior.loc, prior.covariance_matrix)

        # Compare moments
        tol = atol * (paths.weight.shape[-1] ** -0.5 + sample_shape.numel() ** -0.5)
        for exact, estimate in zip(exact_moments, sample_moments):
            self.assertTrue(exact.allclose(estimate, atol=tol, rtol=0))


# TestDrawKernelFeaturePaths: Tests for kernel feature path sampling
# - Tests both single-task and multi-task models
# - Verifies correct shape handling and covariance matching
# - Checks path list operations for model lists
class TestDrawKernelFeaturePaths(BotorchTestCase):
    def setUp(self) -> None:
        """Set up test cases with various model types and configurations.
        - Creates single-task, multi-task, and variational models
        - Sets up model lists for testing path combinations
        - Configures batch shapes and dimensions
        """
        super().setUp()
        config = TestCaseConfig(seed=0, device=self.device)
        batch_config = replace(config, batch_shape=Size([2]))

        # Create test models with different configurations
        self.base_models = [
            (batch_config, gen_module(models.SingleTaskGP, batch_config)),
            (batch_config, gen_module(models.MultiTaskGP, batch_config)),
            (config, gen_module(models.SingleTaskVariationalGP, config)),
        ]
        self.model_lists = [
            (batch_config, gen_module(models.ModelListGP, batch_config))
        ]

    def test_base_models(self, slack: float = 3.0):
        """Test kernel feature path sampling for base models.
        - Verifies correct output shapes and dimensions
        - Checks covariance matrix matching
        - Handles both transformed and untransformed inputs
        - Tests multi-task model task feature handling
        """
        sample_shape = Size([32, 32])
        for config, model in self.base_models:
            kernel = (
                model.model.covar_module
                if isinstance(model, models.SingleTaskVariationalGP)
                else model.covar_module
            )
            with torch.random.fork_rng():
                torch.random.manual_seed(config.seed)
                paths = draw_kernel_feature_paths(
                    model=model,
                    sample_shape=sample_shape,
                    num_random_features=config.num_random_features,
                )
                self.assertIsInstance(paths, GeneralizedLinearPath)
                n = 16
                X = gen_random_inputs(model, batch_shape=[n], transformed=False)

            # Get prior distribution and check shapes
            prior = model.forward(X if model.training else model.input_transform(X))
            if isinstance(prior, MultitaskMultivariateNormal):
                num_tasks = kernel.batch_shape[0]
                exact_mean = prior.mean.view(num_tasks, n)
                exact_covar = prior.covariance_matrix.view(num_tasks, n, num_tasks, n)
                exact_covar = torch.stack(
                    [exact_covar[..., i, :, i, :] for i in range(num_tasks)], dim=-3
                )
            else:
                exact_mean = prior.loc
                exact_covar = prior.covariance_matrix

            # Normalize by standard deviations for comparison
            istd = exact_covar.diagonal(dim1=-2, dim2=-1).rsqrt()
            exact_mean = istd * exact_mean
            exact_covar = istd.unsqueeze(-1) * exact_covar * istd.unsqueeze(-2)

            # Sample paths and transform outputs
            samples = paths(X)
            if hasattr(model, "outcome_transform"):
                model.outcome_transform.train(mode=False)
                if kernel.batch_shape:
                    samples, _ = model.outcome_transform(samples.transpose(-2, -1))
                    samples = samples.transpose(-2, -1)
                else:
                    samples, _ = model.outcome_transform(samples.unsqueeze(-1))
                    samples = samples.squeeze(-1)
                model.outcome_transform.train(mode=model.training)

            # Compute sample statistics
            samples = istd * samples.view(-1, *samples.shape[len(sample_shape) :])
            sample_mean = samples.mean(dim=0)
            sample_covar = (samples - sample_mean).permute(*range(1, samples.ndim), 0)
            sample_covar = torch.divide(
                sample_covar @ sample_covar.transpose(-2, -1), sample_shape.numel()
            )

            # Set tolerance based on number of features
            allclose_kwargs = {"atol": slack * sample_shape.numel() ** -0.5}
            if not is_finite_dimensional(kernel):
                num_random_features_per_map = config.num_random_features / (
                    1
                    if not is_finite_dimensional(kernel, max_depth=0)
                    else sum(
                        not is_finite_dimensional(k)
                        for k in kernel.modules()
                        if k is not kernel
                    )
                )
                allclose_kwargs["atol"] += slack * num_random_features_per_map**-0.5

            # Verify mean and covariance matching
            self.assertTrue(exact_mean.allclose(sample_mean, **allclose_kwargs))
            self.assertTrue(exact_covar.allclose(sample_covar, **allclose_kwargs))

    def test_model_lists(self):
        """Test kernel feature path sampling for model lists.
        - Verifies path list creation and handling
        - Checks individual model path sampling
        - Tests path combination operations
        """
        sample_shape = Size([32, 32])
        for config, model_list in self.model_lists:
            with torch.random.fork_rng():
                torch.random.manual_seed(config.seed)
                path_list = draw_kernel_feature_paths(
                    model=model_list,
                    sample_shape=sample_shape,
                    num_random_features=config.num_random_features,
                )
                self.assertIsInstance(path_list, PathList)

                X = gen_random_inputs(model_list.models[0], batch_shape=[4])
                sample_list = path_list(X)
                self.assertIsInstance(sample_list, list)
                self.assertEqual(len(sample_list), len(model_list.models))
                for path, sample in zip(path_list, sample_list):
                    self.assertTrue(path(X).equal(sample))
