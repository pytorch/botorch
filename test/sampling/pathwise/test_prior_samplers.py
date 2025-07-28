#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import replace

import torch
from botorch import models
from botorch.sampling.pathwise import (
    draw_kernel_feature_paths,
    GeneralizedLinearPath,
    PathList,
)
from botorch.sampling.pathwise.utils import is_finite_dimensional
from botorch.utils.testing import BotorchTestCase
from gpytorch.distributions import MultitaskMultivariateNormal
from torch import Size

from .helpers import gen_module, gen_random_inputs, TestCaseConfig


class TestDrawKernelFeaturePaths(BotorchTestCase):
    def setUp(self) -> None:
        super().setUp()
        config = TestCaseConfig(seed=0, device=self.device)
        batch_config = replace(config, batch_shape=Size([2]))

        self.base_models = [
            (batch_config, gen_module(models.SingleTaskGP, batch_config)),
            (batch_config, gen_module("FixedNoiseGP", batch_config)),
            (batch_config, gen_module(models.MultiTaskGP, batch_config)),
            (config, gen_module(models.SingleTaskVariationalGP, config)),
        ]
        self.model_lists = [
            (batch_config, gen_module(models.ModelListGP, batch_config))
        ]

    def test_base_models(self, slack: float = 3.0):
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

            istd = exact_covar.diagonal(dim1=-2, dim2=-1).rsqrt()
            exact_mean = istd * exact_mean
            exact_covar = istd.unsqueeze(-1) * exact_covar * istd.unsqueeze(-2)

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

            samples = istd * samples.view(-1, *samples.shape[len(sample_shape) :])
            sample_mean = samples.mean(dim=0)
            sample_covar = (samples - sample_mean).permute(*range(1, samples.ndim), 0)
            sample_covar = torch.divide(
                sample_covar @ sample_covar.transpose(-2, -1), sample_shape.numel()
            )

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
            self.assertTrue(exact_mean.allclose(sample_mean, **allclose_kwargs))
            self.assertTrue(exact_covar.allclose(sample_covar, **allclose_kwargs))

    def test_model_lists(self):
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

    def test_weight_generator_custom(self):
        """Test custom weight generator in prior_samplers.py"""
        import torch
        from botorch.sampling.pathwise.prior_samplers import (
            _draw_kernel_feature_paths_fallback,
        )
        from gpytorch.kernels import RBFKernel

        # Create kernel with ard_num_dims to avoid num_ambient_inputs issue
        kernel = RBFKernel(ard_num_dims=2)
        sample_shape = torch.Size([2, 3])

        # Custom weight generator
        def custom_weight_generator(weight_shape):
            return torch.ones(weight_shape)

        result = _draw_kernel_feature_paths_fallback(
            mean_module=None,
            covar_module=kernel,
            sample_shape=sample_shape,
            weight_generator=custom_weight_generator,
        )

        # Verify the result
        self.assertIsNotNone(result.weight)
        # Weight should be all ones (from our custom generator)
        self.assertTrue(torch.allclose(result.weight, torch.ones_like(result.weight)))
