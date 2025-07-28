#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import replace
from functools import partial

import torch
from botorch import models
from botorch.models import SingleTaskVariationalGP
from botorch.sampling.pathwise import (
    draw_kernel_feature_paths,
    draw_matheron_paths,
    MatheronPath,
    PathList,
)
from botorch.sampling.pathwise.utils import is_finite_dimensional
from botorch.utils.context_managers import delattr_ctx
from botorch.utils.testing import BotorchTestCase
from gpytorch.distributions import MultitaskMultivariateNormal
from torch import Size

from .helpers import gen_module, gen_random_inputs, TestCaseConfig


class TestGetMatheronPathModel(BotorchTestCase):
    def test_get_matheron_path_model(self):
        from unittest.mock import patch

        from botorch.exceptions.errors import UnsupportedError
        from botorch.models.deterministic import GenericDeterministicModel
        from botorch.sampling.pathwise.posterior_samplers import get_matheron_path_model

        # Test single output model
        config = TestCaseConfig(seed=0, device=self.device)
        model = gen_module(models.SingleTaskGP, config)
        sample_shape = Size([3])

        path_model = get_matheron_path_model(model, sample_shape=sample_shape)
        self.assertIsInstance(path_model, GenericDeterministicModel)
        self.assertEqual(path_model.num_outputs, 1)
        self.assertTrue(path_model._is_ensemble)

        # Test evaluation
        X = torch.rand(4, config.num_inputs, device=self.device, dtype=config.dtype)
        output = path_model(X)
        self.assertEqual(output.shape, (3, 4, 1))  # sample_shape + batch + output

        # Test without sample_shape
        path_model = get_matheron_path_model(model)
        self.assertFalse(path_model._is_ensemble)
        output = path_model(X)
        self.assertEqual(output.shape, (4, 1))

        # Test ModelListGP
        batch_config = replace(config, batch_shape=Size([2]))
        model_list = gen_module(models.ModelListGP, batch_config)
        path_model = get_matheron_path_model(model_list)
        self.assertEqual(path_model.num_outputs, model_list.num_outputs)

        X = torch.rand(4, config.num_inputs, device=self.device, dtype=config.dtype)
        output = path_model(X)
        self.assertEqual(output.shape, (4, model_list.num_outputs))

        # Test generic ModelList (not ModelListGP)
        from botorch.models.model import ModelList

        # Create a generic ModelList with single-output models
        model1 = gen_module(models.SingleTaskGP, config)
        model2 = gen_module(models.SingleTaskGP, config)
        generic_model_list = ModelList(model1, model2)

        # Create a mock that returns a list when called
        class MockPath:
            def __call__(self, X):
                # Return a list of tensors to trigger the else branch
                return [torch.randn(X.shape[0]), torch.randn(X.shape[0])]

        with patch(
            "botorch.sampling.pathwise.posterior_samplers.draw_matheron_paths",
            return_value=MockPath(),
        ):
            path_model = get_matheron_path_model(generic_model_list)
            self.assertEqual(path_model.num_outputs, 2)

            # Test evaluation
            X = torch.rand(4, config.num_inputs, device=self.device, dtype=config.dtype)
            output = path_model(X)
            self.assertEqual(output.shape, (4, 2))

        # Also test with a ModelListGP that has empty models
        # Create an empty ModelListGP
        empty_model_list = models.ModelListGP()

        with patch(
            "botorch.sampling.pathwise.posterior_samplers.draw_matheron_paths",
            return_value=MockPath(),
        ):
            path_model = get_matheron_path_model(empty_model_list)
            self.assertEqual(path_model.num_outputs, 0)

            # The path should return an empty list for empty model list
            class EmptyMockPath:
                def __call__(self, X):
                    return []

            with patch(
                "botorch.sampling.pathwise.posterior_samplers.draw_matheron_paths",
                return_value=EmptyMockPath(),
            ):
                path_model2 = get_matheron_path_model(empty_model_list)
                # For empty list, torch.stack should create a tensor with shape (..., 0)
                X = torch.rand(4, 2, device=self.device, dtype=config.dtype)
                output2 = path_model2(X)
                self.assertEqual(output2.shape, (4, 0))

        # Test the non-batched ModelListGP case
        from botorch.models.model import ModelList

        # Create models without _num_outputs > 1 to trigger the else branch
        model1 = gen_module(models.SingleTaskGP, config)
        model2 = gen_module(models.SingleTaskGP, config)

        # Create a ModelListGP with non-batched models
        non_batched_model_list = models.ModelListGP(model1, model2)

        # Mock path that returns non-batched outputs
        class NonBatchedMockPath:
            def __call__(self, X):
                # Return list of tensors (non-batched case)
                return [torch.randn(X.shape[0]), torch.randn(X.shape[0])]

        with patch(
            "botorch.sampling.pathwise.posterior_samplers.draw_matheron_paths",
            return_value=NonBatchedMockPath(),
        ):
            path_model3 = get_matheron_path_model(non_batched_model_list)
            self.assertEqual(path_model3.num_outputs, 2)

            X = torch.rand(4, config.num_inputs, device=self.device, dtype=config.dtype)
            output3 = path_model3(X)
            self.assertEqual(output3.shape, (4, 2))

        # Test multi-output model (non-ModelList)
        # TODO: Fix MultiTaskGP support - currently fails with dimension mismatch
        # multi_config = replace(config, num_tasks=3)
        # multi_model = gen_module(models.MultiTaskGP, multi_config)
        # path_model = get_matheron_path_model(multi_model)
        # self.assertEqual(path_model.num_outputs, 3)

        # X = torch.rand(4, config.num_inputs + 1, device=self.device,
        # dtype=config.dtype)  # +1 for task feature
        # output = path_model(X)
        # self.assertEqual(output.shape, (4, 3))

        # Test UnsupportedError for model-list of multi-output models

        # Create a MultiTaskGP which has _task_feature attribute
        multi_config = replace(config, num_tasks=2)
        multi_model = gen_module(models.MultiTaskGP, multi_config)

        # Create a ModelListGP with the multi-output model
        model_list_multi = models.ModelListGP(multi_model)

        with self.assertRaisesRegex(
            UnsupportedError, "A model-list of multi-output models"
        ):
            get_matheron_path_model(model_list_multi)

        # Test the non-ModelList multi-output case
        # Create a mock model with multiple outputs to test the else branch
        # in get_matheron_path_model
        class MockMultiOutputGP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.num_outputs = 3

        mock_multi_model = MockMultiOutputGP()

        # Mock the draw_matheron_paths to return a dummy path
        class MockPath:
            def __call__(self, X):
                # For multi-output case, X is unsqueezed to add joint dimension
                # X has shape (1, batch, d) for multi-output
                # We need to return shape (m, q) so after transpose(-1, -2)
                # we get (q, m)
                if X.ndim == 3:  # multi-output case with unsqueezed dimension
                    # X shape is (1, q, d), return (m, q) where m=3
                    return torch.randn(3, X.shape[1])
                else:
                    return torch.randn(X.shape[0])

        with patch(
            "botorch.sampling.pathwise.posterior_samplers.draw_matheron_paths",
            return_value=MockPath(),
        ):
            path_model = get_matheron_path_model(mock_multi_model)
            self.assertEqual(path_model.num_outputs, 3)

            # Test evaluation - this should trigger the else branch for multi-output
            X = torch.rand(4, config.num_inputs, device=self.device, dtype=config.dtype)
            output = path_model(X)
            # For multi-output model, output should have shape (4, 3)
            self.assertEqual(output.shape, (4, 3))


class TestDrawMatheronPaths(BotorchTestCase):
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

    def test_base_models(self, slack: float = 10.0):
        sample_shape = Size([32, 32])
        for config, model in self.base_models:
            kernel = (
                model.model.covar_module
                if isinstance(model, models.SingleTaskVariationalGP)
                else model.covar_module
            )
            base_features = list(range(config.num_inputs))
            if isinstance(model, models.MultiTaskGP):
                del base_features[model._task_feature]

            with torch.random.fork_rng():
                torch.random.manual_seed(config.seed)
                paths = draw_matheron_paths(
                    model=model,
                    sample_shape=sample_shape,
                    prior_sampler=partial(
                        draw_kernel_feature_paths,
                        num_random_features=config.num_random_features,
                    ),
                )
                self.assertIsInstance(paths, MatheronPath)
                n = 16
                Z = gen_random_inputs(
                    model,
                    batch_shape=[n],
                    transformed=True,
                    task_id=0,  # only used by multi-task models
                )
                X = (
                    model.input_transform.untransform(Z)
                    if hasattr(model, "input_transform")
                    else Z
                )

            samples = paths(X)
            model.eval()
            with delattr_ctx(model, "outcome_transform"):
                posterior = (
                    model.posterior(X[..., base_features], output_indices=[0])
                    if isinstance(model, models.MultiTaskGP)
                    else model.posterior(X)
                )
                mvn = posterior.mvn

            if isinstance(mvn, MultitaskMultivariateNormal):
                num_tasks = kernel.batch_shape[0]
                exact_mean = mvn.mean.transpose(-2, -1)
                exact_covar = mvn.covariance_matrix.view(num_tasks, n, num_tasks, n)
                exact_covar = torch.stack(
                    [exact_covar[..., i, :, i, :] for i in range(num_tasks)], dim=-3
                )
            else:
                exact_mean = mvn.mean
                exact_covar = mvn.covariance_matrix

            if isinstance(model, SingleTaskVariationalGP):
                prior = model.forward(Z)
            else:
                prior = model.forward(Z)
            istd = prior.covariance_matrix.diagonal(dim1=-2, dim2=-1).rsqrt()
            exact_mean = istd * exact_mean
            exact_covar = istd.unsqueeze(-1) * exact_covar * istd.unsqueeze(-2)
            if hasattr(model, "outcome_transform"):
                if kernel.batch_shape:
                    samples, _ = model.outcome_transform(samples.transpose(-2, -1))
                    samples = samples.transpose(-2, -1)
                else:
                    samples, _ = model.outcome_transform(samples.unsqueeze(-1))
                    samples = samples.squeeze(-1)

            samples = istd * samples.view(-1, *samples.shape[len(sample_shape) :])
            sample_mean = samples.mean(dim=0)
            sample_covar = (samples - sample_mean).permute(*range(1, samples.ndim), 0)
            sample_covar = torch.divide(
                sample_covar @ sample_covar.transpose(-2, -1), sample_shape.numel()
            )

            base_atol = slack * sample_shape.numel() ** -0.5
            allclose_kwargs = {"atol": base_atol * 2.0}
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

    def test_model_lists(self, tol: float = 3.0):
        sample_shape = Size([32, 32])
        for config, model_list in self.model_lists:
            with torch.random.fork_rng():
                torch.random.manual_seed(config.seed)
                path_list = draw_matheron_paths(
                    model=model_list,
                    sample_shape=sample_shape,
                )
                self.assertIsInstance(path_list, PathList)

                X = gen_random_inputs(model_list.models[0], batch_shape=[4])
                sample_list = path_list(X)
                self.assertIsInstance(sample_list, list)
                self.assertEqual(len(sample_list), len(model_list.models))
                for path, sample in zip(path_list, sample_list):
                    self.assertTrue(path(X).equal(sample))
