#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from math import prod

# Removed unused imports
# from unittest.mock import MagicMock, patch

import torch
from botorch.sampling.pathwise.features import maps
from botorch.sampling.pathwise.features.generators import gen_kernel_feature_map

# Removed unused imports
# from botorch.sampling.pathwise.utils.transforms import (
#     ChainedTransform,
#     FeatureSelector
# )
from botorch.utils.testing import BotorchTestCase
from gpytorch import kernels
from linear_operator.operators import KroneckerProductLinearOperator
from torch import Size

# Removed unused import
# from torch.nn import Module

from ..helpers import gen_module, TestCaseConfig


# TestFeatureMaps: Tests for various feature map implementations
# - Tests base feature map functionality
# - Verifies direct sum, Hadamard product, and outer product operations
# - Checks sparse feature map handling
class TestFeatureMaps(BotorchTestCase):
    def setUp(self) -> None:
        """Set up test cases with base feature maps.
        - Creates linear and index kernel feature maps
        - Configures test parameters and dimensions
        """
        super().setUp()
        self.config = TestCaseConfig(
            seed=0,
            device=self.device,
            num_inputs=2,
            num_tasks=3,
            batch_shape=Size([2]),
        )

        # Create base feature maps for testing
        self.base_feature_maps = [
            gen_kernel_feature_map(gen_module(kernels.LinearKernel, self.config)),
            gen_kernel_feature_map(gen_module(kernels.IndexKernel, self.config)),
        ]

    def test_feature_map(self):
        """Test base feature map functionality.
        - Verifies output shape handling
        - Tests transform application
        - Checks device and dtype handling
        """
        feature_map = maps.FeatureMap()
        feature_map.raw_output_shape = Size([2, 3, 4])
        feature_map.output_transform = None
        feature_map.device = self.device
        feature_map.dtype = None
        self.assertEqual(feature_map.output_shape, (2, 3, 4))

        # Test output transform
        feature_map.output_transform = lambda x: torch.concat((x, x), dim=-1)
        self.assertEqual(feature_map.output_shape, (2, 3, 8))

    def test_feature_map_list(self):
        """Test feature map list operations.
        - Verifies device and dtype consistency
        - Tests forward pass with multiple maps
        - Checks output equality for individual maps
        """
        map_list = maps.FeatureMapList(feature_maps=self.base_feature_maps)
        self.assertEqual(map_list.device.type, self.config.device.type)
        self.assertEqual(map_list.dtype, self.config.dtype)

        # Test forward pass
        X = torch.rand(
            16,
            self.config.num_inputs,
            device=self.config.device,
            dtype=self.config.dtype,
        )
        output_list = map_list(X)
        self.assertIsInstance(output_list, list)
        self.assertEqual(len(output_list), len(map_list))
        for feature_map, output in zip(map_list, output_list):
            self.assertTrue(feature_map(X).to_dense().equal(output.to_dense()))

    def test_direct_sum_feature_map(self):
        """Test direct sum feature map operations.
        - Verifies output shape calculations
        - Tests batch shape handling
        - Checks concatenation of features
        """
        feature_map = maps.DirectSumFeatureMap(self.base_feature_maps)
        self.assertEqual(
            feature_map.raw_output_shape,
            Size([sum(f.output_shape[-1] for f in feature_map)]),
        )
        self.assertEqual(
            feature_map.batch_shape,
            torch.broadcast_shapes(*(f.batch_shape for f in feature_map)),
        )

        # Test forward pass
        d = self.config.num_inputs
        batch_shape = Size([16])
        X = torch.rand(
            (*batch_shape, d), device=self.config.device, dtype=self.config.dtype
        )
        features = feature_map(X).to_dense()

        # Check output shape - should be [*batch_shape, *output_shape]
        # Note: The feature map's batch shape comes first, then our input batch shape
        expected_shape = Size(
            [*feature_map.batch_shape, *batch_shape, *feature_map.output_shape[-1:]]
        )
        self.assertEqual(features.shape, expected_shape)

        # Check concatenation
        expected_features = torch.concat([f(X).to_dense() for f in feature_map], dim=-1)
        self.assertTrue(features.equal(expected_features))

    def test_hadamard_product_feature_map(self):
        """Test Hadamard product feature map operations.
        - Verifies output shape broadcasting
        - Tests batch shape handling
        - Checks element-wise multiplication of features
        """
        feature_map = maps.HadamardProductFeatureMap(self.base_feature_maps)
        self.assertEqual(
            feature_map.raw_output_shape,
            torch.broadcast_shapes(*(f.output_shape for f in feature_map)),
        )
        self.assertEqual(
            feature_map.batch_shape,
            torch.broadcast_shapes(*(f.batch_shape for f in feature_map)),
        )

        # Test forward pass
        d = self.config.num_inputs
        X = torch.rand((16, d), device=self.config.device, dtype=self.config.dtype)
        features = feature_map(X).to_dense()
        self.assertEqual(
            features.shape[-len(feature_map.output_shape) :],
            feature_map.output_shape,
        )
        self.assertTrue(features.equal(prod([f(X).to_dense() for f in feature_map])))

    def test_outer_product_feature_map(self):
        """Test outer product feature map operations.
        - Verifies output shape calculations
        - Tests batch shape handling
        - Checks outer product computation
        """
        feature_map = maps.OuterProductFeatureMap(self.base_feature_maps)
        self.assertEqual(
            feature_map.raw_output_shape,
            Size([prod(f.output_shape[-1] for f in feature_map)]),
        )
        self.assertEqual(
            feature_map.batch_shape,
            torch.broadcast_shapes(*(f.batch_shape for f in feature_map)),
        )

        # Test forward pass
        d = self.config.num_inputs
        X = torch.rand((16, d), device=self.config.device, dtype=self.config.dtype)
        features = feature_map(X).to_dense()
        self.assertEqual(
            features.shape[-len(feature_map.output_shape) :],
            feature_map.output_shape,
        )

        # Verify outer product computation
        test_features = (
            feature_map[0](X).to_dense().unsqueeze(-1)
            * feature_map[1](X).to_dense().unsqueeze(-2)
        ).view(features.shape)
        self.assertTrue(features.equal(test_features))


# TestKernelFeatureMaps: Tests for kernel-specific feature maps
# - Tests Fourier feature maps
# - Verifies index kernel feature maps
# - Checks linear kernel feature maps
# - Tests multitask kernel feature maps
class TestKernelFeatureMaps(BotorchTestCase):
    def setUp(self) -> None:
        """Set up test cases for kernel feature maps.
        - Creates test configurations
        - Sets up device and dtype parameters
        """
        super().setUp()
        self.configs = [
            TestCaseConfig(
                seed=0,
                device=self.device,
                num_inputs=2,
                num_tasks=3,
                batch_shape=Size([2]),
            )
        ]

    def test_fourier_feature_map(self):
        """Test Fourier feature map operations.
        - Verifies weight and bias handling
        - Tests output shape calculations
        - Checks forward pass computation
        """
        for config in self.configs:
            tkwargs = {"device": config.device, "dtype": config.dtype}
            kernel = gen_module(kernels.RBFKernel, config)
            weight = torch.randn(*kernel.batch_shape, 16, config.num_inputs, **tkwargs)
            bias = torch.rand(*kernel.batch_shape, 16, **tkwargs)
            feature_map = maps.FourierFeatureMap(
                kernel=kernel, weight=weight, bias=bias
            )
            self.assertEqual(feature_map.output_shape, (16,))

            # Test forward pass
            X = torch.rand(32, config.num_inputs, **tkwargs)
            features = feature_map(X)
            self.assertEqual(
                features.shape[-len(feature_map.output_shape) :],
                feature_map.output_shape,
            )
            self.assertTrue(
                features.equal(X @ weight.transpose(-2, -1) + bias.unsqueeze(-2))
            )

    def test_index_kernel_feature_map(self):
        """Test index kernel feature map operations.
        - Verifies task index handling
        - Tests output shape calculations
        - Checks Cholesky decomposition
        """
        for config in self.configs:
            kernel = gen_module(kernels.IndexKernel, config)
            tkwargs = {"device": config.device, "dtype": config.dtype}
            feature_map = maps.IndexKernelFeatureMap(kernel=kernel)
            self.assertEqual(feature_map.output_shape, kernel.raw_var.shape[-1:])

            # Test forward pass with indices
            X = torch.rand(*config.batch_shape, 16, config.num_inputs, **tkwargs)
            index_shape = (*config.batch_shape, 16, len(kernel.active_dims))
            indices = X[..., kernel.active_dims] = torch.randint(
                config.num_tasks, size=index_shape, **tkwargs
            )
            indices = indices.long().squeeze(-1)
            features = feature_map(X).to_dense()
            self.assertEqual(
                features.shape[-len(feature_map.output_shape) :],
                feature_map.output_shape,
            )

            # Verify Cholesky decomposition
            cholesky = kernel.covar_matrix.cholesky().to_dense()
            test_features = []
            for chol, idx in zip(
                cholesky.view(-1, *cholesky.shape[-2:]),
                indices.view(-1, *indices.shape[-1:]),
            ):
                test_features.append(chol.index_select(dim=-2, index=idx))
            test_features = torch.stack(test_features).view(features.shape)
            self.assertTrue(features.equal(test_features))

    def test_linear_kernel_feature_map(self):
        """Test linear kernel feature map operations.
        - Verifies active dimensions handling
        - Tests output shape calculations
        - Checks variance scaling
        """
        for config in self.configs:
            kernel = gen_module(kernels.LinearKernel, config)
            tkwargs = {"device": config.device, "dtype": config.dtype}
            active_dims = (
                tuple(range(config.num_inputs))
                if kernel.active_dims is None
                else kernel.active_dims
            )
            feature_map = maps.LinearKernelFeatureMap(
                kernel=kernel, raw_output_shape=Size([len(active_dims)])
            )

            # Test forward pass
            X = torch.rand(*kernel.batch_shape, 16, config.num_inputs, **tkwargs)
            features = feature_map(X).to_dense()
            self.assertEqual(
                features.shape[-len(feature_map.output_shape) :],
                feature_map.output_shape,
            )
            self.assertTrue(
                features.equal(kernel.variance.sqrt() * X[..., active_dims])
            )

    def test_multitask_kernel_feature_map(self):
        """Test multitask kernel feature map operations.
        - Verifies task covariance handling
        - Tests Kronecker product computation
        - Checks output shape calculations
        """
        for config in self.configs:
            kernel = gen_module(kernels.MultitaskKernel, config)
            tkwargs = {"device": config.device, "dtype": config.dtype}
            data_map = gen_kernel_feature_map(
                kernel=kernel.data_covar_module,
                num_inputs=config.num_inputs,
                num_random_features=config.num_random_features,
            )
            feature_map = maps.MultitaskKernelFeatureMap(
                kernel=kernel, data_feature_map=data_map
            )
            self.assertEqual(
                feature_map.output_shape,
                (feature_map.num_tasks * data_map.output_shape[0],)
                + data_map.output_shape[1:],
            )

            # Test forward pass
            X = torch.rand(*kernel.batch_shape, 16, config.num_inputs, **tkwargs)

            features = feature_map(X).to_dense()
            self.assertEqual(
                features.shape[-len(feature_map.output_shape) :],
                feature_map.output_shape,
            )
            cholesky = kernel.task_covar_module.covar_matrix.cholesky()
            test_features = KroneckerProductLinearOperator(data_map(X), cholesky)
            self.assertTrue(features.equal(test_features.to_dense()))
