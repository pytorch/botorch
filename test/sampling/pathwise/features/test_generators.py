#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from math import ceil

import torch
from botorch.exceptions.errors import UnsupportedError
from botorch.sampling.pathwise.features.generators import gen_kernel_feature_map
from botorch.sampling.pathwise.features.maps import FourierFeatureMap
from botorch.sampling.pathwise.utils import is_finite_dimensional
from botorch.utils.testing import BotorchTestCase
from gpytorch import kernels


class TestGenKernelFeatureMap(BotorchTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.num_inputs = d = 2
        self.num_random_features = 4096
        self.kernels = []

        for kernel in (
            kernels.MaternKernel(nu=0.5, batch_shape=torch.Size([]), ard_num_dims=d),
            kernels.MaternKernel(nu=1.5, ard_num_dims=1, active_dims=[0]),
            kernels.ScaleKernel(
                kernels.MaternKernel(
                    nu=2.5, ard_num_dims=d, batch_shape=torch.Size([2])
                )
            ),
            kernels.ScaleKernel(
                kernels.RBFKernel(ard_num_dims=1, batch_shape=torch.Size([2, 2])),
                active_dims=[1],
            ),
            kernels.ProductKernel(
                kernels.RBFKernel(ard_num_dims=d),
                kernels.MaternKernel(nu=2.5, ard_num_dims=d),
            ),
        ):
            kernel.to(dtype=torch.float64, device=self.device)
            kern = (
                kernel.base_kernel
                if isinstance(kernel, kernels.ScaleKernel)
                else kernel
            )
            if hasattr(kern, "raw_lengthscale"):
                if isinstance(kern, kernels.MaternKernel):
                    shape = (
                        kern.raw_lengthscale.shape
                        if kern.ard_num_dims is None
                        else torch.Size([*kern.batch_shape, 1, kern.ard_num_dims])
                    )
                    kern.raw_lengthscale = torch.nn.Parameter(
                        torch.zeros(shape, dtype=torch.float64, device=self.device)
                    )
                elif isinstance(kern, kernels.RBFKernel):
                    shape = (
                        kern.raw_lengthscale.shape
                        if kern.ard_num_dims is None
                        else torch.Size([*kern.batch_shape, 1, kern.ard_num_dims])
                    )
                    kern.raw_lengthscale = torch.nn.Parameter(
                        torch.zeros(shape, dtype=torch.float64, device=self.device)
                    )

                with torch.random.fork_rng():
                    torch.manual_seed(0)
                    kern.raw_lengthscale.data.add_(
                        torch.rand_like(kern.raw_lengthscale) * 0.2 - 2.0
                    )  # Initialize to small random values

            self.kernels.append(kernel)

    def test_gen_kernel_feature_map(self, slack: float = 3.0):
        for kernel in self.kernels:
            with torch.random.fork_rng():
                torch.random.manual_seed(0)
                feature_map = gen_kernel_feature_map(
                    kernel=kernel,
                    num_ambient_inputs=self.num_inputs,
                    num_random_features=self.num_random_features,
                )

                n = 4
                m = ceil(n * kernel.batch_shape.numel() ** -0.5)
                for input_batch_shape in ((n**2,), (m, *kernel.batch_shape, m)):
                    X = torch.rand(
                        (*input_batch_shape, self.num_inputs),
                        device=kernel.device,
                        dtype=kernel.dtype,
                    )

                    with self.subTest("test_initialization"):
                        if isinstance(feature_map, FourierFeatureMap):
                            self.assertEqual(feature_map.weight.dtype, kernel.dtype)
                            self.assertEqual(feature_map.weight.device, kernel.device)
                            self.assertEqual(
                                feature_map.weight.shape[-1],
                                (
                                    self.num_inputs
                                    if kernel.active_dims is None
                                    else len(kernel.active_dims)
                                ),
                            )

                    with self.subTest("test_covariance"):
                        features = feature_map(X)
                        test_shape = torch.broadcast_shapes(
                            (*X.shape[:-1], feature_map.output_shape[0]),
                            kernel.batch_shape + (1, 1),
                        )
                        self.assertEqual(features.shape, test_shape)

                        K0 = features @ features.transpose(-2, -1)
                        K1 = kernel(X).to_dense()

                        # Normalize by prior standard deviations
                        istd = K1.diagonal(dim1=-2, dim2=-1).rsqrt()
                        K0 = istd.unsqueeze(-1) * K0 * istd.unsqueeze(-2)
                        K1 = istd.unsqueeze(-1) * K1 * istd.unsqueeze(-2)

                        allclose_kwargs = {
                            "atol": slack * self.num_random_features**-0.5
                        }
                        if not is_finite_dimensional(kernel):
                            num_random_features_per_map = self.num_random_features / (
                                1
                                if not is_finite_dimensional(kernel, max_depth=0)
                                else sum(
                                    not is_finite_dimensional(k)
                                    for k in kernel.modules()
                                    if k is not kernel
                                )
                            )
                            allclose_kwargs["atol"] = (
                                slack * num_random_features_per_map**-0.5
                            )

                        self.assertTrue(K0.allclose(K1, **allclose_kwargs))

        # Test requesting an odd number of features
        with self.assertRaisesRegex(UnsupportedError, "Expected an even number"):
            gen_kernel_feature_map(
                kernel=self.kernels[0],
                num_ambient_inputs=self.num_inputs,
                num_random_features=3,
            )
