#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from botorch.models.kernels.categorical import CategoricalKernel
from botorch.utils.testing import BotorchTestCase
from gpytorch.test.base_kernel_test_case import BaseKernelTestCase


class TestCategoricalKernel(BotorchTestCase, BaseKernelTestCase):
    def create_kernel_no_ard(self, **kwargs):
        return CategoricalKernel(**kwargs)

    def create_data_no_batch(self):
        return torch.randint(3, size=(5, 10)).to(dtype=torch.float)

    def create_data_single_batch(self):
        return torch.randint(3, size=(2, 5, 3)).to(dtype=torch.float)

    def create_data_double_batch(self):
        return torch.randint(3, size=(3, 2, 5, 3)).to(dtype=torch.float)

    def test_initialize_lengthscale(self):
        kernel = CategoricalKernel()
        kernel.initialize(lengthscale=1)
        actual_value = torch.tensor(1.0).view_as(kernel.lengthscale)
        self.assertLess(torch.linalg.norm(kernel.lengthscale - actual_value), 1e-5)

    def test_initialize_lengthscale_batch(self):
        kernel = CategoricalKernel(batch_shape=torch.Size([2]))
        ls_init = torch.tensor([1.0, 2.0])
        kernel.initialize(lengthscale=ls_init)
        actual_value = ls_init.view_as(kernel.lengthscale)
        self.assertLess(torch.linalg.norm(kernel.lengthscale - actual_value), 1e-5)

    def test_forward(self):
        x1 = torch.tensor([[4, 2], [3, 1], [8, 5], [7, 6]], dtype=torch.float)
        x2 = torch.tensor([[4, 2], [3, 0], [4, 4]], dtype=torch.float)
        lengthscale = 2
        kernel = CategoricalKernel().initialize(lengthscale=lengthscale)
        kernel.eval()
        sc_dists = (x1.unsqueeze(-2) != x2.unsqueeze(-3)) / lengthscale
        actual = torch.exp(-sc_dists.mean(-1))
        res = kernel(x1, x2).to_dense()
        self.assertAllClose(res, actual)

    def test_active_dims(self):
        x1 = torch.tensor([[4, 2], [3, 1], [8, 5], [7, 6]], dtype=torch.float)
        x2 = torch.tensor([[4, 2], [3, 0], [4, 4]], dtype=torch.float)
        lengthscale = 2
        kernel = CategoricalKernel(active_dims=[0]).initialize(lengthscale=lengthscale)
        kernel.eval()
        dists = x1[:, :1].unsqueeze(-2) != x2[:, :1].unsqueeze(-3)
        sc_dists = dists / lengthscale
        actual = torch.exp(-sc_dists.mean(-1))
        res = kernel(x1, x2).to_dense()
        self.assertAllClose(res, actual)

    def test_ard(self):
        x1 = torch.tensor([[4, 2], [3, 1], [8, 5]], dtype=torch.float)
        x2 = torch.tensor([[4, 2], [3, 0], [4, 4]], dtype=torch.float)
        lengthscales = torch.tensor([1, 2], dtype=torch.float).view(1, 1, 2)

        kernel = CategoricalKernel(ard_num_dims=2)
        kernel.initialize(lengthscale=lengthscales)
        kernel.eval()

        sc_dists = x1.unsqueeze(-2) != x2.unsqueeze(-3)
        sc_dists = sc_dists / lengthscales
        actual = torch.exp(-sc_dists.mean(-1))
        res = kernel(x1, x2).to_dense()
        self.assertAllClose(res, actual)

        # diag
        res = kernel(x1, x2).diagonal()
        actual = torch.diagonal(actual, dim1=-1, dim2=-2)
        self.assertAllClose(res, actual)

        # batch_dims
        actual = torch.exp(-sc_dists).transpose(-1, -3)
        res = kernel(x1, x2, last_dim_is_batch=True).to_dense()
        self.assertAllClose(res, actual)

        # batch_dims + diag
        res = kernel(x1, x2, last_dim_is_batch=True).diagonal()
        self.assertAllClose(res, torch.diagonal(actual, dim1=-1, dim2=-2))

    def test_ard_batch(self):
        x1 = torch.tensor(
            [
                [[4, 2, 1], [3, 1, 5]],
                [[3, 2, 3], [6, 1, 7]],
            ],
            dtype=torch.float,
        )
        x2 = torch.tensor([[[4, 2, 1], [6, 0, 0]]], dtype=torch.float)
        lengthscales = torch.tensor([[[1, 2, 1]]], dtype=torch.float)

        kernel = CategoricalKernel(batch_shape=torch.Size([2]), ard_num_dims=3)
        kernel.initialize(lengthscale=lengthscales)
        kernel.eval()

        sc_dists = x1.unsqueeze(-2) != x2.unsqueeze(-3)
        sc_dists = sc_dists / lengthscales.unsqueeze(-2)
        actual = torch.exp(-sc_dists.mean(-1))
        res = kernel(x1, x2).to_dense()
        self.assertAllClose(res, actual)

    def test_ard_separate_batch(self):
        x1 = torch.tensor(
            [
                [[4, 2, 1], [3, 1, 5]],
                [[3, 2, 3], [6, 1, 7]],
            ],
            dtype=torch.float,
        )
        x2 = torch.tensor([[[4, 2, 1], [6, 0, 0]]], dtype=torch.float)
        lengthscales = torch.tensor([[[1, 2, 1]], [[2, 1, 0.5]]], dtype=torch.float)

        kernel = CategoricalKernel(batch_shape=torch.Size([2]), ard_num_dims=3)
        kernel.initialize(lengthscale=lengthscales)
        kernel.eval()

        sc_dists = x1.unsqueeze(-2) != x2.unsqueeze(-3)
        sc_dists = sc_dists / lengthscales.unsqueeze(-2)
        actual = torch.exp(-sc_dists.mean(-1))
        res = kernel(x1, x2).to_dense()
        self.assertAllClose(res, actual)

        # diag
        res = kernel(x1, x2).diagonal()
        actual = torch.diagonal(actual, dim1=-1, dim2=-2)
        self.assertAllClose(res, actual)

        # batch_dims
        actual = torch.exp(-sc_dists).transpose(-1, -3)
        res = kernel(x1, x2, last_dim_is_batch=True).to_dense()
        self.assertAllClose(res, actual)

        # batch_dims + diag
        res = kernel(x1, x2, last_dim_is_batch=True).diagonal()
        self.assertAllClose(res, torch.diagonal(actual, dim1=-1, dim2=-2))
