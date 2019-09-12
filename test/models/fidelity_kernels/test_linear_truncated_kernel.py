#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
from botorch.exceptions import UnsupportedError
from botorch.models.fidelity_kernels.linear_truncated_fidelity import (
    LinearTruncatedFidelityKernel,
)
from botorch.utils.testing import BotorchTestCase
from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.kernels.rbf_kernel import RBFKernel
from gpytorch.priors.torch_priors import GammaPrior, NormalPrior
from gpytorch.test.base_kernel_test_case import BaseKernelTestCase


class TestLinearTruncatedFidelityKernel(BotorchTestCase, BaseKernelTestCase):
    def create_kernel_no_ard(self, **kwargs):
        return LinearTruncatedFidelityKernel(**kwargs)

    def create_data_no_batch(self):
        return torch.rand(50, 10)

    def create_data_single_batch(self):
        return torch.rand(2, 50, 3)

    def create_data_double_batch(self):
        return torch.rand(3, 2, 50, 3)

    def test_compute_linear_truncated_kernel_no_batch(self):
        x1 = torch.tensor([1, 0.1, 0.2, 2, 0.3, 0.4], dtype=torch.float).view(2, 3)
        x2 = torch.tensor([3, 0.5, 0.6, 4, 0.7, 0.8], dtype=torch.float).view(2, 3)
        t_1 = torch.tensor([0.3584, 0.1856, 0.2976, 0.1584], dtype=torch.float).view(
            2, 2
        )
        for nu in {0.5, 1.5, 2.5}:
            for train_data_fidelity in {False, True}:
                kernel = LinearTruncatedFidelityKernel(
                    nu=nu, dimension=3, train_data_fidelity=train_data_fidelity
                )
                kernel.power = 1
                if train_data_fidelity:
                    active_dimsM = [0]
                    t_2 = torch.tensor(
                        [0.4725, 0.2889, 0.4025, 0.2541], dtype=torch.float
                    ).view(2, 2)
                    t_3 = torch.tensor(
                        [0.1685, 0.0531, 0.1168, 0.0386], dtype=torch.float
                    ).view(2, 2)
                    t = 1 + t_1 + t_2 + t_3
                else:
                    active_dimsM = [0, 1]
                    t = 1 + t_1

                matern_ker = MaternKernel(nu=nu, active_dims=active_dimsM)
                matern_term = matern_ker(x1, x2).evaluate()
                actual = t * matern_term
                res = kernel(x1, x2).evaluate()
                self.assertLess(torch.norm(res - actual), 1e-4)

    def test_compute_linear_truncated_kernel_with_batch(self):
        x1 = torch.tensor(
            [1, 0.1, 0.2, 3, 0.3, 0.4, 5, 0.5, 0.6, 7, 0.7, 0.8], dtype=torch.float
        ).view(2, 2, 3)
        x2 = torch.tensor(
            [2, 0.8, 0.7, 4, 0.6, 0.5, 6, 0.4, 0.3, 8, 0.2, 0.1], dtype=torch.float
        ).view(2, 2, 3)
        t_1 = torch.tensor(
            [0.2736, 0.44, 0.2304, 0.36, 0.3304, 0.3816, 0.1736, 0.1944],
            dtype=torch.float,
        ).view(2, 2, 2)
        batch_shape = torch.Size([2])
        dimension = 3
        for nu in {0.5, 1.5, 2.5}:
            for train_data_fidelity in {False, True}:
                kernel = LinearTruncatedFidelityKernel(
                    nu=nu,
                    dimension=dimension,
                    train_data_fidelity=train_data_fidelity,
                    batch_shape=batch_shape,
                )
                kernel.power = 1
                kernel.train_data_fidelity = train_data_fidelity
                if train_data_fidelity:
                    active_dimsM = [0]
                    t_2 = torch.tensor(
                        [0.0527, 0.167, 0.0383, 0.1159, 0.1159, 0.167, 0.0383, 0.0527],
                        dtype=torch.float,
                    ).view(2, 2, 2)
                    t_3 = torch.tensor(
                        [0.1944, 0.3816, 0.1736, 0.3304, 0.36, 0.44, 0.2304, 0.2736],
                        dtype=torch.float,
                    ).view(2, 2, 2)
                    t = 1 + t_1 + t_2 + t_3
                else:
                    active_dimsM = [0, 1]
                    t = 1 + t_1

                matern_ker = MaternKernel(
                    nu=nu, active_dims=active_dimsM, batch_shape=batch_shape
                )
                matern_term = matern_ker(x1, x2).evaluate()
                actual = t * matern_term
                res = kernel(x1, x2).evaluate()
                self.assertLess(torch.norm(res - actual), 1e-4)

    def test_initialize_lengthscale_prior(self):
        kernel = LinearTruncatedFidelityKernel()
        self.assertTrue(isinstance(kernel.covar_module_1.lengthscale_prior, GammaPrior))
        self.assertTrue(isinstance(kernel.covar_module_2.lengthscale_prior, GammaPrior))
        kernel2 = LinearTruncatedFidelityKernel(lengthscale_prior=NormalPrior(1, 1))
        self.assertTrue(
            isinstance(kernel2.covar_module_1.lengthscale_prior, NormalPrior)
        )
        kernel2 = LinearTruncatedFidelityKernel(lengthscale_2_prior=NormalPrior(1, 1))
        self.assertTrue(
            isinstance(kernel2.covar_module_2.lengthscale_prior, NormalPrior)
        )

    def test_initialize_power_prior(self):
        kernel = LinearTruncatedFidelityKernel(power_prior=NormalPrior(1, 1))
        self.assertTrue(isinstance(kernel.power_prior, NormalPrior))

    def test_initialize_power(self):
        kernel = LinearTruncatedFidelityKernel()
        kernel.initialize(power=1)
        actual_value = torch.tensor(1, dtype=torch.float).view_as(kernel.power)
        self.assertLess(torch.norm(kernel.power - actual_value), 1e-5)

    def test_initialize_power_batch(self):
        kernel = LinearTruncatedFidelityKernel(batch_shape=torch.Size([2]))
        power_init = torch.tensor([1, 2], dtype=torch.float)
        kernel.initialize(power=power_init)
        actual_value = power_init.view_as(kernel.power)
        self.assertLess(torch.norm(kernel.power - actual_value), 1e-5)

    def test_raise_fidelity_error(self):
        kernel = LinearTruncatedFidelityKernel
        with self.assertRaises(UnsupportedError):
            kernel(train_iteration_fidelity=False, train_data_fidelity=False)

    def test_raise_matern_error(self):
        with self.assertRaises(ValueError):
            LinearTruncatedFidelityKernel(nu=1)

    def test_active_dims_list(self):
        kernel = LinearTruncatedFidelityKernel(dimension=10, active_dims=[0, 2, 4, 6])
        x = self.create_data_no_batch()
        covar_mat = kernel(x).evaluate_kernel().evaluate()
        kernel_basic = LinearTruncatedFidelityKernel(dimension=4)
        covar_mat_actual = kernel_basic(x[:, [0, 2, 4, 6]]).evaluate_kernel().evaluate()
        self.assertLess(
            torch.norm(covar_mat - covar_mat_actual) / covar_mat_actual.norm(), 1e-4
        )

    def test_active_dims_range(self):
        active_dims = list(range(3, 9))
        kernel = LinearTruncatedFidelityKernel(dimension=10, active_dims=active_dims)
        x = self.create_data_no_batch()
        covar_mat = kernel(x).evaluate_kernel().evaluate()
        kernel_basic = LinearTruncatedFidelityKernel(dimension=6)
        covar_mat_actual = kernel_basic(x[:, active_dims]).evaluate_kernel().evaluate()

        self.assertLess(
            torch.norm(covar_mat - covar_mat_actual) / covar_mat_actual.norm(), 1e-4
        )

    def test_initialize_covar_module(self):
        kernel = LinearTruncatedFidelityKernel()
        self.assertTrue(isinstance(kernel.covar_module_1, MaternKernel))
        self.assertTrue(isinstance(kernel.covar_module_2, MaternKernel))
        kernel.covar_module_1 = RBFKernel()
        kernel.covar_module_2 = RBFKernel()
        self.assertTrue(isinstance(kernel.covar_module_1, RBFKernel))
        self.assertTrue(isinstance(kernel.covar_module_2, RBFKernel))
        kernel2 = LinearTruncatedFidelityKernel(
            covar_module_1=RBFKernel(), covar_module_2=RBFKernel()
        )
        self.assertTrue(isinstance(kernel2.covar_module_1, RBFKernel))
        self.assertTrue(isinstance(kernel2.covar_module_2, RBFKernel))
