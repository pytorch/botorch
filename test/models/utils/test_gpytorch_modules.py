#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
from botorch.models.utils.gpytorch_modules import (
    get_gaussian_likelihood_with_gamma_prior,
    get_matern_kernel_with_gamma_prior,
    MIN_INFERRED_NOISE_LEVEL,
)
from botorch.utils.testing import BotorchTestCase
from gpytorch.constraints.constraints import GreaterThan
from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood
from gpytorch.priors.torch_priors import GammaPrior


class TestGPyTorchModules(BotorchTestCase):
    def test_get_matern_kernel_with_gamma_prior(self):
        for batch_shape in (None, torch.Size([2])):
            kernel = get_matern_kernel_with_gamma_prior(
                ard_num_dims=2, batch_shape=batch_shape
            )
            self.assertIsInstance(kernel, ScaleKernel)
            self.assertEqual(kernel.batch_shape, batch_shape or torch.Size([]))
            prior = kernel.outputscale_prior
            self.assertIsInstance(prior, GammaPrior)
            self.assertAllClose(prior.concentration.item(), 2.0)
            self.assertAllClose(prior.rate.item(), 0.15)
            base_kernel = kernel.base_kernel
            self.assertIsInstance(base_kernel, MaternKernel)
            self.assertEqual(base_kernel.batch_shape, batch_shape or torch.Size([]))
            self.assertEqual(base_kernel.ard_num_dims, 2)
            prior = base_kernel.lengthscale_prior
            self.assertIsInstance(prior, GammaPrior)
            self.assertAllClose(prior.concentration.item(), 3.0)
            self.assertAllClose(prior.rate.item(), 6.0)

    def test_get_gaussian_likelihood_with_gamma_prior(self):
        for batch_shape in (None, torch.Size([2])):
            likelihood = get_gaussian_likelihood_with_gamma_prior(
                batch_shape=batch_shape
            )
            self.assertIsInstance(likelihood, GaussianLikelihood)
            expected_shape = (batch_shape or torch.Size([])) + (1,)
            self.assertEqual(likelihood.raw_noise.shape, expected_shape)
            prior = likelihood.noise_covar.noise_prior
            self.assertIsInstance(prior, GammaPrior)
            self.assertAllClose(prior.concentration.item(), 1.1)
            self.assertAllClose(prior.rate.item(), 0.05)
            constraint = likelihood.noise_covar.raw_noise_constraint
            self.assertIsInstance(constraint, GreaterThan)
            self.assertAllClose(constraint.lower_bound.item(), MIN_INFERRED_NOISE_LEVEL)
            self.assertIsNone(constraint._transform)
            self.assertAllClose(constraint.initial_value.item(), 2.0)
