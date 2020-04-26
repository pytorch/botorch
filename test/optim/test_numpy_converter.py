#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from botorch.optim.numpy_converter import module_to_array, set_params_with_array
from botorch.utils.testing import BotorchTestCase
from gpytorch.constraints import GreaterThan
from gpytorch.kernels.rbf_kernel import RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means.constant_mean import ConstantMean
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.models.exact_gp import ExactGP


def _get_index(property_dict, parameter_name):
    idx = 0
    for p_name, ta in property_dict.items():
        if p_name == parameter_name:
            break
        idx += ta.shape.numel()
    return idx


class TestModuleToArray(BotorchTestCase):
    def test_basic(self):
        for dtype in (torch.float, torch.double):
            # get a test module
            train_x = torch.tensor([[1.0, 2.0, 3.0]], device=self.device, dtype=dtype)
            train_y = torch.tensor([4.0], device=self.device, dtype=dtype)
            likelihood = GaussianLikelihood()
            model = ExactGP(train_x, train_y, likelihood)
            model.covar_module = RBFKernel(ard_num_dims=3)
            model.mean_module = ConstantMean()
            model.to(device=self.device, dtype=dtype)
            mll = ExactMarginalLogLikelihood(likelihood, model)
            # test the basic case
            x, pdict, bounds = module_to_array(module=mll)
            self.assertTrue(np.array_equal(x, np.zeros(5)))
            expected_sizes = {
                "likelihood.noise_covar.raw_noise": torch.Size([1]),
                "model.covar_module.raw_lengthscale": torch.Size([1, 3]),
                "model.mean_module.constant": torch.Size([1]),
            }
            self.assertEqual(set(pdict.keys()), set(expected_sizes.keys()))
            for pname, val in pdict.items():
                self.assertEqual(val.dtype, dtype)
                self.assertEqual(val.shape, expected_sizes[pname])
                self.assertEqual(val.device.type, self.device.type)
            self.assertIsNone(bounds)

    def test_exclude(self):
        for dtype in (torch.float, torch.double):
            # get a test module
            train_x = torch.tensor([[1.0, 2.0, 3.0]], device=self.device, dtype=dtype)
            train_y = torch.tensor([4.0], device=self.device, dtype=dtype)
            likelihood = GaussianLikelihood()
            model = ExactGP(train_x, train_y, likelihood)
            model.covar_module = RBFKernel(ard_num_dims=3)
            model.mean_module = ConstantMean()
            model.to(device=self.device, dtype=dtype)
            mll = ExactMarginalLogLikelihood(likelihood, model)
            # test the basic case
            x, pdict, bounds = module_to_array(
                module=mll, exclude={"model.mean_module.constant"}
            )
            self.assertTrue(np.array_equal(x, np.zeros(4)))
            expected_sizes = {
                "likelihood.noise_covar.raw_noise": torch.Size([1]),
                "model.covar_module.raw_lengthscale": torch.Size([1, 3]),
            }
            self.assertEqual(set(pdict.keys()), set(expected_sizes.keys()))
            for pname, val in pdict.items():
                self.assertEqual(val.dtype, dtype)
                self.assertEqual(val.shape, expected_sizes[pname])
                self.assertEqual(val.device.type, self.device.type)
            self.assertIsNone(bounds)

    def test_manual_bounds(self):
        for dtype in (torch.float, torch.double):
            # get a test module
            train_x = torch.tensor([[1.0, 2.0, 3.0]], device=self.device, dtype=dtype)
            train_y = torch.tensor([4.0], device=self.device, dtype=dtype)
            likelihood = GaussianLikelihood()
            model = ExactGP(train_x, train_y, likelihood)
            model.covar_module = RBFKernel(ard_num_dims=3)
            model.mean_module = ConstantMean()
            model.to(device=self.device, dtype=dtype)
            mll = ExactMarginalLogLikelihood(likelihood, model)
            # test the basic case
            x, pdict, bounds = module_to_array(
                module=mll, bounds={"model.covar_module.raw_lengthscale": (0.1, None)}
            )
            self.assertTrue(np.array_equal(x, np.zeros(5)))
            expected_sizes = {
                "likelihood.noise_covar.raw_noise": torch.Size([1]),
                "model.covar_module.raw_lengthscale": torch.Size([1, 3]),
                "model.mean_module.constant": torch.Size([1]),
            }
            self.assertEqual(set(pdict.keys()), set(expected_sizes.keys()))
            for pname, val in pdict.items():
                self.assertEqual(val.dtype, dtype)
                self.assertEqual(val.shape, expected_sizes[pname])
                self.assertEqual(val.device.type, self.device.type)
            lower_exp = np.full_like(x, 0.1)
            for p in ("likelihood.noise_covar.raw_noise", "model.mean_module.constant"):
                lower_exp[_get_index(pdict, p)] = -np.inf
            self.assertTrue(np.equal(bounds[0], lower_exp).all())
            self.assertTrue(np.equal(bounds[1], np.full_like(x, np.inf)).all())

    def test_module_bounds(self):
        for dtype in (torch.float, torch.double):
            # get a test module
            train_x = torch.tensor([[1.0, 2.0, 3.0]], device=self.device, dtype=dtype)
            train_y = torch.tensor([4.0], device=self.device, dtype=dtype)
            likelihood = GaussianLikelihood(
                noise_constraint=GreaterThan(1e-5, transform=None)
            )
            model = ExactGP(train_x, train_y, likelihood)
            model.covar_module = RBFKernel(ard_num_dims=3)
            model.mean_module = ConstantMean()
            model.to(device=self.device, dtype=dtype)
            mll = ExactMarginalLogLikelihood(likelihood, model)
            # test the basic case
            x, pdict, bounds = module_to_array(
                module=mll, bounds={"model.covar_module.raw_lengthscale": (0.1, None)}
            )
            self.assertTrue(np.array_equal(x, np.zeros(5)))
            expected_sizes = {
                "likelihood.noise_covar.raw_noise": torch.Size([1]),
                "model.covar_module.raw_lengthscale": torch.Size([1, 3]),
                "model.mean_module.constant": torch.Size([1]),
            }
            self.assertEqual(set(pdict.keys()), set(expected_sizes.keys()))
            for pname, val in pdict.items():
                self.assertEqual(val.dtype, dtype)
                self.assertEqual(val.shape, expected_sizes[pname])
                self.assertEqual(val.device.type, self.device.type)
            lower_exp = np.full_like(x, 0.1)
            lower_exp[_get_index(pdict, "model.mean_module.constant")] = -np.inf
            lower_exp[_get_index(pdict, "likelihood.noise_covar.raw_noise")] = 1e-5
            self.assertTrue(np.allclose(bounds[0], lower_exp))
            self.assertTrue(np.equal(bounds[1], np.full_like(x, np.inf)).all())


class TestSetParamsWithArray(BotorchTestCase):
    def test_set_parameters(self):
        for dtype in (torch.float, torch.double):
            # get a test module
            train_x = torch.tensor([[1.0, 2.0, 3.0]], device=self.device, dtype=dtype)
            train_y = torch.tensor([4.0], device=self.device, dtype=dtype)
            likelihood = GaussianLikelihood()
            model = ExactGP(train_x, train_y, likelihood)
            model.covar_module = RBFKernel(ard_num_dims=3)
            model.mean_module = ConstantMean()
            model.to(device=self.device, dtype=dtype)
            mll = ExactMarginalLogLikelihood(likelihood, model)
            # get parameters
            x, pdict, bounds = module_to_array(module=mll)

            # Set parameters
            mll = set_params_with_array(mll, np.array([1.0, 2.0, 3.0, 4.0, 5.0]), pdict)
            z = dict(mll.named_parameters())
            self.assertTrue(
                torch.equal(
                    z["likelihood.noise_covar.raw_noise"],
                    torch.tensor([1.0], device=self.device, dtype=dtype),
                )
            )
            self.assertTrue(
                torch.equal(
                    z["model.covar_module.raw_lengthscale"],
                    torch.tensor([[2.0, 3.0, 4.0]], device=self.device, dtype=dtype),
                )
            )
            self.assertTrue(
                torch.equal(
                    z["model.mean_module.constant"],
                    torch.tensor([5.0], device=self.device, dtype=dtype),
                )
            )

            # Extract again
            x2, pdict2, bounds2 = module_to_array(module=mll)
            self.assertTrue(np.array_equal(x2, np.array([1.0, 2.0, 3.0, 4.0, 5.0])))
