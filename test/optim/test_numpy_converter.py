#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from math import pi
from unittest.mock import MagicMock, patch
from warnings import catch_warnings, simplefilter

import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.optim import numpy_converter
from botorch.optim.numpy_converter import (
    _scipy_objective_and_grad,
    module_to_array,
    set_params_with_array,
)
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
            with catch_warnings():
                simplefilter("ignore", category=DeprecationWarning)
                x, pdict, bounds = module_to_array(module=mll)
            self.assertTrue(np.array_equal(x, np.zeros(5)))
            expected_sizes = {
                "likelihood.noise_covar.raw_noise": torch.Size([1]),
                "model.covar_module.raw_lengthscale": torch.Size([1, 3]),
                "model.mean_module.raw_constant": torch.Size(),
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
            with catch_warnings():
                simplefilter("ignore", category=DeprecationWarning)
                x, pdict, bounds = module_to_array(
                    module=mll, exclude={"model.mean_module.raw_constant"}
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
            with catch_warnings():
                simplefilter("ignore", category=DeprecationWarning)
                x, pdict, bounds = module_to_array(
                    module=mll,
                    bounds={"model.covar_module.raw_lengthscale": (0.1, None)},
                )
            self.assertTrue(np.array_equal(x, np.zeros(5)))
            expected_sizes = {
                "likelihood.noise_covar.raw_noise": torch.Size([1]),
                "model.covar_module.raw_lengthscale": torch.Size([1, 3]),
                "model.mean_module.raw_constant": torch.Size(),
            }
            self.assertEqual(set(pdict.keys()), set(expected_sizes.keys()))
            for pname, val in pdict.items():
                self.assertEqual(val.dtype, dtype)
                self.assertEqual(val.shape, expected_sizes[pname])
                self.assertEqual(val.device.type, self.device.type)
            lower_exp = np.full_like(x, 0.1)
            for p in (
                "likelihood.noise_covar.raw_noise",
                "model.mean_module.raw_constant",
            ):
                lower_exp[_get_index(pdict, p)] = -np.inf
            self.assertTrue(np.equal(bounds[0], lower_exp).all())
            self.assertTrue(np.equal(bounds[1], np.full_like(x, np.inf)).all())

            with catch_warnings():
                simplefilter("ignore", category=DeprecationWarning)
                x, pdict, bounds = module_to_array(
                    module=mll,
                    bounds={
                        key: (-float("inf"), float("inf"))
                        for key, _ in mll.named_parameters()
                    },
                )
            self.assertIsNone(bounds)

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
            with catch_warnings():
                simplefilter("ignore", category=DeprecationWarning)
                x, pdict, bounds = module_to_array(
                    module=mll,
                    bounds={"model.covar_module.raw_lengthscale": (0.1, None)},
                )
            self.assertTrue(np.array_equal(x, np.zeros(5)))
            expected_sizes = {
                "likelihood.noise_covar.raw_noise": torch.Size([1]),
                "model.covar_module.raw_lengthscale": torch.Size([1, 3]),
                "model.mean_module.raw_constant": torch.Size(),
            }
            self.assertEqual(set(pdict.keys()), set(expected_sizes.keys()))
            for pname, val in pdict.items():
                self.assertEqual(val.dtype, dtype)
                self.assertEqual(val.shape, expected_sizes[pname])
                self.assertEqual(val.device.type, self.device.type)
            lower_exp = np.full_like(x, 0.1)
            lower_exp[_get_index(pdict, "model.mean_module.raw_constant")] = -np.inf
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

            with catch_warnings():
                # Get parameters
                simplefilter("ignore", category=DeprecationWarning)
                x, pdict, bounds = module_to_array(module=mll)

                # Set parameters
                mll = set_params_with_array(
                    mll, np.array([1.0, 2.0, 3.0, 4.0, 5.0]), pdict
                )
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
                    z["model.mean_module.raw_constant"],
                    torch.tensor(5.0, device=self.device, dtype=dtype),
                )
            )

            # Extract again
            with catch_warnings():
                simplefilter("ignore", category=DeprecationWarning)
                x2, pdict2, bounds2 = module_to_array(module=mll)
            self.assertTrue(np.array_equal(x2, np.array([1.0, 2.0, 3.0, 4.0, 5.0])))


class TestScipyObjectiveAndGrad(BotorchTestCase):
    def setUp(self):
        with torch.random.fork_rng():
            torch.manual_seed(0)
            train_X = torch.linspace(0, 1, 10).unsqueeze(-1)
            train_Y = torch.sin((2 * pi) * train_X)
            train_Y = train_Y + 0.1 * torch.randn_like(train_Y)

        model = SingleTaskGP(train_X=train_X, train_Y=train_Y)
        self.mll = ExactMarginalLogLikelihood(model.likelihood, model)

    def test_scipy_objective_and_grad(self):
        with catch_warnings():
            simplefilter("ignore", category=DeprecationWarning)
            x, property_dict, bounds = module_to_array(module=self.mll)
            loss, grad = _scipy_objective_and_grad(x, self.mll, property_dict)

        _dist = self.mll.model(*self.mll.model.train_inputs)
        _loss = -self.mll(_dist, self.mll.model.train_targets)
        _loss.sum().backward()
        _grad = torch.concat(
            [self.mll.get_parameter(name).grad.view(-1) for name in property_dict]
        )
        self.assertEqual(loss, _loss.detach().sum().item())
        self.assertTrue(np.allclose(grad, _grad.detach().numpy()))

        def _getter(*args, **kwargs):
            raise RuntimeError("foo")

        _handler = MagicMock()

        with catch_warnings(), patch.multiple(
            numpy_converter,
            _get_extra_mll_args=_getter,
            _handle_numerical_errors=_handler,
        ):
            simplefilter("ignore", category=DeprecationWarning)
            _scipy_objective_and_grad(x, self.mll, property_dict)
        self.assertEqual(_handler.call_count, 1)
