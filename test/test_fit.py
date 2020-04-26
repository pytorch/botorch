#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import warnings

import torch
from botorch import fit_gpytorch_model, settings
from botorch.exceptions.warnings import BotorchWarning, OptimizationWarning
from botorch.models import FixedNoiseGP, HeteroskedasticSingleTaskGP, SingleTaskGP
from botorch.optim.fit import (
    OptimizationIteration,
    fit_gpytorch_scipy,
    fit_gpytorch_torch,
)
from botorch.utils.testing import BotorchTestCase
from gpytorch.constraints import GreaterThan
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood


NOISE = [
    [0.127],
    [-0.113],
    [-0.345],
    [-0.034],
    [-0.069],
    [-0.272],
    [0.013],
    [0.056],
    [0.087],
    [-0.081],
]

MAX_ITER_MSG = "TOTAL NO. of ITERATIONS REACHED LIMIT"
MAX_RETRY_MSG = "Fitting failed on all retries."


class TestFitGPyTorchModel(BotorchTestCase):
    def _getModel(self, double=False):
        dtype = torch.double if double else torch.float
        train_x = torch.linspace(0, 1, 10, device=self.device, dtype=dtype).unsqueeze(
            -1
        )
        noise = torch.tensor(NOISE, device=self.device, dtype=dtype)
        train_y = torch.sin(train_x * (2 * math.pi)) + noise
        model = SingleTaskGP(train_x, train_y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        return mll.to(device=self.device, dtype=dtype)

    def _getBatchedModel(self, kind="SingleTaskGP", double=False):
        dtype = torch.double if double else torch.float
        train_x = torch.linspace(0, 1, 10, device=self.device, dtype=dtype).unsqueeze(
            -1
        )
        noise = torch.tensor(NOISE, device=self.device, dtype=dtype)
        train_y1 = torch.sin(train_x * (2 * math.pi)) + noise
        train_y2 = torch.sin(train_x * (2 * math.pi)) + noise
        train_y = torch.cat([train_y1, train_y2], dim=-1)
        if kind == "SingleTaskGP":
            model = SingleTaskGP(train_x, train_y)
        elif kind == "FixedNoiseGP":
            model = FixedNoiseGP(train_x, train_y, 0.1 * torch.ones_like(train_y))
        elif kind == "HeteroskedasticSingleTaskGP":
            model = HeteroskedasticSingleTaskGP(
                train_x, train_y, 0.1 * torch.ones_like(train_y)
            )
        else:
            raise NotImplementedError
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        return mll.to(device=self.device, dtype=dtype)

    def test_fit_gpytorch_model(self, optimizer=fit_gpytorch_scipy):
        options = {"disp": False, "maxiter": 5}
        for double in (False, True):
            mll = self._getModel(double=double)
            with warnings.catch_warnings(record=True) as ws, settings.debug(True):
                mll = fit_gpytorch_model(
                    mll, optimizer=optimizer, options=options, max_retries=1
                )
                if optimizer == fit_gpytorch_scipy:
                    self.assertEqual(len(ws), 1)
                    self.assertTrue(MAX_RETRY_MSG in str(ws[0].message))
            model = mll.model
            # Make sure all of the parameters changed
            self.assertGreater(model.likelihood.raw_noise.abs().item(), 1e-3)
            self.assertLess(model.mean_module.constant.abs().item(), 0.1)
            self.assertGreater(
                model.covar_module.base_kernel.raw_lengthscale.abs().item(), 0.1
            )
            self.assertGreater(model.covar_module.raw_outputscale.abs().item(), 1e-3)

            # test overriding the default bounds with user supplied bounds
            mll = self._getModel(double=double)
            with warnings.catch_warnings(record=True) as ws, settings.debug(True):
                mll = fit_gpytorch_model(
                    mll,
                    optimizer=optimizer,
                    options=options,
                    max_retries=1,
                    bounds={"likelihood.noise_covar.raw_noise": (1e-1, None)},
                )
                if optimizer == fit_gpytorch_scipy:
                    self.assertEqual(len(ws), 1)
                    self.assertTrue(MAX_RETRY_MSG in str(ws[0].message))

            model = mll.model
            self.assertGreaterEqual(model.likelihood.raw_noise.abs().item(), 1e-1)
            self.assertLess(model.mean_module.constant.abs().item(), 0.1)
            self.assertGreater(
                model.covar_module.base_kernel.raw_lengthscale.abs().item(), 0.1
            )
            self.assertGreater(model.covar_module.raw_outputscale.abs().item(), 1e-3)

            # test tracking iterations
            mll = self._getModel(double=double)
            if optimizer is fit_gpytorch_torch:
                options["disp"] = True
            with warnings.catch_warnings(record=True) as ws, settings.debug(True):
                mll, info_dict = optimizer(mll, options=options, track_iterations=True)
                if optimizer == fit_gpytorch_scipy:
                    self.assertEqual(len(ws), 1)
                    self.assertTrue(MAX_ITER_MSG in str(ws[0].message))
            self.assertEqual(len(info_dict["iterations"]), options["maxiter"])
            self.assertIsInstance(info_dict["iterations"][0], OptimizationIteration)
            self.assertTrue("fopt" in info_dict)
            self.assertTrue("wall_time" in info_dict)

            # Test different optimizer, for scipy optimizer,
            # because of different scipy OptimizeResult.message type
            if optimizer == fit_gpytorch_scipy:
                with warnings.catch_warnings(record=True) as ws, settings.debug(True):
                    mll, info_dict = optimizer(
                        mll, options=options, track_iterations=False, method="slsqp"
                    )
                self.assertEqual(len(ws), 1)
                self.assertEqual(len(info_dict["iterations"]), 0)
                self.assertTrue("fopt" in info_dict)
                self.assertTrue("wall_time" in info_dict)

            # test extra param that does not affect loss
            options["disp"] = False
            mll = self._getModel(double=double)
            mll.register_parameter(
                "dummy_param",
                torch.nn.Parameter(
                    torch.tensor(
                        [5.0],
                        dtype=torch.double if double else torch.float,
                        device=self.device,
                    )
                ),
            )
            with warnings.catch_warnings(record=True) as ws, settings.debug(True):
                mll = fit_gpytorch_model(
                    mll, optimizer=optimizer, options=options, max_retries=1
                )
                if optimizer == fit_gpytorch_scipy:
                    self.assertEqual(len(ws), 1)
                    self.assertTrue(MAX_RETRY_MSG in str(ws[0].message))
            self.assertTrue(mll.dummy_param.grad is None)

            # test excluding a parameter
            mll = self._getModel(double=double)
            original_raw_noise = mll.model.likelihood.noise_covar.raw_noise.item()
            original_mean_module_constant = mll.model.mean_module.constant.item()
            options["exclude"] = [
                "model.mean_module.constant",
                "likelihood.noise_covar.raw_noise",
            ]
            with warnings.catch_warnings(record=True) as ws, settings.debug(True):
                mll = fit_gpytorch_model(
                    mll, optimizer=optimizer, options=options, max_retries=1
                )
                if optimizer == fit_gpytorch_scipy:
                    self.assertEqual(len(ws), 1)
                    self.assertTrue(MAX_RETRY_MSG in str(ws[0].message))
            model = mll.model
            # Make excluded params did not change
            self.assertEqual(
                model.likelihood.noise_covar.raw_noise.item(), original_raw_noise
            )
            self.assertEqual(
                model.mean_module.constant.item(), original_mean_module_constant
            )
            # Make sure other params did change
            self.assertGreater(
                model.covar_module.base_kernel.raw_lengthscale.abs().item(), 0.1
            )
            self.assertGreater(model.covar_module.raw_outputscale.abs().item(), 1e-3)

            # test non-default setting for approximate MLL computation
            is_scipy = optimizer == fit_gpytorch_scipy
            mll = self._getModel(double=double)
            with warnings.catch_warnings(record=True) as ws, settings.debug(True):
                mll = fit_gpytorch_model(
                    mll,
                    optimizer=optimizer,
                    options=options,
                    max_retries=1,
                    approx_mll=is_scipy,
                )
                if is_scipy:
                    self.assertEqual(len(ws), 1)
                    self.assertTrue(MAX_RETRY_MSG in str(ws[0].message))
            model = mll.model
            # Make sure all of the parameters changed
            self.assertGreater(model.likelihood.raw_noise.abs().item(), 1e-3)
            self.assertLess(model.mean_module.constant.abs().item(), 0.1)
            self.assertGreater(
                model.covar_module.base_kernel.raw_lengthscale.abs().item(), 0.1
            )
            self.assertGreater(model.covar_module.raw_outputscale.abs().item(), 1e-3)

    def test_fit_gpytorch_model_singular(self):
        options = {"disp": False, "maxiter": 5}
        for dtype in (torch.float, torch.double):
            X_train = torch.rand(2, 2, device=self.device, dtype=dtype)
            Y_train = torch.zeros(2, 1, device=self.device, dtype=dtype)
            test_likelihood = GaussianLikelihood(
                noise_constraint=GreaterThan(-1.0, transform=None, initial_value=0.0)
            )
            gp = SingleTaskGP(X_train, Y_train, likelihood=test_likelihood)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            mll.to(device=self.device, dtype=dtype)
            # this will do multiple retries (and emit warnings, which is desired)
            with warnings.catch_warnings(record=True) as ws, settings.debug(True):
                fit_gpytorch_model(mll, options=options, max_retries=2)
                self.assertTrue(
                    any(issubclass(w.category, OptimizationWarning) for w in ws)
                )

    def test_fit_gpytorch_model_torch(self):
        self.test_fit_gpytorch_model(optimizer=fit_gpytorch_torch)

    def test_fit_gpytorch_model_sequential(self):
        options = {"disp": False, "maxiter": 1}
        for double in (False, True):
            for kind in ("SingleTaskGP", "FixedNoiseGP", "HeteroskedasticSingleTaskGP"):
                with warnings.catch_warnings(record=True) as ws, settings.debug(True):
                    mll = self._getBatchedModel(kind=kind, double=double)
                    mll = fit_gpytorch_model(mll, options=options, max_retries=1)
                    mll = self._getBatchedModel(kind=kind, double=double)
                    mll = fit_gpytorch_model(
                        mll, options=options, sequential=True, max_retries=1
                    )
                    mll = self._getBatchedModel(kind=kind, double=double)
                    mll = fit_gpytorch_model(
                        mll, options=options, sequential=False, max_retries=1
                    )
                    if kind == "HeteroskedasticSingleTaskGP":
                        self.assertTrue(
                            any(issubclass(w.category, BotorchWarning) for w in ws)
                        )
                        self.assertTrue(
                            any(
                                "Failed to convert ModelList to batched model"
                                in str(w.message)
                                for w in ws
                            )
                        )
