#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import warnings
from itertools import product
from unittest import mock

import torch
from botorch import fit_gpytorch_model, settings
from botorch.exceptions.warnings import BotorchWarning, OptimizationWarning
from botorch.models import FixedNoiseGP, HeteroskedasticSingleTaskGP, SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim.fit import (
    fit_gpytorch_scipy,
    fit_gpytorch_torch,
    OptimizationIteration,
)
from botorch.utils.testing import BotorchTestCase
from gpytorch.constraints import GreaterThan, Interval
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from gpytorch.utils.errors import NanError, NotPSDError
from gpytorch.utils.warnings import NumericalWarning
from scipy.optimize import OptimizeResult


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

    def _getBatchedModel(
        self, kind="SingleTaskGP", double=False, outcome_transform=False
    ):
        dtype = torch.double if double else torch.float
        train_x = torch.linspace(0, 1, 10, device=self.device, dtype=dtype).unsqueeze(
            -1
        )
        noise = torch.tensor(NOISE, device=self.device, dtype=dtype)
        train_y1 = torch.sin(train_x * (2 * math.pi)) + noise
        train_y2 = torch.sin(train_x * (2 * math.pi)) + noise
        train_y = torch.cat([train_y1, train_y2], dim=-1)
        kwargs = {}
        if outcome_transform:
            kwargs["outcome_transform"] = Standardize(m=2)
        if kind == "SingleTaskGP":
            model = SingleTaskGP(train_x, train_y, **kwargs)
        elif kind == "FixedNoiseGP":
            model = FixedNoiseGP(
                train_x, train_y, 0.1 * torch.ones_like(train_y), **kwargs
            )
        elif kind == "HeteroskedasticSingleTaskGP":
            model = HeteroskedasticSingleTaskGP(
                train_x, train_y, 0.1 * torch.ones_like(train_y), **kwargs
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
                    self.assertTrue(
                        any(issubclass(w.category, OptimizationWarning)) for w in ws
                    )
                    self.assertFalse(any(MAX_RETRY_MSG in str(w.message) for w in ws))
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
            mll = fit_gpytorch_model(
                mll,
                optimizer=optimizer,
                options=options,
                max_retries=1,
                bounds={"likelihood.noise_covar.raw_noise": (1e-1, None)},
            )

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
                    self.assertEqual(
                        sum(1 for w in ws if MAX_ITER_MSG in str(w.message)), 1
                    )
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
                self.assertGreaterEqual(len(ws), 1)
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
            mll = fit_gpytorch_model(
                mll, optimizer=optimizer, options=options, max_retries=1
            )
            self.assertTrue(mll.dummy_param.grad is None)

            # test excluding a parameter
            mll = self._getModel(double=double)
            original_raw_noise = mll.model.likelihood.noise_covar.raw_noise.item()
            original_mean_module_constant = mll.model.mean_module.constant.item()
            options["exclude"] = [
                "model.mean_module.constant",
                "likelihood.noise_covar.raw_noise",
            ]
            mll = fit_gpytorch_model(
                mll, optimizer=optimizer, options=options, max_retries=1
            )
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
            mll = fit_gpytorch_model(
                mll,
                optimizer=optimizer,
                options=options,
                max_retries=1,
                approx_mll=is_scipy,
            )
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
            X_train = torch.ones(2, 2, device=self.device, dtype=dtype)
            Y_train = torch.zeros(2, 1, device=self.device, dtype=dtype)
            test_likelihood = GaussianLikelihood(
                noise_constraint=GreaterThan(-1e-7, transform=None, initial_value=0.0)
            )
            gp = SingleTaskGP(X_train, Y_train, likelihood=test_likelihood)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            mll.to(device=self.device, dtype=dtype)
            # this will do multiple retries (and emit warnings, which is desired)
            with warnings.catch_warnings(record=True) as ws, settings.debug(True):
                fit_gpytorch_model(mll, options=options, max_retries=2)
                self.assertTrue(
                    any(issubclass(w.category, NumericalWarning) for w in ws)
                )
            # ensure that we fail if noise ensures that jitter does not help
            gp.likelihood = GaussianLikelihood(
                noise_constraint=Interval(-2, -1, transform=None, initial_value=-1.5)
            )
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            mll.to(device=self.device, dtype=dtype)
            with self.assertLogs(level="DEBUG") as logs:
                fit_gpytorch_model(mll, options=options, max_retries=2)
            self.assertTrue(any("NotPSDError" in log for log in logs.output))
            # ensure we can handle NaNErrors in the optimizer
            with mock.patch.object(SingleTaskGP, "__call__", side_effect=NanError):
                gp = SingleTaskGP(X_train, Y_train, likelihood=test_likelihood)
                mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
                mll.to(device=self.device, dtype=dtype)
                fit_gpytorch_model(
                    mll, options={"disp": False, "maxiter": 1}, max_retries=1
                )
            # ensure we catch NotPSDErrors
            with mock.patch.object(SingleTaskGP, "__call__", side_effect=NotPSDError):
                mll = self._getModel()
                with self.assertLogs(level="DEBUG") as logs:
                    fit_gpytorch_model(mll, max_retries=2)
                for retry in [1, 2]:
                    self.assertTrue(
                        any(
                            f"Fitting failed on try {retry} due to a NotPSDError."
                            in log
                            for log in logs.output
                        )
                    )

            # Failure due to optimization warning

            def optimize_w_warning(mll, **kwargs):
                warnings.warn("Dummy warning.", OptimizationWarning)
                return mll, None

            mll = self._getModel()
            with self.assertLogs(level="DEBUG") as logs, settings.debug(True):
                fit_gpytorch_model(mll, optimizer=optimize_w_warning, max_retries=2)
            self.assertTrue(
                any("Fitting failed on try 1." in log for log in logs.output)
            )

    def test_fit_gpytorch_model_torch(self):
        self.test_fit_gpytorch_model(optimizer=fit_gpytorch_torch)

    def test_fit_gpytorch_model_sequential(self):
        options = {"disp": False, "maxiter": 1}
        for double, kind, outcome_transform in product(
            (False, True),
            ("SingleTaskGP", "FixedNoiseGP", "HeteroskedasticSingleTaskGP"),
            (False, True),
        ):
            with warnings.catch_warnings(record=True) as ws, settings.debug(True):
                mll = self._getBatchedModel(
                    kind=kind, double=double, outcome_transform=outcome_transform
                )
                mll = fit_gpytorch_model(mll, options=options, max_retries=1)
                mll = self._getBatchedModel(
                    kind=kind, double=double, outcome_transform=outcome_transform
                )
                mll = fit_gpytorch_model(
                    mll, options=options, sequential=True, max_retries=1
                )
                mll = self._getBatchedModel(
                    kind=kind, double=double, outcome_transform=outcome_transform
                )
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

    def test_fit_w_maxiter(self):
        options = {"maxiter": 1}
        with warnings.catch_warnings(record=True) as ws, settings.debug(True):
            mll = self._getModel()
            fit_gpytorch_model(mll, options=options, max_retries=3)
            mll = self._getBatchedModel()
            fit_gpytorch_model(mll, options=options, max_retries=3)
        self.assertFalse(any("ITERATIONS REACHED LIMIT" in str(w.message) for w in ws))

    def test_warnings_on_failed_fit(self):
        mll = self._getModel()
        mock_res = OptimizeResult(
            x=[0.00836791, -0.01646641, 0.33771477, -0.9502073],
            success=True,
            status=0,
            fun=1.0,
            jac=None,
            nfev=40,
            njev=40,
            nhev=40,
            nit=20,
            message="CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH".encode(),
        )

        # Should not get errors here, so single call.
        with mock.patch(
            f"{fit_gpytorch_scipy.__module__}.minimize", return_value=mock_res
        ) as mock_minimize:
            fit_gpytorch_model(mll, max_retries=3)
        mock_minimize.assert_called_once()

        # The following should use all 3 tries due to OptimizationWarning.
        mock_res.fun = float("nan")
        mock_res.message = "ABNORMAL_TERMINATION_IN_LNSRCH".encode()
        mock_res.success = False
        with mock.patch(
            f"{fit_gpytorch_scipy.__module__}.minimize", return_value=mock_res
        ) as mock_minimize:
            fit_gpytorch_model(mll, max_retries=3)
        self.assertEqual(mock_minimize.call_count, 3)

        # Check that it works with SumMarginalLogLikelihood.
        # This should use 6 tries, 3 for each sub-model.
        model = ModelListGP(mll.model, mll.model)
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        with mock.patch(
            f"{fit_gpytorch_scipy.__module__}.minimize", return_value=mock_res
        ) as mock_minimize:
            fit_gpytorch_model(mll, max_retries=3)
        self.assertEqual(mock_minimize.call_count, 6)
