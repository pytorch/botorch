#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from re import compile
from unittest.mock import patch
from warnings import catch_warnings, warn

import torch
from botorch.exceptions.warnings import OptimizationWarning
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.optim import fit
from botorch.optim.fit import OptimizationIteration
from botorch.optim.utils import state_rollback_ctx
from botorch.settings import debug
from botorch.utils.testing import BotorchTestCase
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from scipy.optimize import OptimizeResult


class TestFitGPyTorchScipy(BotorchTestCase):
    def setUp(self):
        self.mlls = {}
        with torch.random.fork_rng():
            torch.manual_seed(0)
            train_X = torch.linspace(0, 1, 10).unsqueeze(-1)
            train_Y = torch.sin((2 * math.pi) * train_X)
            train_Y = train_Y + 0.1 * torch.randn_like(train_Y)

        model = SingleTaskGP(
            train_X=train_X,
            train_Y=train_Y,
            input_transform=Normalize(d=1),
            outcome_transform=Standardize(m=1),
        )
        self.mlls[SingleTaskGP, 1] = ExactMarginalLogLikelihood(model.likelihood, model)

    def test_fit_gpytorch_scipy(self):
        for mll in self.mlls.values():
            for dtype in (torch.float32, torch.float64):
                self._test_fit_gpytorch_scipy(mll.to(dtype=dtype))

    def _test_fit_gpytorch_scipy(self, mll):
        options = {"disp": False, "maxiter": 3, "maxfun": 2}
        ckpt = {k: (v.detach().clone(), {}) for k, v in mll.state_dict().items()}
        with self.subTest("main"), state_rollback_ctx(mll, checkpoint=ckpt):
            with catch_warnings(record=True) as ws, debug(True):
                _, info_dict = fit.fit_gpytorch_scipy(
                    mll, track_iterations=True, options=options
                )

            # Test only parameters requiring gradients have changed
            self.assertTrue(
                all(
                    param.equal(ckpt[name][0]) != param.requires_grad
                    for name, param in mll.named_parameters()
                )
            )

            # Test maxiter warning message
            self.assertTrue(any("TOTAL NO. of" in str(w.message) for w in ws))
            self.assertTrue(
                any(issubclass(w.category, OptimizationWarning) for w in ws)
            )

            # Test iteration tracking
            self.assertLessEqual(len(info_dict["iterations"]), options["maxiter"])
            self.assertIsInstance(info_dict["iterations"][0], OptimizationIteration)
            self.assertTrue("fopt" in info_dict)
            self.assertTrue("wall_time" in info_dict)
            self.assertEqual(sum(1 for w in ws if "TOTAL NO. of" in str(w.message)), 1)

        # Test that user provided bounds and `exclude` argument are respected
        exclude = "model.mean_module.constant", compile("raw_lengthscale$")
        with self.subTest("bounds"), state_rollback_ctx(mll, checkpoint=ckpt):
            fit.fit_gpytorch_scipy(
                mll,
                bounds={"likelihood.noise_covar.raw_noise": (123, 456)},
                options={**options, "exclude": exclude},
            )

            self.assertTrue(
                mll.likelihood.noise_covar.raw_noise >= 123
                and mll.likelihood.noise_covar.raw_noise <= 456
            )

            for name, param in mll.named_parameters():
                if (
                    name
                    in (
                        "model.mean_module.constant",
                        "model.covar_module.base_kernel.raw_lengthscale",
                    )
                    or not param.requires_grad
                ):
                    self.assertTrue(param.equal(ckpt[name][0]))
                else:
                    self.assertFalse(param.equal(ckpt[name][0]))

        # Test use of `approx_mll` flag
        with self.subTest("approx_mll"), state_rollback_ctx(mll, checkpoint=ckpt):
            fit.fit_gpytorch_scipy(mll, approx_mll=True, options=options)
            self.assertTrue(
                all(
                    param.equal(ckpt[name][0]) != param.requires_grad
                    for name, param in mll.named_parameters()
                )
            )

        # Test handling of scipy optimization failures and parameter assignments
        mock_x = []
        assignments = {}
        for name, param in mll.named_parameters():
            if not param.requires_grad:
                continue  # pragma: no cover

            values = assignments[name] = torch.rand_like(param)
            mock_x.append(values.view(-1))

        with state_rollback_ctx(mll, checkpoint=ckpt), patch.object(
            fit, "minimize"
        ) as mock_minimize:
            mock_minimize.return_value = OptimizeResult(
                x=torch.concat(mock_x).tolist(),
                success=False,
                status=0,
                fun=float("nan"),
                jac=None,
                nfev=1,
                njev=1,
                nhev=1,
                nit=1,
                message="ABNORMAL_TERMINATION_IN_LNSRCH".encode(),
            )
            with catch_warnings(record=True) as ws, debug(True):
                fit.fit_gpytorch_scipy(mll, options=options)

            # Test that warning gets raised
            self.assertTrue(
                any("ABNORMAL_TERMINATION_IN_LNSRCH" in str(w.message) for w in ws)
            )

            # Test that parameter values get assigned correctly
            self.assertTrue(
                all(
                    param.equal(assignments[name])
                    for name, param in mll.named_parameters()
                    if param.requires_grad
                )
            )


class TestFitGPyTorchTorch(BotorchTestCase):
    def setUp(self):
        self.mlls = {}
        with torch.random.fork_rng():
            torch.manual_seed(0)
            train_X = torch.linspace(0, 1, 10).unsqueeze(-1)
            train_Y = torch.sin((2 * math.pi) * train_X)
            train_Y = train_Y + 0.1 * torch.randn_like(train_Y)

        model = SingleTaskGP(
            train_X=train_X,
            train_Y=train_Y,
            input_transform=Normalize(d=1),
            outcome_transform=Standardize(m=1),
        )
        self.mlls[SingleTaskGP, 1] = ExactMarginalLogLikelihood(model.likelihood, model)

    def test_fit_gpytorch_torch(self):
        for mll in self.mlls.values():
            for dtype in (torch.float32, torch.float64):
                self._test_fit_gpytorch_torch(mll.to(dtype=dtype))

    def _test_fit_gpytorch_torch(self, mll):
        options = {"disp": False, "maxiter": 3}
        ckpt = {k: (v.detach().clone(), {}) for k, v in mll.state_dict().items()}
        with self.subTest("main"), state_rollback_ctx(mll, checkpoint=ckpt):
            with catch_warnings(record=True) as ws, debug(True):
                _, info_dict = fit.fit_gpytorch_torch(
                    mll, track_iterations=True, options=options
                )

            # Test only parameters requiring gradients have changed
            self.assertTrue(
                all(
                    param.equal(ckpt[name][0]) != param.requires_grad
                    for name, param in mll.named_parameters()
                )
            )

            # Test iteration tracking
            self.assertEqual(len(info_dict["iterations"]), options["maxiter"])
            self.assertIsInstance(info_dict["iterations"][0], OptimizationIteration)
            self.assertTrue("fopt" in info_dict)
            self.assertTrue("wall_time" in info_dict)

        # Test that user provided bounds and `exclude` argument are respected
        exclude = "model.mean_module.constant", compile("raw_lengthscale$")
        with self.subTest("bounds"), state_rollback_ctx(mll, checkpoint=ckpt):
            fit.fit_gpytorch_torch(
                mll,
                bounds={"likelihood.noise_covar.raw_noise": (123, 456)},
                options={**options, "exclude": exclude},
            )

            self.assertTrue(
                mll.likelihood.noise_covar.raw_noise >= 123
                and mll.likelihood.noise_covar.raw_noise <= 456
            )

            for name, param in mll.named_parameters():
                if (
                    name
                    in (
                        "model.mean_module.constant",
                        "model.covar_module.base_kernel.raw_lengthscale",
                    )
                    or not param.requires_grad
                ):
                    self.assertTrue(param.equal(ckpt[name][0]))
                else:
                    self.assertFalse(param.equal(ckpt[name][0]))

        # Test use of `approx_mll` flag
        with self.subTest("approx_mll"), state_rollback_ctx(mll, checkpoint=ckpt):
            fit.fit_gpytorch_torch(mll, approx_mll=True, options=options)
            self.assertTrue(
                all(
                    param.equal(ckpt[name][0]) != param.requires_grad
                    for name, param in mll.named_parameters()
                )
            )

        with patch.object(fit, "print", new=warn), catch_warnings(record=True) as ws:
            fit.fit_gpytorch_torch(mll, options={"disp": True, "maxiter": 11})
        self.assertEqual(len(ws), 2)
