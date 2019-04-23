#! /usr/bin/env python3

import math
import unittest

import torch
from botorch import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.optim.fit import (
    OptimizationIteration,
    fit_gpytorch_scipy,
    fit_gpytorch_torch,
)
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood


NOISE = [0.127, -0.113, -0.345, -0.034, -0.069, -0.272, 0.013, 0.056, 0.087, -0.081]


class TestFitGPyTorchModel(unittest.TestCase):
    def _getModel(self, double=False, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        dtype = torch.double if double else torch.float
        train_x = torch.linspace(0, 1, 10, device=device, dtype=dtype).unsqueeze(-1)
        noise = torch.tensor(NOISE, device=device, dtype=dtype)
        train_y = torch.sin(train_x.view(-1) * (2 * math.pi)) + noise
        model = SingleTaskGP(train_x, train_y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        return mll.to(device=device, dtype=dtype)

    def test_fit_gpytorch_model(self, cuda=False, optimizer=fit_gpytorch_scipy):
        options = {"disp": False, "maxiter": 5}
        for double in (False, True):
            mll = self._getModel(double=double, cuda=cuda)
            mll = fit_gpytorch_model(mll, optimizer=optimizer, options=options)
            model = mll.model
            # Make sure all of the parameters changed
            self.assertGreater(model.likelihood.raw_noise.abs().item(), 1e-3)
            self.assertLess(model.mean_module.constant.abs().item(), 0.1)
            self.assertGreater(
                model.covar_module.base_kernel.raw_lengthscale.abs().item(), 0.1
            )
            self.assertGreater(model.covar_module.raw_outputscale.abs().item(), 1e-3)

            # test overriding the default bounds with user supplied bounds
            mll = self._getModel(double=double, cuda=cuda)
            mll = fit_gpytorch_model(
                mll,
                optimizer=optimizer,
                options=options,
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
            mll = self._getModel(double=double, cuda=cuda)
            if optimizer is fit_gpytorch_torch:
                options["disp"] = True
            mll, iterations = optimizer(mll, options=options, track_iterations=True)
            self.assertEqual(len(iterations), options["maxiter"])
            self.assertIsInstance(iterations[0], OptimizationIteration)

            # test extra param that does not affect loss
            options["disp"] = False
            mll = self._getModel(double=double, cuda=cuda)
            mll.register_parameter(
                "dummy_param",
                torch.nn.Parameter(
                    torch.tensor(
                        [5.0],
                        dtype=torch.double if double else torch.float,
                        device=torch.device("cuda" if cuda else "cpu"),
                    )
                ),
            )
            mll = fit_gpytorch_model(mll, optimizer=optimizer, options=options)
            self.assertTrue(mll.dummy_param.grad is None)

    def test_fit_gpytorch_model_scipy_cuda(self):
        if torch.cuda.is_available():
            self.test_fit_gpytorch_model(cuda=True)

    def test_fit_gpytorch_model_torch(self, cuda=False):
        self.test_fit_gpytorch_model(cuda=cuda, optimizer=fit_gpytorch_torch)

    def test_fit_gpytorch_model_torch_cuda(self):
        if torch.cuda.is_available():
            self.test_fit_gpytorch_model_torch(cuda=True)
