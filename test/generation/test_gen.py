#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import warnings
from unittest import mock

import torch
from botorch.acquisition import qExpectedImprovement
from botorch.exceptions.warnings import OptimizationWarning
from botorch.fit import fit_gpytorch_model
from botorch.generation.gen import (
    gen_candidates_scipy,
    gen_candidates_torch,
    get_best_candidates,
)
from botorch.models import SingleTaskGP
from botorch.utils.testing import BotorchTestCase
from gpytorch import settings as gpt_settings
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood


EPS = 1e-8

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


class TestBaseCandidateGeneration(BotorchTestCase):
    def _setUp(self, double=False, expand=False):
        dtype = torch.double if double else torch.float
        train_x = torch.linspace(0, 1, 10, device=self.device, dtype=dtype).unsqueeze(
            -1
        )
        train_y = torch.sin(train_x * (2 * math.pi))
        noise = torch.tensor(NOISE, device=self.device, dtype=dtype)
        self.train_x = train_x
        self.train_y = train_y + noise
        if expand:
            self.train_x = self.train_x.expand(-1, 2)
            ics = torch.tensor([[0.5, 1.0]], device=self.device, dtype=dtype)
        else:
            ics = torch.tensor([[0.5]], device=self.device, dtype=dtype)
        self.initial_conditions = ics
        self.f_best = self.train_y.max().item()
        model = SingleTaskGP(self.train_x, self.train_y)
        self.model = model.to(device=self.device, dtype=dtype)
        self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=OptimizationWarning)
            self.mll = fit_gpytorch_model(
                self.mll, options={"maxiter": 1}, max_retries=1
            )


class TestGenCandidates(TestBaseCandidateGeneration):
    def test_gen_candidates(self, gen_candidates=gen_candidates_scipy, options=None):
        options = options or {}
        options = {**options, "maxiter": 5}
        for double in (True, False):
            self._setUp(double=double)
            qEI = qExpectedImprovement(self.model, best_f=self.f_best)
            candidates, _ = gen_candidates(
                initial_conditions=self.initial_conditions,
                acquisition_function=qEI,
                lower_bounds=0,
                upper_bounds=1,
                options=options or {},
            )
            self.assertTrue(-EPS <= candidates <= 1 + EPS)

    def test_gen_candidates_torch(self):
        self.test_gen_candidates(
            gen_candidates=gen_candidates_torch, options={"disp": False}
        )

    def test_gen_candidates_with_none_fixed_features(
        self,
        gen_candidates=gen_candidates_scipy,
        options=None,
    ):
        options = options or {}
        options = {**options, "maxiter": 5}
        for double in (True, False):
            self._setUp(double=double, expand=True)
            qEI = qExpectedImprovement(self.model, best_f=self.f_best)
            candidates, _ = gen_candidates(
                initial_conditions=self.initial_conditions,
                acquisition_function=qEI,
                lower_bounds=0,
                upper_bounds=1,
                fixed_features={1: None},
                options=options or {},
            )
            candidates = candidates.squeeze(0)
            self.assertTrue(-EPS <= candidates[0] <= 1 + EPS)
            self.assertTrue(candidates[1].item() == 1.0)

    def test_gen_candidates_torch_with_none_fixed_features(self):
        self.test_gen_candidates_with_none_fixed_features(
            gen_candidates=gen_candidates_torch, options={"disp": False}
        )

    def test_gen_candidates_with_fixed_features(
        self, gen_candidates=gen_candidates_scipy, options=None
    ):
        options = options or {}
        options = {**options, "maxiter": 5}
        for double in (True, False):
            self._setUp(double=double, expand=True)
            qEI = qExpectedImprovement(self.model, best_f=self.f_best)
            candidates, _ = gen_candidates(
                initial_conditions=self.initial_conditions,
                acquisition_function=qEI,
                lower_bounds=0,
                upper_bounds=1,
                fixed_features={1: 0.25},
                options=options,
            )
            candidates = candidates.squeeze(0)
            self.assertTrue(-EPS <= candidates[0] <= 1 + EPS)
            self.assertTrue(candidates[1].item() == 0.25)

    def test_gen_candidates_scipy_with_fixed_features_inequality_constraints(self):
        options = {"maxiter": 5}
        for double in (True, False):
            self._setUp(double=double, expand=True)
            qEI = qExpectedImprovement(self.model, best_f=self.f_best)
            candidates, _ = gen_candidates_scipy(
                initial_conditions=self.initial_conditions.reshape(1, 1, -1),
                acquisition_function=qEI,
                inequality_constraints=[
                    (torch.tensor([0]), torch.tensor([1]), 0),
                    (torch.tensor([1]), torch.tensor([-1]), -1),
                ],
                fixed_features={1: 0.25},
                options=options,
            )
            # candidates is of dimension 1 x 1 x 2
            # so we are squeezing all the singleton dimensions
            candidates = candidates.squeeze()
            self.assertTrue(-EPS <= candidates[0] <= 1 + EPS)
            self.assertTrue(candidates[1].item() == 0.25)

    def test_gen_candidates_torch_with_fixed_features(self):
        self.test_gen_candidates_with_fixed_features(
            gen_candidates=gen_candidates_torch, options={"disp": False}
        )

    def test_gen_candidates_scipy_nan_handling(self):
        for dtype, expected_regex in [
            (torch.float, "Consider using"),
            (torch.double, "gradient array"),
        ]:
            ckwargs = {"dtype": dtype, "device": self.device}

            test_ics = torch.rand(3, 1, **ckwargs)
            test_grad = torch.tensor([0.5, 0.2, float("nan")], **ckwargs)
            # test NaN in grad
            with mock.patch("torch.autograd.grad", return_value=[test_grad]):
                with self.assertRaisesRegex(RuntimeError, expected_regex):
                    gen_candidates_scipy(
                        initial_conditions=test_ics,
                        acquisition_function=mock.Mock(return_value=test_ics),
                    )

            # test NaN in `x`
            test_ics = torch.tensor([0.0, 0.0, float("nan")], **ckwargs)
            with self.assertRaisesRegex(RuntimeError, "array `x` are NaN."):
                gen_candidates_scipy(
                    initial_conditions=test_ics,
                    acquisition_function=mock.Mock(),
                )


class TestRandomRestartOptimization(TestBaseCandidateGeneration):
    def test_random_restart_optimization(self):
        for double in (True, False):
            self._setUp(double=double)
            with gpt_settings.debug(False):
                best_f = self.model(self.train_x).mean.max().item()
            qEI = qExpectedImprovement(self.model, best_f=best_f)
            bounds = torch.tensor([[0.0], [1.0]]).type_as(self.train_x)
            batch_ics = torch.rand(2, 1).type_as(self.train_x)
            batch_candidates, batch_acq_values = gen_candidates_scipy(
                initial_conditions=batch_ics,
                acquisition_function=qEI,
                lower_bounds=bounds[0],
                upper_bounds=bounds[1],
                options={"maxiter": 3},
            )
            candidates = get_best_candidates(
                batch_candidates=batch_candidates, batch_values=batch_acq_values
            )
            self.assertTrue(-EPS <= candidates <= 1 + EPS)
