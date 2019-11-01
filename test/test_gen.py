#! /usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import warnings

import torch
from botorch.acquisition import qExpectedImprovement
from botorch.exceptions.warnings import OptimizationWarning
from botorch.fit import fit_gpytorch_model
from botorch.gen import gen_candidates_scipy, gen_candidates_torch
from botorch.models import SingleTaskGP
from botorch.utils.testing import BotorchTestCase
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood

from .test_fit import NOISE


EPS = 1e-8


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
    def test_gen_candidates(self, gen_candidates=gen_candidates_scipy):
        for double in (True, False):
            self._setUp(double=double)
            qEI = qExpectedImprovement(self.model, best_f=self.f_best)
            candidates, _ = gen_candidates(
                initial_conditions=self.initial_conditions,
                acquisition_function=qEI,
                lower_bounds=0,
                upper_bounds=1,
            )
            self.assertTrue(-EPS <= candidates <= 1 + EPS)

    def test_gen_candidates_torch(self):
        self.test_gen_candidates(gen_candidates=gen_candidates_torch)

    def test_gen_candidates_with_none_fixed_features(
        self, gen_candidates=gen_candidates_scipy
    ):
        for double in (True, False):
            self._setUp(double=double, expand=True)
            qEI = qExpectedImprovement(self.model, best_f=self.f_best)
            candidates, _ = gen_candidates(
                initial_conditions=self.initial_conditions,
                acquisition_function=qEI,
                lower_bounds=0,
                upper_bounds=1,
                fixed_features={1: None},
            )
            candidates = candidates.squeeze(0)
            self.assertTrue(-EPS <= candidates[0] <= 1 + EPS)
            self.assertTrue(candidates[1].item() == 1.0)

    def test_gen_candidates_torch_with_none_fixed_features(self):
        self.test_gen_candidates_with_none_fixed_features(
            gen_candidates=gen_candidates_torch
        )

    def test_gen_candidates_with_fixed_features(
        self, gen_candidates=gen_candidates_scipy
    ):
        for double in (True, False):
            self._setUp(double=double, expand=True)
            qEI = qExpectedImprovement(self.model, best_f=self.f_best)
            candidates, _ = gen_candidates(
                initial_conditions=self.initial_conditions,
                acquisition_function=qEI,
                lower_bounds=0,
                upper_bounds=1,
                fixed_features={1: 0.25},
            )
            candidates = candidates.squeeze(0)
            self.assertTrue(-EPS <= candidates[0] <= 1 + EPS)
            self.assertTrue(candidates[1].item() == 0.25)

    def test_gen_candidates_torch_with_fixed_features(self):
        self.test_gen_candidates_with_fixed_features(
            gen_candidates=gen_candidates_torch
        )
