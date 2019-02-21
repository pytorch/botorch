#! /usr/bin/env python3

import math
import unittest

import torch
from botorch import fit_model, gen_candidates_scipy
from botorch.acquisition import qExpectedImprovement
from botorch.models import SingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood

from .test_fit import NOISE


EPS = 1e-8


class TestBaseCandidateGeneration(unittest.TestCase):
    def _setUp(self, double=False, cuda=False, expand=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        dtype = torch.double if double else torch.float
        train_x = torch.linspace(0, 1, 10, device=device, dtype=dtype)
        train_y = torch.sin(train_x * (2 * math.pi))
        noise = torch.tensor(NOISE, device=device, dtype=dtype)
        self.train_x = train_x.unsqueeze(-1)
        self.train_y = train_y + noise

        if expand:
            self.train_x = self.train_x.expand(-1, 2)
            self.initial_candidates = torch.tensor(
                [[0.5, 1.0]], device=device, dtype=dtype
            )
        else:
            self.initial_candidates = torch.tensor([[0.5]], device=device, dtype=dtype)
        self.f_best = self.train_y.max().item()
        model = SingleTaskGP(self.train_x, self.train_y)
        self.model = model.to(device=device, dtype=dtype)
        self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        self.mll = fit_model(self.mll, options={"maxiter": 1})


class TestGenCandidates(TestBaseCandidateGeneration):
    def test_gen_candidates_scipy(self, cuda=False):
        for double in (True, False):
            self._setUp(double=double, cuda=cuda)
            qEI = qExpectedImprovement(self.model, best_f=self.f_best)
            candidates, _ = gen_candidates_scipy(
                initial_candidates=self.initial_candidates,
                acquisition_function=qEI,
                lower_bounds=0,
                upper_bounds=1,
            )
            self.assertTrue(-EPS <= candidates <= 1 + EPS)

    def test_gen_candidates_scipy_cuda(self):
        if torch.cuda.is_available():
            self.test_gen_candidates_scipy(cuda=True)

    def test_gen_candidates_scipy_with_none_fixed_features(self, cuda=False):
        for double in (True, False):
            self._setUp(double=double, cuda=cuda, expand=True)
            qEI = qExpectedImprovement(self.model, best_f=self.f_best)
            candidates, _ = gen_candidates_scipy(
                initial_candidates=self.initial_candidates,
                acquisition_function=qEI,
                lower_bounds=0,
                upper_bounds=1,
                fixed_features={1: None},
            )
            candidates = candidates.squeeze(0)
            self.assertTrue(-EPS <= candidates[0] <= 1 + EPS)
            self.assertTrue(candidates[1].item() == 1.0)

    def test_gen_candidates_scipy_with_none_fixed_features_cuda(self):
        if torch.cuda.is_available():
            self.test_gen_candidates_scipy_with_none_fixed_features(cuda=True)

    def test_gen_candidates_scipy_with_fixed_features(self, cuda=False):
        for double in (True, False):
            self._setUp(double=double, cuda=cuda, expand=True)
            qEI = qExpectedImprovement(self.model, best_f=self.f_best)
            candidates, _ = gen_candidates_scipy(
                initial_candidates=self.initial_candidates,
                acquisition_function=qEI,
                lower_bounds=0,
                upper_bounds=1,
                fixed_features={1: 0.25},
            )
            candidates = candidates.squeeze(0)
            self.assertTrue(-EPS <= candidates[0] <= 1 + EPS)
            self.assertTrue(candidates[1].item() == 0.25)

    def test_gen_candidates_scipy_with_fixed_features_cuda(self, cuda=False):
        if torch.cuda.is_available():
            self.test_gen_candidates_scipy_with_fixed_features(cuda=True)
