#! /usr/bin/env python3

import math
import unittest

import torch
from botorch.acquisition import ExpectedImprovement, qExpectedImprovement
from botorch.fit import fit_gpytorch_model
from botorch.models import FixedNoiseGP, SingleTaskGP
from botorch.optim import joint_optimize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood

from .test_fit import NOISE


EPS = 1e-8


class TestEndToEnd(unittest.TestCase):
    def _setUp(self, double=False, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        dtype = torch.double if double else torch.float
        train_x = torch.linspace(0, 1, 10, device=device, dtype=dtype).unsqueeze(-1)
        train_y = torch.sin(train_x * (2 * math.pi)).squeeze(-1)
        train_yvar = torch.tensor(0.1 ** 2, device=device)
        noise = torch.tensor(NOISE, device=device, dtype=dtype)
        self.train_x = train_x
        self.train_y = train_y + noise
        self.train_yvar = train_yvar
        self.bounds = torch.tensor([[0.0], [1.0]], device=device, dtype=dtype)
        model_st = SingleTaskGP(self.train_x, self.train_y)
        self.model_st = model_st.to(device=device, dtype=dtype)
        self.mll_st = ExactMarginalLogLikelihood(
            self.model_st.likelihood, self.model_st
        )
        self.mll_st = fit_gpytorch_model(self.mll_st, options={"maxiter": 5})
        model_fn = FixedNoiseGP(
            self.train_x, self.train_y, self.train_yvar.expand_as(self.train_y)
        )
        self.model_fn = model_fn.to(device=device, dtype=dtype)
        self.mll_fn = ExactMarginalLogLikelihood(
            self.model_fn.likelihood, self.model_fn
        )
        self.mll_fn = fit_gpytorch_model(self.mll_fn, options={"maxiter": 5})

    def test_qEI(self, cuda=False):
        for double in (True, False):
            self._setUp(double=double, cuda=cuda)
            qEI = qExpectedImprovement(self.model_st, best_f=0.0)
            candidates = joint_optimize(
                acq_function=qEI,
                bounds=self.bounds,
                q=3,
                num_restarts=10,
                raw_samples=20,
                options={"maxiter": 5},
            )
            self.assertTrue(torch.all(-EPS <= candidates))
            self.assertTrue(torch.all(candidates <= 1 + EPS))
            qEI = qExpectedImprovement(self.model_fn, best_f=0.0)
            candidates = joint_optimize(
                acq_function=qEI,
                bounds=self.bounds,
                q=3,
                num_restarts=10,
                raw_samples=20,
                options={"maxiter": 5},
            )
            self.assertTrue(torch.all(-EPS <= candidates))
            self.assertTrue(torch.all(candidates <= 1 + EPS))

    def test_qEI_cuda(self):
        if torch.cuda.is_available():
            self.test_qEI(cuda=True)

    def test_EI(self, cuda=False):
        for double in (True, False):
            self._setUp(double=double, cuda=cuda)
            EI = ExpectedImprovement(self.model_st, best_f=0.0)
            candidates = joint_optimize(
                acq_function=EI,
                bounds=self.bounds,
                q=1,
                num_restarts=10,
                raw_samples=20,
                options={"maxiter": 5},
            )
            self.assertTrue(-EPS <= candidates <= 1 + EPS)
            EI = ExpectedImprovement(self.model_fn, best_f=0.0)
            candidates = joint_optimize(
                acq_function=EI,
                bounds=self.bounds,
                q=1,
                num_restarts=10,
                raw_samples=20,
                options={"maxiter": 5},
            )
            self.assertTrue(-EPS <= candidates <= 1 + EPS)

    def test_EI_cuda(self):
        if torch.cuda.is_available():
            self.test_EI(cuda=True)
