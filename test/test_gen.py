#! /usr/bin/env python3

import math
import unittest

import torch
from botorch import fit_model, gen_candidates
from botorch.acquisition import qExpectedImprovement
from botorch.models import GPRegressionModel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood

from .test_fit import NOISE


class TestBaseCandidateGeneration(unittest.TestCase):
    def setUp(self, cuda=False):
        train_x = torch.linspace(0, 1, 10)
        self.train_x = train_x.unsqueeze(1)
        self.train_y = torch.sin(train_x * (2 * math.pi)) + torch.tensor(NOISE)
        self.f_best = self.train_y.max().item()
        self.initial_candidates = torch.tensor([[0.5]])
        self.likelihood = GaussianLikelihood()
        self.model = GPRegressionModel(
            self.train_x.cuda() if cuda else self.train_x,
            self.train_y.cuda() if cuda else self.train_y,
            self.likelihood,
        )
        self.mll = ExactMarginalLogLikelihood(self.likelihood, self.model)
        self.mll = fit_model(self.mll, options={"maxiter": 1})


class TestGenCandidates(TestBaseCandidateGeneration):
    def setUp(self):
        super(TestGenCandidates, self).setUp()
        self.f_best = self.train_y.max().item()
        self.initial_candidates = torch.tensor([[0.5]])

    def test_gen_candidates(self, cuda=False):
        ics = self.initial_candidates.cuda() if cuda else self.initial_candidates
        qEI = qExpectedImprovement(self.model, best_f=self.f_best)
        candidates, _ = gen_candidates(
            initial_candidates=ics,
            acquisition_function=qEI,
            lower_bounds=0,
            upper_bounds=1,
            max_iter=5,
            verbose=False,
        )
        self.assertTrue(0 <= candidates <= 1)


if __name__ == "__main__":
    unittest.main()
