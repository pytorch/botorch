#! /usr/bin/env python3

import math
import unittest

import torch
from botorch import fit_model, gen_candidates_scipy
from botorch.acquisition import qExpectedImprovement
from botorch.models import SingleTaskGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood

from .test_fit import NOISE


class TestBaseCandidateGeneration(unittest.TestCase):
    def _setUp(self, double=False, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        dtype = torch.double if double else torch.float
        train_x = torch.linspace(0, 1, 10, device=device, dtype=dtype)
        train_y = torch.sin(train_x * (2 * math.pi))
        noise = torch.tensor(NOISE, device=device, dtype=dtype)
        self.train_x = train_x.unsqueeze(-1)
        self.train_y = train_y + noise
        self.f_best = self.train_y.max().item()
        self.initial_candidates = torch.tensor([[0.5]], device=device, dtype=dtype)
        self.likelihood = GaussianLikelihood()
        model = SingleTaskGP(self.train_x, self.train_y, self.likelihood)
        self.model = model.to(device=device, dtype=dtype)
        self.mll = ExactMarginalLogLikelihood(self.likelihood, self.model)
        self.mll = fit_model(self.mll, options={"maxiter": 1})


class TestGenCandidates(TestBaseCandidateGeneration):
    def _setUp(self, double=False, cuda=False):
        super()._setUp(double=double, cuda=cuda)
        self.f_best = self.train_y.max().item()
        self.initial_candidates = torch.tensor([[0.5]]).type_as(self.train_x)

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
            self.assertTrue(0 <= candidates <= 1)

    def test_gen_candidates_scipy_cuda(self):
        if torch.cuda.is_available():
            self.test_gen_candidates_scipy(cuda=True)


if __name__ == "__main__":
    unittest.main()
