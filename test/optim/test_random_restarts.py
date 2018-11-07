#! /usr/bin/env python3

import unittest

import torch
from botorch.acquisition import qExpectedImprovement
from botorch.optim.random_restarts import random_restarts
from botorch.utils import gen_x_uniform

from ..test_gen import TestBaseCandidateGeneration


class TestRandomRestarts(TestBaseCandidateGeneration):
    def setUp(self):
        super(TestRandomRestarts, self).setUp()

    def test_random_restarts(self, cuda=False):

        qEI = qExpectedImprovement(
            self.model, best_f=self.model(self.train_x).mean.max().item()
        )
        bounds = torch.tensor([[0.0], [1.0]])
        candidates = random_restarts(
            gen_function=lambda n: gen_x_uniform(n, bounds),
            acq_function=qEI,
            q=1,
            num_starting_points=3,
            lower_bounds=0,
            upper_bounds=1,
            max_iter=3,
        )
        print(candidates.shape)
        self.assertTrue(0 <= candidates <= 1)


if __name__ == "__main__":
    unittest.main()
