#!/usr/bin/env python3

import unittest

import torch
from botorch.benchmarks.optimize import (
    OptimizeConfig,
    greedy,
    optimize,
    optimize_multiple_runs,
)
from botorch.benchmarks.output import BenchmarkOutput, ClosedLoopOutput
from botorch.utils import gen_x_uniform

from ..utils.mock import MockModel, MockPosterior


def get_bounds(cuda, dtype):
    device = torch.device("cuda") if cuda else torch.device("cpu")
    return torch.tensor([-1.0, 1.0], dtype=dtype, device=device).view(2, 1)


def get_gen_x(bounds):
    def gen_x(num_samples):
        return gen_x_uniform(num_samples, bounds=bounds)

    return gen_x


def test_func(x):
    y = torch.sin(x) + 0.1 * torch.rand_like(x, device=x.device, dtype=x.dtype)
    return (y.view(-1), torch.tensor(()))


class TestOptimize(unittest.TestCase):
    def setUp(self):
        self.config = OptimizeConfig(
            acquisition_function_name="qEI",
            initial_points=10,
            q=2,
            n_batch=2,
            candidate_gen_max_iter=3,
            model_max_iter=3,
            num_starting_points=1,
        )
        self.func = test_func

    def test_optimize(self, cuda=False):
        for dtype in [torch.float, torch.double]:
            bounds = get_bounds(cuda, dtype=dtype)
            gen_x = get_gen_x(bounds)
            output = optimize(
                func=self.func,
                gen_function=gen_x,
                config=self.config,
                lower_bounds=bounds[0],
                upper_bounds=bounds[1],
            )
            self.assertTrue(isinstance(output, ClosedLoopOutput))
            self.assertEqual(len(output.Xs), 2)

    def test_optimize_cuda(self):
        if torch.cuda.is_available():
            self.test_optimize(cuda=True)

    def test_optimize_multiple_runs(self, cuda=False):
        for dtype in [torch.float, torch.double]:
            bounds = get_bounds(cuda, dtype=dtype)
            gen_x = get_gen_x(bounds)
            outputs = optimize_multiple_runs(
                func=self.func,
                gen_function=gen_x,
                config=self.config,
                lower_bounds=bounds[0],
                upper_bounds=bounds[1],
                num_runs=2,
            )
            self.assertTrue(isinstance(outputs, BenchmarkOutput))
            self.assertEqual(len(outputs.Xs), 2)

    def test_optimize_multiple_runs_cuda(self):
        if torch.cuda.is_available():
            self.test_optimize_multiple_runs(cuda=True)


class TestGreedy(unittest.TestCase):
    def test_greedy(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in [torch.float, torch.double]:
            X = torch.tensor([1.0, 2.0, 3.0], device=device, dtype=dtype)
            model = MockModel(MockPosterior(samples=X.view(1, -1) * 2.0))
            (best_point, best_obj, feasiblity) = greedy(X=X, model=model)
            print(best_point.item())
            self.assertTrue(best_point.item() == 3.0)
            self.assertTrue(best_obj == 6.0)
            self.assertTrue(feasiblity == 1)

    def test_greedy_cuda(self):
        if torch.cuda.is_available():
            self.test_greedy(cuda=True)
