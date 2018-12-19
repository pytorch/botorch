#!/usr/bin/env python3

import unittest
from unittest import mock

import torch
from botorch.benchmarks.optimize import (
    OptimizeConfig,
    greedy,
    run_benchmark,
    run_closed_loop,
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


def test_func(X):
    f_X = torch.sin(X)
    return (f_X + torch.randn_like(f_X)).view(-1), torch.tensor([])


class TestRunClosedLoop(unittest.TestCase):
    def setUp(self):
        self.config = OptimizeConfig(
            acquisition_function_name="qEI",
            initial_points=5,
            q=2,
            n_batch=2,
            candidate_gen_max_iter=3,
            model_max_iter=3,
            num_starting_points=1,
            max_retries=0,
        )
        self.func = test_func

    @mock.patch("botorch.benchmarks.optimize.fit_model")
    @mock.patch("botorch.benchmarks.optimize.ExactMarginalLogLikelihood")
    @mock.patch("botorch.benchmarks.optimize.random_restarts")
    @mock.patch("botorch.benchmarks.optimize.SingleTaskGP")
    def test_run_closed_loop(
        self,
        mock_single_task_gp,
        mock_random_restarts,
        _mock_mll,
        _mock_fit_model,
        cuda=False,
    ):
        for dtype in [torch.float, torch.double]:
            bounds = get_bounds(cuda, dtype=dtype)
            tkwargs = {"dtype": dtype, "device": bounds.device}
            gen_x = get_gen_x(bounds)
            mock_random_restarts.side_effect = [
                gen_x(self.config.q) for _ in range(self.config.n_batch - 1)
            ]
            mean1 = torch.ones(self.config.initial_points, **tkwargs)
            samples1 = torch.zeros([2, self.config.initial_points], **tkwargs)
            mm1 = MockModel(MockPosterior(mean=mean1, samples=samples1))
            samples2 = torch.zeros([2, self.config.q], **tkwargs)
            mm2 = MockModel(MockPosterior(samples=samples2))
            mock_single_task_gp.side_effect = [mm1, mm2]
            # basic test for output shapes and types
            output = run_closed_loop(
                func=self.func,
                gen_function=gen_x,
                config=self.config,
                lower_bounds=bounds[0],
                upper_bounds=bounds[1],
            )
            self.assertTrue(isinstance(output, ClosedLoopOutput))
            self.assertEqual(len(output.Xs), 2)
            self.assertEqual(output.Xs[0].shape[0], self.config.initial_points)
            self.assertEqual(output.Xs[1].shape[0], self.config.q)
            self.assertEqual(len(output.Ys), 2)
            self.assertEqual(output.Ys[0].shape[0], self.config.initial_points)
            self.assertEqual(output.Ys[1].shape[0], self.config.q)
            self.assertEqual(len(output.best), 2)
            self.assertEqual(len(output.best_model_objective), 2)
            self.assertEqual(len(output.best_model_feasibility), 2)
            self.assertEqual(output.costs, [1.0 for _ in range(self.config.n_batch)])
            self.assertTrue(output.runtime > 0.0)

    def test_run_closed_loop_cuda(self):
        if torch.cuda.is_available():
            self.test_run_closed_loop(cuda=True)


class TestRunBenchmark(unittest.TestCase):
    def setUp(self):
        self.config = OptimizeConfig(
            acquisition_function_name="qEI",
            initial_points=5,
            q=2,
            n_batch=2,
            candidate_gen_max_iter=3,
            model_max_iter=3,
            num_starting_points=1,
            max_retries=0,
        )
        self.func = lambda X: (X + 0.25, torch.tensor([]))
        self.global_optimum = 5.0

    @mock.patch("botorch.benchmarks.optimize.run_closed_loop")
    def test_run_benchmark(self, mock_run_closed_loop, cuda=False):
        for dtype in [torch.float, torch.double]:
            bounds = get_bounds(cuda, dtype=dtype)
            tkwargs = {"dtype": dtype, "device": bounds.device}
            Xs = [
                torch.tensor([-1.0, 0.0], **tkwargs).view(-1, 1),
                torch.tensor([1.0, 0.0], **tkwargs).view(-1, 1),
            ]
            Ys = []
            best_model_objective = []
            best_model_feasibility = []
            best = []
            costs = []
            for X in Xs:
                Ys.append(X + 0.5)
                best_obj, best_idx = torch.max(X, dim=0)
                best_model_objective.append(best_obj.item())
                best.append(X[best_idx])
                best_model_feasibility.append(1.0)
                costs.append(1.0)
            closed_loop_output = ClosedLoopOutput(
                Xs=Xs,
                Ys=Ys,
                Ycovs=[],
                best=best,
                best_model_objective=best_model_objective,
                best_model_feasibility=best_model_feasibility,
                costs=costs,
                runtime=1.0,
            )
            mock_run_closed_loop.return_value = closed_loop_output
            gen_x = get_gen_x(bounds)
            outputs = run_benchmark(
                func=self.func,
                gen_function=gen_x,
                config=self.config,
                lower_bounds=bounds[0],
                upper_bounds=bounds[1],
                num_runs=2,
                true_func=self.func,
                global_optimum=self.global_optimum,
            )
            self.assertTrue(isinstance(outputs, BenchmarkOutput))
            # Check 2 trials
            self.assertEqual(len(outputs.Xs), 2)
            # Check iterations
            self.assertEqual(len(outputs.Xs[0]), 2)
            expected_best_true_objective = torch.tensor([0.25, 1.25], **tkwargs)
            self.assertTrue(
                torch.equal(
                    outputs.best_true_objective[0], expected_best_true_objective
                )
            )
            expected_best_true_feasibility = torch.ones(2, **tkwargs)
            self.assertTrue(
                torch.equal(
                    outputs.best_true_feasibility[0], expected_best_true_feasibility
                )
            )
            expected_regrets_trial = torch.tensor(
                [(self.global_optimum - self.func(X)[0]).sum().item() for X in Xs]
            ).type_as(Xs[0])
            self.assertEqual(len(outputs.regrets[0]), len(expected_regrets_trial))
            self.assertTrue(
                torch.equal(outputs.regrets[0][0], expected_regrets_trial[0])
            )
            self.assertTrue(
                torch.equal(
                    outputs.cumulative_regrets[0],
                    torch.cumsum(expected_regrets_trial, dim=0),
                )
            )
            # test modifying the objective
            outputs2 = run_benchmark(
                func=self.func,
                gen_function=gen_x,
                config=self.config,
                objective=lambda Y: Y + 0.1,
                lower_bounds=bounds[0],
                upper_bounds=bounds[1],
                true_func=self.func,
            )
            expected_best_true_objective2 = torch.tensor([0.35, 1.35], **tkwargs)
            self.assertTrue(
                torch.equal(
                    outputs2.best_true_objective[0], expected_best_true_objective2
                )
            )
            # test constraints
            outputs3 = run_benchmark(
                func=self.func,
                gen_function=gen_x,
                config=self.config,
                constraints=[lambda Y: torch.ones_like(Y)],
                lower_bounds=bounds[0],
                upper_bounds=bounds[1],
                true_func=self.func,
            )
            expected_best_true_feasibility = torch.zeros(2, **tkwargs)
            self.assertTrue(
                torch.equal(
                    outputs3.best_true_feasibility[0], expected_best_true_feasibility
                )
            )

    def test_run_benchmark_cuda(self):
        if torch.cuda.is_available():
            self.test_run_benchmark(cuda=True)


class TestGreedy(unittest.TestCase):
    def test_greedy(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in [torch.float, torch.double]:
            # basic test
            X = torch.tensor([1.0, 2.0, 3.0], device=device, dtype=dtype).view(-1, 1)
            model = MockModel(MockPosterior(samples=X.view(1, -1) * 2.0))
            # basic test
            (best_point, best_obj, feasiblity) = greedy(X=X, model=model)
            self.assertAlmostEqual(best_point.item(), 3.0, places=6)
            # interestingly, on the GPU this comparison is not exact
            self.assertAlmostEqual(best_obj, 6.0, places=6)
            self.assertAlmostEqual(feasiblity, 1.0, places=6)
            # test objective
            (best_point2, best_obj2, feasiblity2) = greedy(
                X=X, model=model, objective=lambda Y: 0.5 * Y
            )
            print((best_point2, best_obj2, feasiblity2))
            self.assertAlmostEqual(best_point2.item(), 3.0, places=6)
            self.assertAlmostEqual(best_obj2, 3.0, places=6)
            self.assertAlmostEqual(feasiblity2, 1.0, places=6)
            # test constraints
            feasiblity3 = greedy(
                X=X, model=model, constraints=[lambda Y: torch.ones_like(Y)]
            )[2]
            self.assertAlmostEqual(feasiblity3, 0.0, places=6)

    def test_greedy_cuda(self):
        if torch.cuda.is_available():
            self.test_greedy(cuda=True)
