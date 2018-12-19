#!/usr/bin/env python3

import unittest

import torch
from botorch.benchmarks.aggregate import aggregate_benchmark
from botorch.benchmarks.output import AggregatedBenchmarkOutput, BenchmarkOutput


class TestAggregateBenchMark(unittest.TestCase):
    def test_aggregate_benchmark(self, cuda=False):

        num_iterations = 4
        num_trials = 3
        global_optimum = 10.0
        device = torch.device("cuda" if cuda else "cpu")
        tkwargs = {"device": device}
        for dtype in (torch.float, torch.double):
            tkwargs["dtype"] = dtype
            X = [
                torch.tensor([i], **tkwargs).view(-1, 1) for i in range(num_iterations)
            ]
            Xs = [X for trial in range(num_trials)]
            # f_hat(X) = X + 0.5
            best_model_objective = [
                (torch.cat(X_trial).view(-1) + 0.5).tolist() for X_trial in Xs
            ]
            # f(X) = X + 0.75
            best_true_objective = [
                torch.cat([(X_i + 0.75).view(-1) for X_i in X_trial], dim=0).view(
                    num_iterations
                )
                for X_trial in Xs
            ]
            best_true_feasibility = [
                torch.ones_like(torch.cat(X_trial).view(-1), **tkwargs)
                for X_trial in Xs
            ]
            best_model_feasibility = [feas.tolist() for feas in best_true_feasibility]
            regrets = [
                (global_optimum - best_true_objective[trial])
                for trial in range(num_trials)
            ]
            cumulative_regrets = [
                torch.cumsum(regrets[trial], dim=0) for trial in range(num_trials)
            ]
            runtimes = torch.ones(num_trials, **tkwargs)
            output = BenchmarkOutput(
                Xs=Xs,
                Ys=[[]],
                Ycovs=[[]],
                best=[[]],
                best_model_objective=best_model_objective,
                best_model_feasibility=best_model_feasibility,
                costs=best_model_feasibility,
                runtime=runtimes,
                best_true_objective=best_true_objective,
                best_true_feasibility=best_true_feasibility,
                regrets=regrets,
                cumulative_regrets=cumulative_regrets,
            )
            agg_results = aggregate_benchmark(output)
            self.assertTrue(isinstance(agg_results, AggregatedBenchmarkOutput))
            self.assertEqual(agg_results.num_trials, 3)
            # on the GPU we can't check for exact equality b/c of numerical issues
            self.assertAlmostEqual(agg_results.mean_runtime, 1.0)
            self.assertAlmostEqual(agg_results.var_runtime, 0.0)
            self.assertTrue(
                torch.equal(
                    agg_results.batch_iterations,
                    torch.tensor([1, 2, 3, 4], dtype=torch.long, device=device),
                )
            )
            ni_shape = torch.Size([num_iterations])

            self.assertEqual(agg_results.mean_cost.shape, ni_shape)
            self.assertLess(torch.norm(agg_results.mean_cost - 1), 1e-6)
            self.assertEqual(agg_results.var_cost.shape, ni_shape)
            self.assertLess(torch.norm(agg_results.var_cost), 1e-6)
            mbmo = torch.tensor(best_model_objective[0], **tkwargs)
            self.assertTrue(agg_results.mean_best_model_objective.shape == mbmo.shape)
            self.assertLess(
                torch.norm(agg_results.mean_best_model_objective - mbmo), 1e-6
            )
            self.assertEqual(agg_results.var_best_model_objective.shape, ni_shape)
            self.assertLess(torch.norm(agg_results.var_best_model_objective), 1e-6)
            self.assertEqual(agg_results.mean_best_model_feasibility.shape, ni_shape)
            self.assertLess(
                torch.norm(agg_results.mean_best_model_feasibility - 1), 1e-6
            )
            self.assertEqual(agg_results.var_best_model_feasibility.shape, ni_shape)
            self.assertLess(torch.norm(agg_results.var_best_model_feasibility), 1e-6)
            bto = torch.tensor(best_true_objective[0], **tkwargs)
            self.assertEqual(agg_results.mean_best_true_objective.shape, bto.shape)
            self.assertLess(
                torch.norm(agg_results.mean_best_true_objective - bto), 1e-6
            )
            self.assertEqual(agg_results.var_best_true_objective.shape, ni_shape)
            self.assertLess(torch.norm(agg_results.var_best_true_objective), 1e-6)
            mr = torch.tensor(regrets[0], **tkwargs)
            self.assertEqual(agg_results.mean_regret.shape, mr.shape)
            self.assertLess(torch.norm(agg_results.mean_regret - mr), 1e-6)
            self.assertEqual(agg_results.var_regret.shape, ni_shape)
            self.assertLess(torch.norm(agg_results.var_regret), 1e-6)
            self.assertEqual(
                agg_results.mean_cumulative_regret.shape, cumulative_regrets[0].shape
            )
            self.assertLess(
                torch.norm(agg_results.mean_cumulative_regret - cumulative_regrets[0]),
                1e-5,
            )
            self.assertEqual(agg_results.var_cumulative_regret.shape, ni_shape)
            self.assertLess(torch.norm(agg_results.var_cumulative_regret), 1e-6)

    def test_aggregate_benchmark_cuda(self):
        if torch.cuda.is_available():
            self.test_aggregate_benchmark(cuda=True)
