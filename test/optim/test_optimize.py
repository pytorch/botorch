#!/usr/bin/env python3

import warnings
from unittest import TestCase, mock

import torch
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.optim.optimize import (
    gen_batch_initial_arms,
    joint_optimize,
    sequential_optimize,
)
from torch import Tensor


class MockAcquisitionFunction:
    def __init__(self):
        self.X_baseline = None
        self.model = None

    def __call__(self, X):
        return X[..., 0].max(dim=-1)[0]


def rounding_func(X: Tensor) -> Tensor:
    return X.round()


class GenBatchInitialArmsTest(TestCase):
    def test_gen_batch_initial_arms(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            bounds = torch.tensor([[0, 0], [1, 1]], device=device, dtype=dtype)
            for simple in (True, False):
                batch_initial_arms = gen_batch_initial_arms(
                    acq_function=MockAcquisitionFunction(),
                    bounds=bounds,
                    n=1,
                    num_restarts=2,
                    raw_samples=10,
                    options={"simple_init": simple},
                )
                self.assertEqual(batch_initial_arms.shape, torch.Size([2, 1, 2]))
                self.assertEqual(batch_initial_arms.device, bounds.device)
                self.assertEqual(batch_initial_arms.dtype, bounds.dtype)

    def test_gen_batch_initial_arms_cuda(self):
        if torch.cuda.is_available():
            self.test_gen_batch_initial_arms(cuda=True)

    def test_gen_batch_initial_arms_simple_warning(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            bounds = torch.tensor([[0, 0], [1, 1]], device=device, dtype=dtype)
            with warnings.catch_warnings(record=True) as ws:
                with mock.patch(
                    "botorch.optim.optimize.draw_sobol_samples",
                    return_value=torch.zeros(10, 1, 2, device=device, dtype=dtype),
                ):
                    batch_initial_arms = gen_batch_initial_arms(
                        acq_function=MockAcquisitionFunction(),
                        bounds=bounds,
                        n=1,
                        num_restarts=2,
                        raw_samples=10,
                        options={"simple_init": True},
                    )
                    self.assertEqual(len(ws), 1)
                    self.assertTrue(
                        issubclass(ws[-1].category, BadInitialCandidatesWarning)
                    )
                    self.assertTrue(
                        torch.equal(
                            batch_initial_arms,
                            torch.zeros(2, 1, 2, device=device, dtype=dtype),
                        )
                    )

    def test_gen_batch_initial_arms_simple_raises_cuda(self):
        if torch.cuda.is_available():
            self.test_gen_batch_initial_arms_simple_warning(cuda=True)


class TestSequentialOptimize(TestCase):
    @mock.patch("botorch.optim.optimize.joint_optimize")
    def test_sequential_optimize(self, mock_joint_optimize, cuda=False):
        n = 3
        num_restarts = 2
        raw_samples = 10
        options = {}
        mock_acq_function = MockAcquisitionFunction()
        tkwargs = {"device": torch.device("cuda") if cuda else torch.device("cpu")}
        for dtype in (torch.float, torch.double):
            tkwargs["dtype"] = dtype
            joint_optimize_return_values = [
                torch.tensor([[[1.1, 2.1, 3.1]]], **tkwargs) for _ in range(n)
            ]
            mock_joint_optimize.side_effect = joint_optimize_return_values
            expected_candidates = torch.cat(
                joint_optimize_return_values, dim=-2
            ).round()
            bounds = torch.stack(
                [torch.zeros(3, **tkwargs), 4 * torch.ones(3, **tkwargs)]
            )
            candidates = sequential_optimize(
                acq_function=mock_acq_function,
                bounds=bounds,
                n=n,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                options=options,
                post_processing_func=rounding_func,
            )
            self.assertTrue(torch.equal(candidates, expected_candidates))

            expected_call_kwargs = {
                "acq_function": mock_acq_function,
                "bounds": bounds,
                "n": 1,
                "num_restarts": num_restarts,
                "raw_samples": raw_samples,
                "options": options,
                "fixed_features": None,
            }
            call_args_list = mock_joint_optimize.call_args_list[-n:]
            for i in range(n):
                self.assertEqual(call_args_list[i][1], expected_call_kwargs)

    def test_sequential_optimize_cuda(self):
        if torch.cuda.is_available():
            self.test_sequential_optimize(cuda=True)


class TestJointOptimize(TestCase):
    @mock.patch("botorch.optim.optimize.gen_batch_initial_arms")
    @mock.patch("botorch.optim.optimize.gen_candidates_scipy")
    @mock.patch("botorch.optim.optimize.get_best_candidates")
    def test_joint_optimize(
        self,
        mock_get_best_candidates,
        mock_gen_candidates,
        mock_gen_batch_initial_arms,
        cuda=False,
    ):
        n = 3
        num_restarts = 2
        raw_samples = 10
        options = {}
        mock_acq_function = MockAcquisitionFunction()
        tkwargs = {"device": torch.device("cuda") if cuda else torch.device("cpu")}
        for dtype in (torch.float, torch.double):
            tkwargs["dtype"] = dtype
            mock_gen_batch_initial_arms.return_value = torch.zeros(
                num_restarts, n, 3, **tkwargs
            )
            mock_gen_candidates.return_value = torch.cat(
                [i * torch.ones(1, n, 3, **tkwargs) for i in range(num_restarts)], dim=0
            )
            mock_get_best_candidates.return_value = torch.ones(1, n, 3, **tkwargs)
            expected_candidates = mock_get_best_candidates.return_value
            bounds = torch.stack(
                [torch.zeros(3, **tkwargs), 4 * torch.ones(3, **tkwargs)]
            )
            candidates = joint_optimize(
                acq_function=mock_acq_function,
                bounds=bounds,
                n=n,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                options=options,
            )
            self.assertTrue(torch.equal(candidates, expected_candidates))

    def test_joint_optimize_cuda(self):
        if torch.cuda.is_available():
            self.test_joint_optimize(cuda=True)
