#!/usr/bin/env python3

import warnings
from unittest import TestCase, mock

import torch
from botorch.exceptions.errors import UnsupportedError
from botorch.exceptions.warnings import BadInitialCandidatesWarning
from botorch.optim.optimize import (
    gen_batch_initial_conditions,
    joint_optimize,
    sequential_optimize,
)
from torch import Tensor


class MockAcquisitionFunction:
    def __init__(self, has_X_baseline_attr=True):
        if has_X_baseline_attr:
            self.X_baseline = None
        self.model = None

    def __call__(self, X):
        return X[..., 0].max(dim=-1)[0]


def rounding_func(X: Tensor) -> Tensor:
    return X.round()


class TestGenBatchInitialcandidates(TestCase):
    def test_gen_batch_initial_conditions(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            bounds = torch.tensor([[0, 0], [1, 1]], device=device, dtype=dtype)
            for nonnegative in (True, False):
                batch_initial_conditions = gen_batch_initial_conditions(
                    acq_function=MockAcquisitionFunction(),
                    bounds=bounds,
                    q=1,
                    num_restarts=2,
                    raw_samples=10,
                    options={"nonnegative": nonnegative, "eta": 0.01, "alpha": 0.1},
                )
                expected_shape = torch.Size([2, 1, 2])
                self.assertEqual(batch_initial_conditions.shape, expected_shape)
                self.assertEqual(batch_initial_conditions.device, bounds.device)
                self.assertEqual(batch_initial_conditions.dtype, bounds.dtype)

    def test_gen_batch_initial_conditions_cuda(self):
        if torch.cuda.is_available():
            self.test_gen_batch_initial_conditions(cuda=True)

    def test_gen_batch_initial_conditions_simple_warning(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            bounds = torch.tensor([[0, 0], [1, 1]], device=device, dtype=dtype)
            with warnings.catch_warnings(record=True) as ws:
                with mock.patch(
                    "botorch.optim.optimize.draw_sobol_samples",
                    return_value=torch.zeros(10, 1, 2, device=device, dtype=dtype),
                ):
                    batch_initial_conditions = gen_batch_initial_conditions(
                        acq_function=MockAcquisitionFunction(),
                        bounds=bounds,
                        q=1,
                        num_restarts=2,
                        raw_samples=10,
                    )
                    self.assertEqual(len(ws), 1)
                    self.assertTrue(
                        issubclass(ws[-1].category, BadInitialCandidatesWarning)
                    )
                    self.assertTrue(
                        torch.equal(
                            batch_initial_conditions,
                            torch.zeros(2, 1, 2, device=device, dtype=dtype),
                        )
                    )

    def test_gen_batch_initial_conditions_simple_raises_cuda(self):
        if torch.cuda.is_available():
            self.test_gen_batch_initial_conditions_simple_warning(cuda=True)


class TestSequentialOptimize(TestCase):
    @mock.patch("botorch.optim.optimize.joint_optimize")
    def test_sequential_optimize(self, mock_joint_optimize, cuda=False):
        q = 3
        num_restarts = 2
        raw_samples = 10
        options = {}
        tkwargs = {"device": torch.device("cuda") if cuda else torch.device("cpu")}
        for dtype in (torch.float, torch.double):
            mock_acq_function = MockAcquisitionFunction()
            tkwargs["dtype"] = dtype
            joint_optimize_return_values = [
                torch.tensor([[[1.1, 2.1, 3.1]]], **tkwargs) for _ in range(q)
            ]
            mock_joint_optimize.side_effect = joint_optimize_return_values
            expected_candidates = torch.cat(
                joint_optimize_return_values, dim=-2
            ).round()
            bounds = torch.stack(
                [torch.zeros(3, **tkwargs), 4 * torch.ones(3, **tkwargs)]
            )
            inequality_constraints = [
                (torch.tensor([3]), torch.tensor([4]), torch.tensor(5))
            ]
            candidates = sequential_optimize(
                acq_function=mock_acq_function,
                bounds=bounds,
                q=q,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                options=options,
                inequality_constraints=inequality_constraints,
                post_processing_func=rounding_func,
            )
            self.assertTrue(torch.equal(candidates, expected_candidates))

            expected_call_kwargs = {
                "acq_function": mock_acq_function,
                "bounds": bounds,
                "q": 1,
                "num_restarts": num_restarts,
                "raw_samples": raw_samples,
                "options": options,
                "inequality_constraints": inequality_constraints,
                "equality_constraints": None,
                "fixed_features": None,
            }
            call_args_list = mock_joint_optimize.call_args_list[-q:]
            for i in range(q):
                self.assertEqual(call_args_list[i][1], expected_call_kwargs)

            # test that error is raised for acquisition functions without X_baseline
            mock_acq_function = MockAcquisitionFunction(has_X_baseline_attr=False)
            with self.assertRaises(UnsupportedError):
                sequential_optimize(
                    acq_function=mock_acq_function,
                    bounds=bounds,
                    q=q,
                    num_restarts=num_restarts,
                    raw_samples=raw_samples,
                )

    def test_sequential_optimize_cuda(self):
        if torch.cuda.is_available():
            self.test_sequential_optimize(cuda=True)


class TestJointOptimize(TestCase):
    @mock.patch("botorch.optim.optimize.gen_batch_initial_conditions")
    @mock.patch("botorch.optim.optimize.gen_candidates_scipy")
    @mock.patch("botorch.optim.optimize.get_best_candidates")
    def test_joint_optimize(
        self,
        mock_get_best_candidates,
        mock_gen_candidates,
        mock_gen_batch_initial_conditions,
        cuda=False,
    ):
        q = 3
        num_restarts = 2
        raw_samples = 10
        options = {}
        mock_acq_function = MockAcquisitionFunction()
        tkwargs = {"device": torch.device("cuda") if cuda else torch.device("cpu")}
        for dtype in (torch.float, torch.double):
            tkwargs["dtype"] = dtype
            mock_gen_batch_initial_conditions.return_value = torch.zeros(
                num_restarts, q, 3, **tkwargs
            )
            mock_gen_candidates.return_value = torch.cat(
                [i * torch.ones(1, q, 3, **tkwargs) for i in range(num_restarts)], dim=0
            )
            mock_get_best_candidates.return_value = torch.ones(1, q, 3, **tkwargs)
            expected_candidates = mock_get_best_candidates.return_value
            bounds = torch.stack(
                [torch.zeros(3, **tkwargs), 4 * torch.ones(3, **tkwargs)]
            )
            candidates = joint_optimize(
                acq_function=mock_acq_function,
                bounds=bounds,
                q=q,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                options=options,
            )
            self.assertTrue(torch.equal(candidates, expected_candidates))

    def test_joint_optimize_cuda(self):
        if torch.cuda.is_available():
            self.test_joint_optimize(cuda=True)
