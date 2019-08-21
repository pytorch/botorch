#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import warnings
from typing import Optional
from unittest import mock

import torch
from botorch import settings
from botorch.exceptions.warnings import BadInitialCandidatesWarning
from botorch.optim.optimize import (
    gen_batch_initial_conditions,
    joint_optimize,
    optimize_acqf,
    sequential_optimize,
)
from torch import Tensor

from ..botorch_test_case import BotorchTestCase


class MockAcquisitionFunction:
    def __init__(self):
        self.model = None
        self.X_pending = None

    def __call__(self, X):
        return X[..., 0].max(dim=-1)[0]

    def set_X_pending(self, X_pending: Optional[Tensor] = None):
        self.X_pending = X_pending


def rounding_func(X: Tensor) -> Tensor:
    batch_shape, d = X.shape[:-1], X.shape[-1]
    X_round = torch.stack([x.round() for x in X.view(-1, d)])
    return X_round.view(*batch_shape, d)


class TestDeprecatedOptimize(BotorchTestCase):

    shared_kwargs = {
        "acq_function": MockAcquisitionFunction(),
        "bounds": torch.zeros(2, 2),
        "q": 3,
        "num_restarts": 2,
        "raw_samples": 10,
        "options": {},
        "inequality_constraints": None,
        "equality_constraints": None,
        "fixed_features": None,
        "post_processing_func": None,
    }

    @mock.patch("botorch.optim.optimize.optimize_acqf", return_value=(None, None))
    def test_joint_optimize(self, mock_optimize_acqf):
        kwargs = {
            **self.shared_kwargs,
            "return_best_only": True,
            "batch_initial_conditions": None,
        }
        with warnings.catch_warnings(record=True) as ws:
            candidates, acq_values = joint_optimize(**kwargs)
            self.assertTrue(any(issubclass(w.category, DeprecationWarning) for w in ws))
            self.assertTrue(
                any("joint_optimize is deprecated" in str(w.message) for w in ws)
            )
            mock_optimize_acqf.assert_called_once_with(**kwargs, sequential=False)
            self.assertIsNone(candidates)
            self.assertIsNone(acq_values)

    @mock.patch("botorch.optim.optimize.optimize_acqf", return_value=(None, None))
    def test_sequential_optimize(self, mock_optimize_acqf):
        with warnings.catch_warnings(record=True) as ws:
            candidates, acq_values = sequential_optimize(**self.shared_kwargs)
            self.assertTrue(any(issubclass(w.category, DeprecationWarning) for w in ws))
            self.assertTrue(
                any("sequential_optimize is deprecated" in str(w.message) for w in ws)
            )
            mock_optimize_acqf.assert_called_once_with(
                **self.shared_kwargs,
                return_best_only=True,
                sequential=True,
                batch_initial_conditions=None,
            )
            self.assertIsNone(candidates)
            self.assertIsNone(acq_values)


class TestGenBatchInitialcandidates(BotorchTestCase):
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
            with warnings.catch_warnings(record=True) as ws, settings.debug(True):
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


class TestOptimizeAcqf(BotorchTestCase):
    @mock.patch("botorch.optim.optimize.gen_batch_initial_conditions")
    @mock.patch("botorch.optim.optimize.gen_candidates_scipy")
    def test_optimize_acqf_joint(
        self, mock_gen_candidates, mock_gen_batch_initial_conditions, cuda=False
    ):
        q = 3
        num_restarts = 2
        raw_samples = 10
        options = {}
        mock_acq_function = MockAcquisitionFunction()
        tkwargs = {"device": torch.device("cuda") if cuda else torch.device("cpu")}
        cnt = 1
        for dtype in (torch.float, torch.double):
            tkwargs["dtype"] = dtype
            mock_gen_batch_initial_conditions.return_value = torch.zeros(
                num_restarts, q, 3, **tkwargs
            )
            base_cand = torch.ones(1, q, 3, **tkwargs)
            mock_candidates = torch.cat(
                [i * base_cand for i in range(num_restarts)], dim=0
            )
            mock_acq_values = num_restarts - torch.arange(num_restarts, **tkwargs)
            mock_gen_candidates.return_value = (mock_candidates, mock_acq_values)
            bounds = torch.stack(
                [torch.zeros(3, **tkwargs), 4 * torch.ones(3, **tkwargs)]
            )
            candidates, acq_vals = optimize_acqf(
                acq_function=mock_acq_function,
                bounds=bounds,
                q=q,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                options=options,
            )
            self.assertTrue(torch.equal(candidates, mock_candidates[0]))
            self.assertTrue(torch.equal(acq_vals, mock_acq_values[0]))

            candidates, acq_vals = optimize_acqf(
                acq_function=mock_acq_function,
                bounds=bounds,
                q=q,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                options=options,
                return_best_only=False,
                batch_initial_conditions=torch.zeros(num_restarts, q, 3, **tkwargs),
            )
            self.assertTrue(torch.equal(candidates, mock_candidates))
            self.assertTrue(torch.equal(acq_vals, mock_acq_values))
            self.assertEqual(mock_gen_batch_initial_conditions.call_count, cnt)
            cnt += 1

    def test_optimize_acqf_joint_cuda(self):
        if torch.cuda.is_available():
            self.test_optimize_acqf_joint(cuda=True)

    @mock.patch("botorch.optim.optimize.gen_batch_initial_conditions")
    @mock.patch("botorch.optim.optimize.gen_candidates_scipy")
    def test_optimize_acqf_sequential(
        self, mock_gen_candidates_scipy, mock_gen_batch_initial_conditions, cuda=False
    ):
        q = 3
        num_restarts = 2
        raw_samples = 10
        options = {}
        tkwargs = {"device": torch.device("cuda") if cuda else torch.device("cpu")}
        for dtype in (torch.float, torch.double):
            mock_acq_function = MockAcquisitionFunction()
            tkwargs["dtype"] = dtype
            mock_gen_batch_initial_conditions.side_effect = [
                torch.zeros(num_restarts, **tkwargs) for _ in range(q)
            ]
            gcs_return_vals = [
                (
                    torch.tensor([[[1.1, 2.1, 3.1]]], **tkwargs),
                    torch.tensor([i], **tkwargs),
                )
                for i in range(q)
            ]
            mock_gen_candidates_scipy.side_effect = gcs_return_vals
            expected_candidates = torch.cat(
                [rv[0][0] for rv in gcs_return_vals], dim=-2
            ).round()
            bounds = torch.stack(
                [torch.zeros(3, **tkwargs), 4 * torch.ones(3, **tkwargs)]
            )
            inequality_constraints = [
                (torch.tensor([3]), torch.tensor([4]), torch.tensor(5))
            ]
            candidates, acq_value = optimize_acqf(
                acq_function=mock_acq_function,
                bounds=bounds,
                q=q,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                options=options,
                inequality_constraints=inequality_constraints,
                post_processing_func=rounding_func,
                sequential=True,
            )
            self.assertTrue(torch.equal(candidates, expected_candidates))
            self.assertTrue(
                torch.equal(acq_value, torch.cat([rv[1] for rv in gcs_return_vals]))
            )

    def test_optimize_acqf_sequential_notimplemented(self):
        with self.assertRaises(NotImplementedError):
            optimize_acqf(
                acq_function=MockAcquisitionFunction(),
                bounds=torch.stack([torch.zeros(3), 4 * torch.ones(3)]),
                q=3,
                num_restarts=2,
                raw_samples=10,
                return_best_only=False,
                sequential=True,
            )

    def test_optimize_acqf_sequential_cuda(self):
        if torch.cuda.is_available():
            self.test_optimize_acqf_sequential(cuda=True)
