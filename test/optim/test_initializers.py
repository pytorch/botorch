#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from contextlib import ExitStack
from unittest import mock

import torch
from botorch import settings
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.exceptions import BadInitialCandidatesWarning, SamplingWarning
from botorch.optim import initialize_q_batch, initialize_q_batch_nonneg
from botorch.optim.optimize import (
    gen_batch_initial_conditions,
    gen_one_shot_kg_initial_conditions,
)
from botorch.utils.testing import (
    BotorchTestCase,
    MockAcquisitionFunction,
    MockModel,
    MockPosterior,
)


class TestInitializeQBatch(BotorchTestCase):
    def test_initialize_q_batch_nonneg(self):
        for dtype in (torch.float, torch.double):
            # basic test
            X = torch.rand(5, 3, 4, device=self.device, dtype=dtype)
            Y = torch.rand(5, device=self.device, dtype=dtype)
            ics = initialize_q_batch_nonneg(X=X, Y=Y, n=2)
            self.assertEqual(ics.shape, torch.Size([2, 3, 4]))
            self.assertEqual(ics.device, X.device)
            self.assertEqual(ics.dtype, X.dtype)
            # ensure nothing happens if we want all samples
            ics = initialize_q_batch_nonneg(X=X, Y=Y, n=5)
            self.assertTrue(torch.equal(X, ics))
            # make sure things work with constant inputs
            Y = torch.ones(5, device=self.device, dtype=dtype)
            ics = initialize_q_batch_nonneg(X=X, Y=Y, n=2)
            self.assertEqual(ics.shape, torch.Size([2, 3, 4]))
            self.assertEqual(ics.device, X.device)
            self.assertEqual(ics.dtype, X.dtype)
            # ensure raises correct warning
            Y = torch.zeros(5, device=self.device, dtype=dtype)
            with warnings.catch_warnings(record=True) as w, settings.debug(True):
                ics = initialize_q_batch_nonneg(X=X, Y=Y, n=2)
                self.assertEqual(len(w), 1)
                self.assertTrue(issubclass(w[-1].category, BadInitialCandidatesWarning))
            self.assertEqual(ics.shape, torch.Size([2, 3, 4]))
            with self.assertRaises(RuntimeError):
                initialize_q_batch_nonneg(X=X, Y=Y, n=10)
            # test less than `n` positive acquisition values
            Y = torch.arange(5, device=self.device, dtype=dtype) - 3
            ics = initialize_q_batch_nonneg(X=X, Y=Y, n=2)
            self.assertEqual(ics.shape, torch.Size([2, 3, 4]))
            self.assertEqual(ics.device, X.device)
            self.assertEqual(ics.dtype, X.dtype)
            # check that we chose the point with the positive acquisition value
            self.assertTrue(torch.equal(ics[0], X[-1]) or torch.equal(ics[1], X[-1]))
            # test less than `n` alpha_pos values
            Y = torch.arange(5, device=self.device, dtype=dtype)
            ics = initialize_q_batch_nonneg(X=X, Y=Y, n=2, alpha=1.0)
            self.assertEqual(ics.shape, torch.Size([2, 3, 4]))
            self.assertEqual(ics.device, X.device)
            self.assertEqual(ics.dtype, X.dtype)

    def test_initialize_q_batch(self):
        for dtype in (torch.float, torch.double):
            # basic test
            X = torch.rand(5, 3, 4, device=self.device, dtype=dtype)
            Y = torch.rand(5, device=self.device, dtype=dtype)
            ics = initialize_q_batch(X=X, Y=Y, n=2)
            self.assertEqual(ics.shape, torch.Size([2, 3, 4]))
            self.assertEqual(ics.device, X.device)
            self.assertEqual(ics.dtype, X.dtype)
            # ensure nothing happens if we want all samples
            ics = initialize_q_batch(X=X, Y=Y, n=5)
            self.assertTrue(torch.equal(X, ics))
            # ensure raises correct warning
            Y = torch.zeros(5, device=self.device, dtype=dtype)
            with warnings.catch_warnings(record=True) as w, settings.debug(True):
                ics = initialize_q_batch(X=X, Y=Y, n=2)
                self.assertEqual(len(w), 1)
                self.assertTrue(issubclass(w[-1].category, BadInitialCandidatesWarning))
            self.assertEqual(ics.shape, torch.Size([2, 3, 4]))
            with self.assertRaises(RuntimeError):
                initialize_q_batch(X=X, Y=Y, n=10)

    def test_initialize_q_batch_largeZ(self):
        for dtype in (torch.float, torch.double):
            # testing large eta*Z
            X = torch.rand(5, 3, 4, device=self.device, dtype=dtype)
            Y = torch.tensor([-1e12, 0, 0, 0, 1e12], device=self.device, dtype=dtype)
            ics = initialize_q_batch(X=X, Y=Y, n=2, eta=100)
            self.assertEqual(ics.shape[0], 2)


class TestGenBatchInitialCandidates(BotorchTestCase):
    def test_gen_batch_initial_conditions(self):
        for dtype in (torch.float, torch.double):
            bounds = torch.tensor([[0, 0], [1, 1]], device=self.device, dtype=dtype)
            for nonnegative in (True, False):
                for seed in (None, 1234):
                    batch_initial_conditions = gen_batch_initial_conditions(
                        acq_function=MockAcquisitionFunction(),
                        bounds=bounds,
                        q=1,
                        num_restarts=2,
                        raw_samples=10,
                        options={
                            "nonnegative": nonnegative,
                            "eta": 0.01,
                            "alpha": 0.1,
                            "seed": seed,
                        },
                    )
                    expected_shape = torch.Size([2, 1, 2])
                    self.assertEqual(batch_initial_conditions.shape, expected_shape)
                    self.assertEqual(batch_initial_conditions.device, bounds.device)
                    self.assertEqual(batch_initial_conditions.dtype, bounds.dtype)

    def test_gen_batch_initial_conditions_highdim(self):
        d = 120
        bounds = torch.stack([torch.zeros(d), torch.ones(d)])
        for dtype in (torch.float, torch.double):
            bounds = bounds.to(device=self.device, dtype=dtype)
            for nonnegative in (True, False):
                for seed in (None, 1234):
                    with warnings.catch_warnings(record=True) as ws, settings.debug(
                        True
                    ):
                        batch_initial_conditions = gen_batch_initial_conditions(
                            acq_function=MockAcquisitionFunction(),
                            bounds=bounds,
                            q=10,
                            num_restarts=1,
                            raw_samples=2,
                            options={
                                "nonnegative": nonnegative,
                                "eta": 0.01,
                                "alpha": 0.1,
                                "seed": seed,
                            },
                        )
                        self.assertTrue(
                            any(issubclass(w.category, SamplingWarning) for w in ws)
                        )
                    expected_shape = torch.Size([1, 10, d])
                    self.assertEqual(batch_initial_conditions.shape, expected_shape)
                    self.assertEqual(batch_initial_conditions.device, bounds.device)
                    self.assertEqual(batch_initial_conditions.dtype, bounds.dtype)

    def test_gen_batch_initial_conditions_warning(self):
        for dtype in (torch.float, torch.double):
            bounds = torch.tensor([[0, 0], [1, 1]], device=self.device, dtype=dtype)
            samples = torch.zeros(10, 1, 2, device=self.device, dtype=dtype)
            with ExitStack() as es:
                ws = es.enter_context(warnings.catch_warnings(record=True))
                es.enter_context(settings.debug(True))
                es.enter_context(
                    mock.patch(
                        "botorch.optim.initializers.draw_sobol_samples",
                        return_value=samples,
                    )
                )
                batch_initial_conditions = gen_batch_initial_conditions(
                    acq_function=MockAcquisitionFunction(),
                    bounds=bounds,
                    q=1,
                    num_restarts=2,
                    raw_samples=10,
                    options={"seed": 1234},
                )
                self.assertEqual(len(ws), 1)
                self.assertTrue(
                    any(issubclass(w.category, BadInitialCandidatesWarning) for w in ws)
                )
                self.assertTrue(
                    torch.equal(
                        batch_initial_conditions,
                        torch.zeros(2, 1, 2, device=self.device, dtype=dtype),
                    )
                )


class TestGenOneShotKGInitialConditions(BotorchTestCase):
    def test_gen_one_shot_kg_initial_conditions(self):
        num_fantasies = 8
        num_restarts = 4
        raw_samples = 16
        for dtype in (torch.float, torch.double):
            mean = torch.zeros(1, 1, device=self.device, dtype=dtype)
            mm = MockModel(MockPosterior(mean=mean))
            mock_kg = qKnowledgeGradient(model=mm, num_fantasies=num_fantasies)
            bounds = torch.tensor([[0, 0], [1, 1]], device=self.device, dtype=dtype)
            # test option error
            with self.assertRaises(ValueError):
                gen_one_shot_kg_initial_conditions(
                    acq_function=mock_kg,
                    bounds=bounds,
                    q=1,
                    num_restarts=num_restarts,
                    raw_samples=raw_samples,
                    options={"frac_random": 2.0},
                )
            # test generation logic
            q = 2
            mock_random_ics = torch.rand(num_restarts, q + num_fantasies, 2)
            mock_fantasy_cands = torch.ones(20, 1, 2)
            mock_fantasy_vals = torch.randn(20)
            with ExitStack() as es:
                mock_gbics = es.enter_context(
                    mock.patch(
                        "botorch.optim.initializers.gen_batch_initial_conditions",
                        return_value=mock_random_ics,
                    )
                )
                mock_optacqf = es.enter_context(
                    mock.patch(
                        "botorch.optim.optimize.optimize_acqf",
                        return_value=(mock_fantasy_cands, mock_fantasy_vals),
                    )
                )
                ics = gen_one_shot_kg_initial_conditions(
                    acq_function=mock_kg,
                    bounds=bounds,
                    q=q,
                    num_restarts=num_restarts,
                    raw_samples=raw_samples,
                )
                mock_gbics.assert_called_once()
                mock_optacqf.assert_called_once()
                n_value = int((1 - 0.1) * num_fantasies)
                self.assertTrue(
                    torch.equal(
                        ics[..., :-n_value, :], mock_random_ics[..., :-n_value, :]
                    )
                )
                self.assertTrue(torch.all(ics[..., -n_value:, :] == 1))
