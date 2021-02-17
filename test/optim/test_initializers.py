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
from botorch.acquisition.analytic import PosteriorMean
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.exceptions import BadInitialCandidatesWarning, SamplingWarning
from botorch.models import SingleTaskGP
from botorch.optim import initialize_q_batch, initialize_q_batch_nonneg
from botorch.optim.initializers import (
    gen_batch_initial_conditions,
    gen_one_shot_kg_initial_conditions,
    gen_value_function_initial_conditions,
)
from botorch.sampling import IIDNormalSampler
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
            for batch_shape in (torch.Size(), [3, 2], (2,), torch.Size([2, 3, 4]), []):
                # basic test
                X = torch.rand(5, *batch_shape, 3, 4, device=self.device, dtype=dtype)
                Y = torch.rand(5, *batch_shape, device=self.device, dtype=dtype)
                ics = initialize_q_batch(X=X, Y=Y, n=2)
                self.assertEqual(ics.shape, torch.Size([2, *batch_shape, 3, 4]))
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
                    self.assertTrue(
                        issubclass(w[-1].category, BadInitialCandidatesWarning)
                    )
                self.assertEqual(ics.shape, torch.Size([2, *batch_shape, 3, 4]))
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
                    mock_acqf = MockAcquisitionFunction()
                    for init_batch_limit in (None, 1):
                        mock_acqf = MockAcquisitionFunction()
                        with mock.patch.object(
                            MockAcquisitionFunction,
                            "__call__",
                            wraps=mock_acqf.__call__,
                        ) as mock_acqf_call:
                            batch_initial_conditions = gen_batch_initial_conditions(
                                acq_function=mock_acqf,
                                bounds=bounds,
                                q=1,
                                num_restarts=2,
                                raw_samples=10,
                                options={
                                    "nonnegative": nonnegative,
                                    "eta": 0.01,
                                    "alpha": 0.1,
                                    "seed": seed,
                                    "init_batch_limit": init_batch_limit,
                                },
                            )
                            expected_shape = torch.Size([2, 1, 2])
                            self.assertEqual(
                                batch_initial_conditions.shape, expected_shape
                            )
                            self.assertEqual(
                                batch_initial_conditions.device, bounds.device
                            )
                            self.assertEqual(
                                batch_initial_conditions.dtype, bounds.dtype
                            )
                            batch_shape = (
                                torch.Size([])
                                if init_batch_limit is None
                                else torch.Size([init_batch_limit])
                            )
                            raw_samps = mock_acqf_call.call_args[0][0]
                            batch_shape = (
                                torch.Size([10])
                                if init_batch_limit is None
                                else torch.Size([init_batch_limit])
                            )
                            expected_raw_samps_shape = batch_shape + torch.Size([1, 2])
                            self.assertEqual(raw_samps.shape, expected_raw_samps_shape)

    def test_gen_batch_initial_conditions_highdim(self):
        d = 2200  # 2200 * 10 (q) > 21201 (sobol max dim)
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


class TestGenValueFunctionInitialConditions(BotorchTestCase):
    def test_gen_value_function_initial_conditions(self):
        num_fantasies = 2
        num_solutions = 3
        num_restarts = 4
        raw_samples = 5
        n_train = 6
        dim = 2
        dtype = torch.float
        # run a thorough test with dtype float
        train_X = torch.rand(n_train, dim, device=self.device, dtype=dtype)
        train_Y = torch.rand(n_train, 1, device=self.device, dtype=dtype)
        model = SingleTaskGP(train_X, train_Y)
        fant_X = torch.rand(num_solutions, 1, dim, device=self.device, dtype=dtype)
        fantasy_model = model.fantasize(fant_X, IIDNormalSampler(num_fantasies))
        bounds = torch.tensor([[0, 0], [1, 1]], device=self.device, dtype=dtype)
        value_function = PosteriorMean(fantasy_model)
        # test option error
        with self.assertRaises(ValueError):
            gen_value_function_initial_conditions(
                acq_function=value_function,
                bounds=bounds,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                current_model=model,
                options={"frac_random": 2.0},
            )
        # test output shape
        ics = gen_value_function_initial_conditions(
            acq_function=value_function,
            bounds=bounds,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            current_model=model,
        )
        self.assertEqual(
            ics.shape, torch.Size([num_restarts, num_fantasies, num_solutions, 1, dim])
        )
        # test bounds
        self.assertTrue(torch.all(ics >= bounds[0]))
        self.assertTrue(torch.all(ics <= bounds[1]))
        # test dtype
        self.assertEqual(dtype, ics.dtype)

        # minimal test cases for when all raw samples are random, with dtype double
        dtype = torch.double
        n_train = 2
        dim = 1
        num_solutions = 1
        train_X = torch.rand(n_train, dim, device=self.device, dtype=dtype)
        train_Y = torch.rand(n_train, 1, device=self.device, dtype=dtype)
        model = SingleTaskGP(train_X, train_Y)
        fant_X = torch.rand(1, 1, dim, device=self.device, dtype=dtype)
        fantasy_model = model.fantasize(fant_X, IIDNormalSampler(num_fantasies))
        bounds = torch.tensor([[0], [1]], device=self.device, dtype=dtype)
        value_function = PosteriorMean(fantasy_model)
        ics = gen_value_function_initial_conditions(
            acq_function=value_function,
            bounds=bounds,
            num_restarts=1,
            raw_samples=1,
            current_model=model,
            options={"frac_random": 0.99},
        )
        self.assertEqual(
            ics.shape, torch.Size([1, num_fantasies, num_solutions, 1, dim])
        )
        # test bounds
        self.assertTrue(torch.all(ics >= bounds[0]))
        self.assertTrue(torch.all(ics <= bounds[1]))
        # test dtype
        self.assertEqual(dtype, ics.dtype)
