#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import warnings
from unittest.mock import patch

import torch
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.exceptions.warnings import BadInitialCandidatesWarning
from botorch.generation.gen import gen_candidates_scipy
from botorch.models.gp_regression import SingleTaskGP
from botorch.optim.core import scipy_minimize
from botorch.optim.initializers import gen_batch_initial_conditions, initialize_q_batch
from botorch.optim.optimize import optimize_acqf

from botorch.test_utils.mock import fast_optimize, fast_optimize_context_manager
from botorch.utils.testing import BotorchTestCase, MockAcquisitionFunction


class SinAcqusitionFunction(MockAcquisitionFunction):
    """Simple acquisition function with known numerical properties."""

    def __init__(self, *args, **kwargs):  # noqa: D107
        return

    def __call__(self, X):
        return torch.sin(X[..., 0].max(dim=-1).values)


class TestMock(BotorchTestCase):
    def test_fast_optimize_context_manager(self):
        with self.subTest("gen_candidates_scipy"):
            with fast_optimize_context_manager():
                cand, value = gen_candidates_scipy(
                    initial_conditions=torch.tensor([[0.0]]),
                    acquisition_function=SinAcqusitionFunction(),
                )
            # When not using `fast_optimize`, the value is 1.0. With it, the value is
            # around 0.84
            self.assertLess(value.item(), 0.99)

        with self.subTest("scipy_minimize"):
            x = torch.tensor([0.0])

            def closure():
                return torch.sin(x), [torch.cos(x)]

            with fast_optimize_context_manager():
                result = scipy_minimize(closure=closure, parameters={"x": x})
            self.assertEqual(
                result.message, "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT"
            )

        with self.subTest("optimize_acqf"):
            with fast_optimize_context_manager():
                cand, value = optimize_acqf(
                    acq_function=SinAcqusitionFunction(),
                    bounds=torch.tensor([[-2.0], [2.0]]),
                    q=1,
                    num_restarts=32,
                    batch_initial_conditions=torch.tensor([[0.0]]),
                )
            self.assertLess(value.item(), 0.99)

        with self.subTest("gen_batch_initial_conditions"):
            with fast_optimize_context_manager(), patch(
                "botorch.optim.initializers.initialize_q_batch",
                wraps=initialize_q_batch,
            ) as mock_init_q_batch:
                cand, value = optimize_acqf(
                    acq_function=SinAcqusitionFunction(),
                    bounds=torch.tensor([[-2.0], [2.0]]),
                    q=1,
                    num_restarts=32,
                    raw_samples=16,
                )
            self.assertEqual(mock_init_q_batch.call_args[1]["n"], 2)

    @fast_optimize
    def test_decorator(self) -> None:
        model = SingleTaskGP(
            train_X=torch.tensor([[0.0]], dtype=torch.double),
            train_Y=torch.tensor([[0.0]], dtype=torch.double),
        )
        acqf = qKnowledgeGradient(model=model, num_fantasies=64)
        # this is called within gen_one_shot_kg_initial_conditions
        with patch(
            "botorch.optim.initializers.gen_batch_initial_conditions",
            wraps=gen_batch_initial_conditions,
        ) as mock_gen_batch_ics, warnings.catch_warnings():
            warnings.simplefilter("ignore", category=BadInitialCandidatesWarning)
            cand, value = optimize_acqf(
                acq_function=acqf,
                bounds=torch.tensor([[-2.0], [2.0]]),
                q=1,
                num_restarts=32,
                raw_samples=16,
            )

        called_with = mock_gen_batch_ics.call_args[1]
        self.assertEqual(called_with["num_restarts"], 2)
        self.assertEqual(called_with["raw_samples"], 4)

    def test_raises_when_unused(self) -> None:
        with self.assertRaisesRegex(AssertionError, "No mocks were called"):
            with fast_optimize_context_manager():
                pass
