#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
from botorch.acquisition.objective import LinearMCObjective
from botorch.acquisition.risk_measures import (
    CVaR,
    Expectation,
    RiskMeasureMCObjective,
    VaR,
    WorstCase,
)
from botorch.utils.testing import BotorchTestCase
from torch import Tensor


class NotSoAbstractRiskMeasure(RiskMeasureMCObjective):
    def forward(self, samples: Tensor, X: Tensor | None = None) -> Tensor:
        prepared_samples = self._prepare_samples(samples)
        return prepared_samples.sum(dim=-1)


class TestRiskMeasureMCObjective(BotorchTestCase):
    def test_risk_measure_mc_objective(self):
        # abstract raises
        with self.assertRaises(TypeError):
            RiskMeasureMCObjective(n_w=3)

        for dtype in (torch.float, torch.double):
            samples = torch.tensor(
                [[[1.0], [0.5], [2.0], [3.0], [1.0], [5.0]]],
                device=self.device,
                dtype=dtype,
            )
            obj = NotSoAbstractRiskMeasure(n_w=3)
            # MO samples without weights
            with self.assertRaises(RuntimeError):
                obj(torch.ones(3, 2, device=self.device, dtype=dtype))
            # test _prepare_samples
            expected_samples = torch.tensor(
                [[[1.0, 0.5, 2.0], [3.0, 1.0, 5.0]]],
                device=self.device,
                dtype=dtype,
            )
            prepared_samples = obj._prepare_samples(samples)
            self.assertTrue(torch.equal(prepared_samples, expected_samples))
            # test batches
            samples = torch.rand(5, 3, 6, 1, device=self.device, dtype=dtype)
            expected_samples = samples.view(5, 3, 2, 3)
            prepared_samples = obj._prepare_samples(samples)
            self.assertTrue(torch.equal(prepared_samples, expected_samples))
            # negating with preprocessing function.
            obj = NotSoAbstractRiskMeasure(
                n_w=3,
                preprocessing_function=LinearMCObjective(
                    weights=torch.tensor([-1.0], device=self.device, dtype=dtype)
                ),
            )
            prepared_samples = obj._prepare_samples(samples)
            self.assertTrue(torch.equal(prepared_samples, -expected_samples))
            # MO with weights
            obj = NotSoAbstractRiskMeasure(
                n_w=2,
                preprocessing_function=LinearMCObjective(
                    weights=torch.tensor([1.0, 2.0], device=self.device, dtype=dtype)
                ),
            )
            samples = torch.tensor(
                [
                    [
                        [1.0, 2.0],
                        [0.5, 0.7],
                        [2.0, 1.5],
                        [3.0, 4.0],
                        [1.0, 0.0],
                        [5.0, 3.0],
                    ]
                ],
                device=self.device,
                dtype=dtype,
            )
            expected_samples = torch.tensor(
                [[[5.0, 1.9], [5.0, 11.0], [1.0, 11.0]]],
                device=self.device,
                dtype=dtype,
            )
            prepared_samples = obj._prepare_samples(samples)
            self.assertTrue(torch.equal(prepared_samples, expected_samples))


class TestCVaR(BotorchTestCase):
    def test_cvar(self):
        obj = CVaR(alpha=0.5, n_w=3)
        self.assertEqual(obj.alpha_idx, 1)
        with self.assertRaises(ValueError):
            CVaR(alpha=3, n_w=3)
        for dtype in (torch.float, torch.double):
            obj = CVaR(alpha=0.5, n_w=3)
            samples = torch.tensor(
                [[[1.0], [0.5], [2.0], [3.0], [1.0], [5.0]]],
                device=self.device,
                dtype=dtype,
            )
            rm_samples = obj(samples)
            self.assertTrue(
                torch.equal(
                    rm_samples,
                    torch.tensor([[0.75, 2.0]], device=self.device, dtype=dtype),
                )
            )
            # w/ preprocessing function
            obj = CVaR(
                alpha=0.5,
                n_w=3,
                preprocessing_function=LinearMCObjective(
                    weights=torch.tensor([-1.0], device=self.device, dtype=dtype)
                ),
            )
            rm_samples = obj(samples)
            self.assertTrue(
                torch.equal(
                    rm_samples,
                    torch.tensor([[-1.5, -4.0]], device=self.device, dtype=dtype),
                )
            )


class TestVaR(BotorchTestCase):
    def test_var(self):
        for dtype in (torch.float, torch.double):
            obj = VaR(alpha=0.5, n_w=3)
            samples = torch.tensor(
                [[[1.0], [0.5], [2.0], [3.0], [1.0], [5.0]]],
                device=self.device,
                dtype=dtype,
            )
            rm_samples = obj(samples)
            self.assertTrue(
                torch.equal(
                    rm_samples,
                    torch.tensor([[1.0, 3.0]], device=self.device, dtype=dtype),
                )
            )
            # w/ preprocessing function
            obj = VaR(
                alpha=0.5,
                n_w=3,
                preprocessing_function=LinearMCObjective(
                    weights=torch.tensor([-1.0], device=self.device, dtype=dtype)
                ),
            )
            rm_samples = obj(samples)
            self.assertTrue(
                torch.equal(
                    rm_samples,
                    torch.tensor([[-1.0, -3.0]], device=self.device, dtype=dtype),
                )
            )


class TestWorstCase(BotorchTestCase):
    def test_worst_case(self):
        for dtype in (torch.float, torch.double):
            obj = WorstCase(n_w=3)
            samples = torch.tensor(
                [[[1.0], [0.5], [2.0], [3.0], [1.0], [5.0]]],
                device=self.device,
                dtype=dtype,
            )
            rm_samples = obj(samples)
            self.assertTrue(
                torch.equal(
                    rm_samples,
                    torch.tensor([[0.5, 1.0]], device=self.device, dtype=dtype),
                )
            )
            # w/ preprocessing function
            obj = WorstCase(
                n_w=3,
                preprocessing_function=LinearMCObjective(
                    weights=torch.tensor([-1.0], device=self.device, dtype=dtype)
                ),
            )
            rm_samples = obj(samples)
            self.assertTrue(
                torch.equal(
                    rm_samples,
                    torch.tensor([[-2.0, -5.0]], device=self.device, dtype=dtype),
                )
            )


class TestExpectation(BotorchTestCase):
    def test_expectation(self):
        for dtype in (torch.float, torch.double):
            obj = Expectation(n_w=3)
            samples = torch.tensor(
                [[[1.0], [0.5], [1.5], [3.0], [1.0], [5.0]]],
                device=self.device,
                dtype=dtype,
            )
            rm_samples = obj(samples)
            self.assertTrue(
                torch.equal(
                    rm_samples,
                    torch.tensor([[1.0, 3.0]], device=self.device, dtype=dtype),
                )
            )
            # w/ preprocessing function
            samples = torch.tensor(
                [
                    [
                        [1.0, 3.0],
                        [0.5, 1.0],
                        [1.5, 2.0],
                        [3.0, 1.0],
                        [1.0, 2.0],
                        [5.0, 3.0],
                    ]
                ],
                device=self.device,
                dtype=dtype,
            )
            obj = Expectation(
                n_w=3,
                preprocessing_function=LinearMCObjective(
                    weights=torch.tensor([-1.0, 2.0], device=self.device, dtype=dtype)
                ),
            )
            rm_samples = obj(samples)
            self.assertTrue(
                torch.equal(
                    rm_samples,
                    torch.tensor([[3.0, 1.0]], device=self.device, dtype=dtype),
                )
            )
