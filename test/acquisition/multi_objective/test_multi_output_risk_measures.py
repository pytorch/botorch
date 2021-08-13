#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.testing import BotorchTestCase
from botorch_fb.acquisition.multi_output_risk_measures import (
    IndependentCVaR,
    IndependentVaR,
    MultiOutputRiskMeasureMCObjective,
    MultiOutputWorstCase,
    MVaR,
)
from torch import Tensor


class NotSoAbstractMORiskMeasure(MultiOutputRiskMeasureMCObjective):
    def forward(self, samples: Tensor, X: Optional[Tensor] = None) -> Tensor:
        prepared_samples = self._prepare_samples(samples)
        return prepared_samples.sum(dim=-2)


class TestMultiOutputRiskMeasureMCObjective(BotorchTestCase):
    def test_multi_output_risk_measure_mc_objective(self):
        # abstract raises
        with self.assertRaises(TypeError):
            MultiOutputRiskMeasureMCObjective(n_w=3)

        for dtype in (torch.float, torch.double):
            samples = torch.tensor(
                [
                    [
                        [1.0, 1.2],
                        [0.5, 0.7],
                        [2.0, 2.2],
                        [3.0, 3.4],
                        [1.0, 1.2],
                        [5.0, 5.6],
                    ]
                ],
                device=self.device,
                dtype=dtype,
            )
            obj = NotSoAbstractMORiskMeasure(n_w=3, weights=None)
            # test _prepare_samples
            expected_samples = samples.view(1, 2, 3, 2)
            prepared_samples = obj._prepare_samples(samples)
            self.assertTrue(torch.equal(prepared_samples, expected_samples))
            # test batches
            samples = torch.rand(5, 3, 6, 3, device=self.device, dtype=dtype)
            expected_samples = samples.view(5, 3, 2, 3, 3)
            prepared_samples = obj._prepare_samples(samples)
            self.assertTrue(torch.equal(prepared_samples, expected_samples))
            # negating with weights
            obj = NotSoAbstractMORiskMeasure(
                n_w=3,
                weights=torch.tensor(
                    [-1.0, -1.0, -1.0], device=self.device, dtype=dtype
                ),
            )
            prepared_samples = obj._prepare_samples(samples)
            self.assertTrue(torch.equal(prepared_samples, -expected_samples))


class TestIndependentCVaR(BotorchTestCase):
    def test_independent_cvar(self):
        obj = IndependentCVaR(alpha=0.5, n_w=3)
        self.assertEqual(obj.alpha_idx, 1)
        with self.assertRaises(ValueError):
            IndependentCVaR(alpha=3, n_w=3)
        for dtype in (torch.float, torch.double):
            obj = IndependentCVaR(alpha=0.5, n_w=3)
            samples = torch.tensor(
                [
                    [
                        [1.0, 1.2],
                        [0.5, 0.7],
                        [2.0, 2.2],
                        [3.0, 1.2],
                        [1.0, 7.2],
                        [5.0, 5.8],
                    ]
                ],
                device=self.device,
                dtype=dtype,
            )
            rm_samples = obj(samples)
            self.assertTrue(
                torch.allclose(
                    rm_samples,
                    torch.tensor(
                        [[[0.75, 0.95], [2.0, 3.5]]], device=self.device, dtype=dtype
                    ),
                )
            )
            # w/ first output negated
            obj.weights = torch.tensor([-1.0, 1.0], device=self.device, dtype=dtype)
            rm_samples = obj(samples)
            self.assertTrue(
                torch.allclose(
                    rm_samples,
                    torch.tensor(
                        [[[-1.5, 0.95], [-4.0, 3.5]]], device=self.device, dtype=dtype
                    ),
                )
            )


class TestIndependentVaR(BotorchTestCase):
    def test_independent_var(self):
        for dtype in (torch.float, torch.double):
            obj = IndependentVaR(alpha=0.5, n_w=3)
            samples = torch.tensor(
                [
                    [
                        [1.0, 3.2],
                        [0.5, 0.7],
                        [2.0, 2.2],
                        [3.0, 1.2],
                        [1.0, 7.2],
                        [5.0, 5.8],
                    ]
                ],
                device=self.device,
                dtype=dtype,
            )
            rm_samples = obj(samples)
            self.assertTrue(
                torch.equal(
                    rm_samples,
                    torch.tensor(
                        [[[1.0, 2.2], [3.0, 5.8]]], device=self.device, dtype=dtype
                    ),
                )
            )
            # w/ weights
            obj.weights = torch.tensor([0.5, -1.0], device=self.device, dtype=dtype)
            rm_samples = obj(samples)
            self.assertTrue(
                torch.equal(
                    rm_samples,
                    torch.tensor(
                        [[[0.5, -2.2], [1.5, -5.8]]], device=self.device, dtype=dtype
                    ),
                )
            )


class TestMultiOutputWorstCase(BotorchTestCase):
    def test_multi_output_worst_case(self):
        for dtype in (torch.float, torch.double):
            obj = MultiOutputWorstCase(n_w=3)
            samples = torch.tensor(
                [
                    [
                        [1.0, 3.2],
                        [5.5, 0.7],
                        [2.0, 2.2],
                        [3.0, 1.2],
                        [5.0, 7.2],
                        [5.0, 5.8],
                    ]
                ],
                device=self.device,
                dtype=dtype,
            )
            rm_samples = obj(samples)
            self.assertTrue(
                torch.equal(
                    rm_samples,
                    torch.tensor(
                        [[[1.0, 0.7], [3.0, 1.2]]], device=self.device, dtype=dtype
                    ),
                )
            )
            # w/ weights
            obj.weights = torch.tensor([-1.0, 2.0], device=self.device, dtype=dtype)
            rm_samples = obj(samples)
            self.assertTrue(
                torch.equal(
                    rm_samples,
                    torch.tensor(
                        [[[-5.5, 1.4], [-5.0, 2.4]]], device=self.device, dtype=dtype
                    ),
                )
            )


class TestMVaR(BotorchTestCase):
    def test_mvar(self):
        with self.assertRaises(ValueError):
            MVaR(n_w=5, alpha=3.0)

        def set_equals(t1: Tensor, t2: Tensor) -> bool:
            r"""Check if two `k x m`-dim tensors are equivalent after possibly
            reordering the `k` dimension. Ignores duplicate entries.
            """
            t1 = t1.unique(dim=0)
            t2 = t2.unique(dim=0)
            if t1.shape != t2.shape:
                return False
            equals_sum = (t1.unsqueeze(-2) == t2).all(dim=-1).sum(dim=-1)
            return torch.equal(equals_sum, torch.ones_like(equals_sum))

        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            mvar = MVaR(n_w=5, alpha=0.6)
            # a simple negatively correlated example
            Y = torch.stack(
                [torch.linspace(1, 5, 5), torch.linspace(5, 1, 5)],
                dim=-1,
            ).to(**tkwargs)
            expected_set = torch.stack(
                [torch.linspace(1, 3, 3), torch.linspace(3, 1, 3)],
                dim=-1,
            )
            # check that both versions produce the correct set
            cpu_mvar = mvar.get_mvar_set_cpu(Y)  # For 2d input, returns k x m
            gpu_mvar = mvar.get_mvar_set_gpu(Y)[0]  # returns a batch list of k x m
            self.assertTrue(set_equals(cpu_mvar, gpu_mvar))
            self.assertTrue(set_equals(cpu_mvar, expected_set))
            # check that the `filter_dominated` works correctly
            mvar = MVaR(
                n_w=5,
                alpha=0.4,
                filter_dominated=False,
            )
            # negating the input to treat large values as undesirable
            Y = -torch.tensor(
                [
                    [1, 4],
                    [2, 3],
                    [3, 2],
                    [4, 1],
                    [3.5, 3.5],
                ],
                **tkwargs
            )
            cpu_mvar = mvar.get_mvar_set_cpu(Y)
            gpu_mvar = mvar.get_mvar_set_gpu(Y)[0]
            self.assertTrue(set_equals(cpu_mvar, gpu_mvar))
            # negating here as well
            expected_w_dominated = -torch.tensor(
                [
                    [2, 4],
                    [3, 3],
                    [3.5, 3],
                    [3, 3.5],
                    [4, 2],
                ],
                **tkwargs
            )
            self.assertTrue(set_equals(cpu_mvar, expected_w_dominated))
            expected_non_dominated = expected_w_dominated[
                is_non_dominated(expected_w_dominated)
            ]
            mvar.filter_dominated = True
            cpu_mvar = mvar.get_mvar_set_cpu(Y)
            gpu_mvar = mvar.get_mvar_set_gpu(Y)[0]
            self.assertTrue(set_equals(cpu_mvar, gpu_mvar))
            self.assertTrue(set_equals(cpu_mvar, expected_non_dominated))

            # test batched w/ random input
            mvar = MVaR(
                n_w=10,
                alpha=0.5,
                filter_dominated=False,
            )
            Y = torch.rand(4, 10, 2, **tkwargs)
            cpu_mvar = mvar.get_mvar_set_cpu(Y)
            gpu_mvar = mvar.get_mvar_set_gpu(Y)
            # check that the two agree
            self.assertTrue(
                all([set_equals(cpu_mvar[i], gpu_mvar[i]) for i in range(4)])
            )
            # check that the MVaR is dominated by `alpha` fraction (maximization).
            dominated_count = (Y[0].unsqueeze(-2) >= cpu_mvar[0]).all(dim=-1).sum(dim=0)
            expected_count = (
                torch.ones(cpu_mvar[0].shape[0], device=self.device, dtype=torch.long)
                * 5
            )
            self.assertTrue(torch.equal(dominated_count, expected_count))

            # test forward pass
            # with `expectation=True`
            mvar = MVaR(
                n_w=10,
                alpha=0.5,
                expectation=True,
            )
            samples = torch.rand(2, 20, 2, **tkwargs)
            mvar_exp = mvar(samples)
            expected = [
                mvar.get_mvar_set_cpu(Y).mean(dim=0) for Y in samples.view(4, 10, 2)
            ]
            self.assertTrue(torch.equal(mvar_exp, torch.stack(expected).view(2, 2, 2)))

            # m > 2
            samples = torch.rand(2, 20, 3, **tkwargs)
            mvar_exp = mvar(samples)
            expected = [
                mvar.get_mvar_set_gpu(Y)[0].mean(dim=0) for Y in samples.view(4, 10, 3)
            ]
            self.assertTrue(torch.equal(mvar_exp, torch.stack(expected).view(2, 2, 3)))

            # with `expectation=False`
            mvar = MVaR(
                n_w=10,
                alpha=0.5,
                expectation=False,
                pad_to_n_w=True,
            )
            samples = torch.rand(2, 20, 2, **tkwargs)
            mvar_vals = mvar(samples)
            self.assertTrue(mvar_vals.shape == samples.shape)
            expected = [mvar.get_mvar_set_cpu(Y) for Y in samples.view(4, 10, 2)]
            for i in range(4):
                batch_idx = i // 2
                q_idx_start = 10 * (i % 2)
                expected_ = expected[i]
                # check that the actual values are there
                self.assertTrue(
                    set_equals(
                        mvar_vals[
                            batch_idx, q_idx_start : q_idx_start + expected_.shape[0]
                        ],
                        expected_,
                    )
                )
                # check for correct padding
                self.assertTrue(
                    torch.equal(
                        mvar_vals[
                            batch_idx,
                            q_idx_start + expected_.shape[0] : q_idx_start + 10,
                        ],
                        mvar_vals[
                            batch_idx, q_idx_start + expected_.shape[0] - 1
                        ].expand(10 - expected_.shape[0], -1),
                    )
                )

            # Test the no-exact alpha level points case.
            # This happens when there are duplicates in the input.
            Y = torch.ones(10, 2, **tkwargs)
            cpu_mvar = mvar.get_mvar_set_cpu(Y)
            gpu_mvar = mvar.get_mvar_set_gpu(Y)[0]
            self.assertTrue(torch.equal(cpu_mvar, Y[:1]))
            self.assertTrue(torch.equal(cpu_mvar, Y[:1]))

            # TODO: Test grad support once properly implemented.
