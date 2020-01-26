#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
from unittest import mock

import torch
from botorch.acquisition.acquisition import OneShotAcquisitionFunction
from botorch.optim.optimize import optimize_acqf, optimize_acqf_cyclic
from botorch.utils.testing import BotorchTestCase, MockAcquisitionFunction
from torch import Tensor


class MockOneShotAcquisitionFunction(
    MockAcquisitionFunction, OneShotAcquisitionFunction
):
    def __init__(self, num_fantasies=2):
        super().__init__()
        self.num_fantasies = num_fantasies

    def get_augmented_q_batch_size(self, q: int) -> int:
        return q + self.num_fantasies

    def extract_candidates(self, X_full: Tensor) -> Tensor:
        return X_full[..., : -self.num_fantasies, :]

    def forward(self, X):
        pass


def rounding_func(X: Tensor) -> Tensor:
    batch_shape, d = X.shape[:-1], X.shape[-1]
    X_round = torch.stack([x.round() for x in X.view(-1, d)])
    return X_round.view(*batch_shape, d)


class TestOptimizeAcqf(BotorchTestCase):
    @mock.patch("botorch.optim.optimize.gen_batch_initial_conditions")
    @mock.patch("botorch.optim.optimize.gen_candidates_scipy")
    def test_optimize_acqf_joint(
        self, mock_gen_candidates, mock_gen_batch_initial_conditions
    ):
        q = 3
        num_restarts = 2
        raw_samples = 10
        options = {}
        mock_acq_function = MockAcquisitionFunction()
        cnt = 1
        for dtype in (torch.float, torch.double):
            mock_gen_batch_initial_conditions.return_value = torch.zeros(
                num_restarts, q, 3, device=self.device, dtype=dtype
            )
            base_cand = torch.ones(1, q, 3, device=self.device, dtype=dtype)
            mock_candidates = torch.cat(
                [i * base_cand for i in range(num_restarts)], dim=0
            )
            mock_acq_values = num_restarts - torch.arange(
                num_restarts, device=self.device, dtype=dtype
            )
            mock_gen_candidates.return_value = (mock_candidates, mock_acq_values)
            bounds = torch.stack(
                [
                    torch.zeros(3, device=self.device, dtype=dtype),
                    4 * torch.ones(3, device=self.device, dtype=dtype),
                ]
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
                batch_initial_conditions=torch.zeros(
                    num_restarts, q, 3, device=self.device, dtype=dtype
                ),
            )
            self.assertTrue(torch.equal(candidates, mock_candidates))
            self.assertTrue(torch.equal(acq_vals, mock_acq_values))
            self.assertEqual(mock_gen_batch_initial_conditions.call_count, cnt)
            cnt += 1

        # test OneShotAcquisitionFunction
        mock_acq_function = MockOneShotAcquisitionFunction()
        candidates, acq_vals = optimize_acqf(
            acq_function=mock_acq_function,
            bounds=bounds,
            q=q,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            options=options,
        )
        self.assertTrue(
            torch.equal(
                candidates, mock_acq_function.extract_candidates(mock_candidates[0])
            )
        )
        self.assertTrue(torch.equal(acq_vals, mock_acq_values[0]))

    @mock.patch("botorch.optim.optimize.gen_batch_initial_conditions")
    @mock.patch("botorch.optim.optimize.gen_candidates_scipy")
    def test_optimize_acqf_sequential(
        self, mock_gen_candidates_scipy, mock_gen_batch_initial_conditions
    ):
        q = 3
        num_restarts = 2
        raw_samples = 10
        options = {}
        for dtype in (torch.float, torch.double):
            mock_acq_function = MockAcquisitionFunction()
            mock_gen_batch_initial_conditions.side_effect = [
                torch.zeros(num_restarts, device=self.device, dtype=dtype)
                for _ in range(q)
            ]
            gcs_return_vals = [
                (
                    torch.tensor([[[1.1, 2.1, 3.1]]], device=self.device, dtype=dtype),
                    torch.tensor([i], device=self.device, dtype=dtype),
                )
                for i in range(q)
            ]
            mock_gen_candidates_scipy.side_effect = gcs_return_vals
            expected_candidates = torch.cat(
                [rv[0][0] for rv in gcs_return_vals], dim=-2
            ).round()
            bounds = torch.stack(
                [
                    torch.zeros(3, device=self.device, dtype=dtype),
                    4 * torch.ones(3, device=self.device, dtype=dtype),
                ]
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
        # verify error when using a OneShotAcquisitionFunction
        with self.assertRaises(NotImplementedError):
            optimize_acqf(
                acq_function=mock.Mock(spec=OneShotAcquisitionFunction),
                bounds=bounds,
                q=q,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                sequential=True,
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


class TestOptimizeAcqfCyclic(BotorchTestCase):
    @mock.patch("botorch.optim.optimize.optimize_acqf")  # noqa: C901
    def test_optimize_acqf_cyclic(self, mock_optimize_acqf):
        num_restarts = 2
        raw_samples = 10
        num_cycles = 2
        options = {}
        tkwargs = {"device": self.device}
        bounds = torch.stack([torch.zeros(3), 4 * torch.ones(3)])
        inequality_constraints = [
            [torch.tensor([3]), torch.tensor([4]), torch.tensor(5)]
        ]
        mock_acq_function = MockAcquisitionFunction()
        for q, dtype in itertools.product([1, 3], (torch.float, torch.double)):
            inequality_constraints[0] = [
                t.to(**tkwargs) for t in inequality_constraints[0]
            ]
            mock_optimize_acqf.reset_mock()
            tkwargs["dtype"] = dtype
            bounds = bounds.to(**tkwargs)
            candidate_rvs = []
            acq_val_rvs = []
            for cycle_j in range(num_cycles):
                gcs_return_vals = [
                    (torch.rand(1, 3, **tkwargs), torch.rand(1, **tkwargs))
                    for _ in range(q)
                ]
                if cycle_j == 0:
                    # return `q` candidates for first call
                    candidate_rvs.append(
                        torch.cat([rv[0] for rv in gcs_return_vals], dim=-2)
                    )
                    acq_val_rvs.append(torch.cat([rv[1] for rv in gcs_return_vals]))
                else:
                    # return 1 candidate for subsequent calls
                    for rv in gcs_return_vals:
                        candidate_rvs.append(rv[0])
                        acq_val_rvs.append(rv[1])
            mock_optimize_acqf.side_effect = list(zip(candidate_rvs, acq_val_rvs))
            orig_candidates = candidate_rvs[0].clone()
            # wrap the set_X_pending method for checking that call arguments
            with mock.patch.object(
                MockAcquisitionFunction,
                "set_X_pending",
                wraps=mock_acq_function.set_X_pending,
            ) as mock_set_X_pending:
                candidates, acq_value = optimize_acqf_cyclic(
                    acq_function=mock_acq_function,
                    bounds=bounds,
                    q=q,
                    num_restarts=num_restarts,
                    raw_samples=raw_samples,
                    options=options,
                    inequality_constraints=inequality_constraints,
                    post_processing_func=rounding_func,
                    cyclic_options={"maxiter": num_cycles},
                )
                # check that X_pending is set correctly in cyclic optimization
                if q > 1:
                    x_pending_call_args_list = mock_set_X_pending.call_args_list
                    idxr = torch.ones(q, dtype=torch.bool, device=self.device)
                    for i in range(len(x_pending_call_args_list) - 1):
                        idxr[i] = 0
                        self.assertTrue(
                            torch.equal(
                                x_pending_call_args_list[i][0][0], orig_candidates[idxr]
                            )
                        )
                        idxr[i] = 1
                        orig_candidates[i] = candidate_rvs[i + 1]
                    # check reset to base_X_pendingg
                    self.assertIsNone(x_pending_call_args_list[-1][0][0])
                else:
                    mock_set_X_pending.assert_not_called()
            # check final candidates
            expected_candidates = (
                torch.cat(candidate_rvs[-q:], dim=0) if q > 1 else candidate_rvs[0]
            )
            self.assertTrue(torch.equal(candidates, expected_candidates))
            # check call arguments for optimize_acqf
            call_args_list = mock_optimize_acqf.call_args_list
            expected_call_args = {
                "acq_function": mock_acq_function,
                "bounds": bounds,
                "num_restarts": num_restarts,
                "raw_samples": raw_samples,
                "options": options,
                "inequality_constraints": inequality_constraints,
                "equality_constraints": None,
                "fixed_features": None,
                "post_processing_func": rounding_func,
                "return_best_only": True,
                "sequential": True,
            }
            orig_candidates = candidate_rvs[0].clone()
            for i in range(len(call_args_list)):
                if i == 0:
                    # first cycle
                    expected_call_args.update(
                        {"batch_initial_conditions": None, "q": q}
                    )
                else:
                    expected_call_args.update(
                        {"batch_initial_conditions": orig_candidates[i - 1 : i], "q": 1}
                    )
                    orig_candidates[i - 1] = candidate_rvs[i]
                for k, v in call_args_list[i][1].items():
                    if torch.is_tensor(v):
                        self.assertTrue(torch.equal(expected_call_args[k], v))
                    elif k == "acq_function":
                        self.assertIsInstance(
                            mock_acq_function, MockAcquisitionFunction
                        )
                    else:
                        self.assertEqual(expected_call_args[k], v)
