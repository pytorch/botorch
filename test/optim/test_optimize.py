#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import warnings
from inspect import signature
from unittest import mock

import numpy as np
import torch
from botorch.acquisition.acquisition import (
    AcquisitionFunction,
    OneShotAcquisitionFunction,
)
from botorch.exceptions import InputDataError, UnsupportedError
from botorch.generation.gen import gen_candidates_scipy, gen_candidates_torch
from botorch.optim.optimize import (
    _filter_infeasible,
    _filter_invalid,
    _gen_batch_initial_conditions_local_search,
    _generate_neighbors,
    optimize_acqf,
    optimize_acqf_cyclic,
    optimize_acqf_discrete,
    optimize_acqf_discrete_local_search,
    optimize_acqf_list,
    optimize_acqf_mixed,
)
from botorch.optim.parameter_constraints import (
    _arrayify,
    _make_f_and_grad_nonlinear_inequality_constraints,
)
from botorch.utils.testing import BotorchTestCase, MockAcquisitionFunction
from scipy.optimize import OptimizeResult
from torch import Tensor


class MockOneShotAcquisitionFunction(
    MockAcquisitionFunction, OneShotAcquisitionFunction
):
    def __init__(self, num_fantasies=2):
        r"""
        Args:
            num_fantasies: The number of fantasies.
        """
        super().__init__()
        self.num_fantasies = num_fantasies

    def get_augmented_q_batch_size(self, q: int) -> int:
        return q + self.num_fantasies

    def extract_candidates(self, X_full: Tensor) -> Tensor:
        return X_full[..., : -self.num_fantasies, :]

    def forward(self, X):
        pass


class SquaredAcquisitionFunction(AcquisitionFunction):
    def __init__(self, model=None):  # noqa: D107
        super().__init__(model=model)

    def forward(self, X):
        return torch.norm(X, dim=-1).squeeze(-1)


class MockOneShotEvaluateAcquisitionFunction(MockOneShotAcquisitionFunction):
    def evaluate(self, X: Tensor, bounds: Tensor):
        return X.sum()


class SinOneOverXAcqusitionFunction(MockAcquisitionFunction):
    """
    Acquisition function for sin(1/x).

    This is useful for testing because it behaves pathologically only zero, so
    optimization is likely to fail when initializing near zero but not
    elsewhere.
    """

    def __call__(self, X):
        return torch.sin(1 / X[..., 0].max(dim=-1).values)


def rounding_func(X: Tensor) -> Tensor:
    batch_shape, d = X.shape[:-1], X.shape[-1]
    X_round = torch.stack([x.round() for x in X.view(-1, d)])
    return X_round.view(*batch_shape, d)


class TestOptimizeAcqf(BotorchTestCase):
    @mock.patch("botorch.generation.gen.gen_candidates_torch")
    @mock.patch("botorch.optim.optimize.gen_batch_initial_conditions")
    @mock.patch("botorch.optim.optimize.gen_candidates_scipy")
    @mock.patch("botorch.optim.utils.common.signature")
    def test_optimize_acqf_joint(
        self,
        mock_signature,
        mock_gen_candidates_scipy,
        mock_gen_batch_initial_conditions,
        mock_gen_candidates_torch,
    ):
        q = 3
        num_restarts = 2
        raw_samples = 10
        options = {}
        mock_acq_function = MockAcquisitionFunction()
        cnt = 0

        for dtype in (torch.float, torch.double):
            for mock_gen_candidates in (
                mock_gen_candidates_scipy,
                mock_gen_candidates_torch,
            ):
                if mock_gen_candidates == mock_gen_candidates_torch:
                    mock_signature.return_value = signature(gen_candidates_torch)
                else:
                    mock_signature.return_value = signature(gen_candidates_scipy)

                mock_gen_batch_initial_conditions.return_value = torch.zeros(
                    num_restarts, q, 3, device=self.device, dtype=dtype
                )
                base_cand = torch.arange(3, device=self.device, dtype=dtype).expand(
                    1, q, 3
                )
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
                mock_gen_candidates.reset_mock()
                candidates, acq_vals = optimize_acqf(
                    acq_function=mock_acq_function,
                    bounds=bounds,
                    q=q,
                    num_restarts=num_restarts,
                    raw_samples=raw_samples,
                    options=options,
                    gen_candidates=mock_gen_candidates,
                )
                mock_gen_candidates.assert_called_once()
                self.assertTrue(torch.equal(candidates, mock_candidates[0]))
                self.assertTrue(torch.equal(acq_vals, mock_acq_values[0]))
                cnt += 1
                self.assertEqual(mock_gen_batch_initial_conditions.call_count, cnt)

                # test generation with provided initial conditions
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
                    gen_candidates=mock_gen_candidates,
                )
                self.assertTrue(torch.equal(candidates, mock_candidates))
                self.assertTrue(torch.equal(acq_vals, mock_acq_values))
                self.assertEqual(mock_gen_batch_initial_conditions.call_count, cnt)

                # test fixed features
                fixed_features = {0: 0.1}
                mock_candidates[:, 0] = 0.1
                mock_gen_candidates.return_value = (mock_candidates, mock_acq_values)
                candidates, acq_vals = optimize_acqf(
                    acq_function=mock_acq_function,
                    bounds=bounds,
                    q=q,
                    num_restarts=num_restarts,
                    raw_samples=raw_samples,
                    options=options,
                    fixed_features=fixed_features,
                    gen_candidates=mock_gen_candidates,
                )
                self.assertEqual(
                    mock_gen_candidates.call_args[1]["fixed_features"], fixed_features
                )
                self.assertTrue(torch.equal(candidates, mock_candidates[0]))
                cnt += 1
                self.assertEqual(mock_gen_batch_initial_conditions.call_count, cnt)

                # test trivial case when all features are fixed
                candidates, acq_vals = optimize_acqf(
                    acq_function=mock_acq_function,
                    bounds=bounds,
                    q=q,
                    num_restarts=num_restarts,
                    raw_samples=raw_samples,
                    options=options,
                    fixed_features={0: 0.1, 1: 0.2, 2: 0.3},
                    gen_candidates=mock_gen_candidates,
                )
                self.assertTrue(
                    torch.equal(
                        candidates,
                        torch.tensor(
                            [0.1, 0.2, 0.3], device=self.device, dtype=dtype
                        ).expand(3, 3),
                    )
                )
                self.assertEqual(mock_gen_batch_initial_conditions.call_count, cnt)

        # test OneShotAcquisitionFunction
        mock_acq_function = MockOneShotAcquisitionFunction()
        candidates, acq_vals = optimize_acqf(
            acq_function=mock_acq_function,
            bounds=bounds,
            q=q,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            options=options,
            gen_candidates=mock_gen_candidates,
        )
        self.assertTrue(
            torch.equal(
                candidates, mock_acq_function.extract_candidates(mock_candidates[0])
            )
        )
        self.assertTrue(torch.equal(acq_vals, mock_acq_values[0]))

        # verify ValueError
        with self.assertRaisesRegex(ValueError, "Must specify"):
            optimize_acqf(
                acq_function=MockAcquisitionFunction(),
                bounds=bounds,
                q=q,
                num_restarts=num_restarts,
                options=options,
                gen_candidates=mock_gen_candidates,
            )

    @mock.patch("botorch.optim.optimize.gen_batch_initial_conditions")
    @mock.patch("botorch.optim.optimize.gen_candidates_scipy")
    @mock.patch("botorch.generation.gen.gen_candidates_torch")
    @mock.patch("botorch.optim.utils.common.signature")
    def test_optimize_acqf_sequential(
        self,
        mock_signature,
        mock_gen_candidates_torch,
        mock_gen_candidates_scipy,
        mock_gen_batch_initial_conditions,
        timeout_sec=None,
    ):
        for mock_gen_candidates in (
            mock_gen_candidates_scipy,
            mock_gen_candidates_torch,
        ):
            if mock_gen_candidates == mock_gen_candidates_torch:
                mock_signature.return_value = signature(gen_candidates_torch)
            else:
                mock_signature.return_value = signature(gen_candidates_scipy)
            mock_gen_candidates.__name__ = "gen_candidates"
            q = 3
            num_restarts = 2
            raw_samples = 10
            options = {}
            for dtype in (torch.float, torch.double):
                mock_acq_function = MockAcquisitionFunction()
                mock_gen_batch_initial_conditions.side_effect = [
                    torch.zeros(num_restarts, 1, 3, device=self.device, dtype=dtype)
                    for _ in range(q)
                ]
                gcs_return_vals = [
                    (
                        torch.tensor(
                            [[[1.1, 2.1, 3.1]]], device=self.device, dtype=dtype
                        ),
                        torch.tensor([i], device=self.device, dtype=dtype),
                    )
                    for i in range(q)
                ]
                mock_gen_candidates.side_effect = gcs_return_vals
                expected_candidates = torch.cat(
                    [cands[0] for cands, _ in gcs_return_vals], dim=-2
                ).round()
                bounds = torch.stack(
                    [
                        torch.zeros(3, device=self.device, dtype=dtype),
                        4 * torch.ones(3, device=self.device, dtype=dtype),
                    ]
                )
                inequality_constraints = [
                    (torch.tensor([2]), torch.tensor([4]), torch.tensor(5))
                ]
                mock_gen_candidates.reset_mock()
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
                    timeout_sec=timeout_sec,
                    gen_candidates=mock_gen_candidates,
                )
                self.assertEqual(mock_gen_candidates.call_count, q)
                self.assertTrue(torch.equal(candidates, expected_candidates))
                self.assertTrue(
                    torch.equal(
                        acq_value, torch.cat([acqval for _, acqval in gcs_return_vals])
                    )
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
            # Verify error for passing in incorrect bounds
            with self.assertRaisesRegex(
                ValueError,
                "bounds should be a `2 x d` tensor",
            ):
                optimize_acqf(
                    acq_function=mock_acq_function,
                    bounds=bounds.T,
                    q=q,
                    num_restarts=num_restarts,
                    raw_samples=raw_samples,
                    sequential=True,
                )

            # Verify error when using sequential=True in
            # conjunction with user-supplied batch_initial_conditions
            with self.assertRaisesRegex(
                UnsupportedError,
                "`batch_initial_conditions` is not supported for sequential "
                "optimization. Either avoid specifying `batch_initial_conditions` "
                "to use the custom initializer or use the `ic_generator` kwarg to "
                "generate initial conditions for the case of "
                "nonlinear inequality constraints.",
            ):
                optimize_acqf(
                    acq_function=mock_acq_function,
                    bounds=bounds,
                    q=q,
                    num_restarts=num_restarts,
                    raw_samples=raw_samples,
                    batch_initial_conditions=mock_gen_batch_initial_conditions,
                    sequential=True,
                )

    def test_optimize_acqf_sequential_timeout(self):
        self.test_optimize_acqf_sequential(timeout_sec=1e-4)

    def test_optimize_acqf_sequential_notimplemented(self):
        # Sequential acquisition function optimization only supported
        # when return_best_only=True
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

    def test_optimize_acqf_sequential_q_constraint_notimplemented(self):
        # Sequential acquisition function not supported with q-constraints
        with self.assertRaises(UnsupportedError):
            optimize_acqf(
                acq_function=MockAcquisitionFunction(),
                bounds=torch.stack([torch.zeros(3), 4 * torch.ones(3)]),
                equality_constraints=[
                    (
                        torch.tensor(
                            [[0, 0], [1, 0]], device=self.device, dtype=torch.int64
                        ),
                        torch.tensor(
                            [1.0, -1.0], device=self.device, dtype=torch.float64
                        ),
                        0,
                    ),
                ],
                q=3,
                num_restarts=2,
                raw_samples=10,
                return_best_only=True,
                sequential=True,
            )
        with self.assertRaises(UnsupportedError):
            optimize_acqf(
                acq_function=MockAcquisitionFunction(),
                bounds=torch.stack([torch.zeros(3), 4 * torch.ones(3)]),
                inequality_constraints=[
                    (
                        torch.tensor(
                            [[0, 0], [1, 0]], device=self.device, dtype=torch.int64
                        ),
                        torch.tensor(
                            [1.0, -1.0], device=self.device, dtype=torch.float64
                        ),
                        0,
                    ),
                ],
                q=3,
                num_restarts=2,
                raw_samples=10,
                return_best_only=True,
                sequential=True,
            )

    def test_optimize_acqf_batch_limit(self) -> None:
        num_restarts = 3
        raw_samples = 5
        dim = 4
        q = 4
        batch_limit = 2

        options = {"batch_limit": batch_limit}
        initial_conditions = [
            torch.ones(shape) for shape in [(1, 2, dim), (2, 1, dim), (1, dim)]
        ] + [None]

        for gen_candidates, ics in zip(
            [gen_candidates_scipy, gen_candidates_torch], initial_conditions
        ):
            with self.subTest(gen_candidates=gen_candidates, initial_conditions=ics):
                _, acq_value_list = optimize_acqf(
                    acq_function=SinOneOverXAcqusitionFunction(),
                    bounds=torch.stack([-1 * torch.ones(dim), torch.ones(dim)]),
                    q=q,
                    num_restarts=num_restarts,
                    raw_samples=raw_samples,
                    options=options,
                    return_best_only=False,
                    gen_candidates=gen_candidates,
                    batch_initial_conditions=ics,
                )
                expected_shape = (num_restarts,) if ics is None else (ics.shape[0],)
                self.assertEqual(acq_value_list.shape, expected_shape)

    def test_optimize_acqf_runs_given_batch_initial_conditions(self):
        num_restarts, raw_samples, dim = 1, 2, 3

        opt_x = 2 / np.pi
        # -x[i] * 1 >= -opt_x * 1.01 => x[i] <= opt_x * 1.01
        inequality_constraints = [
            (torch.tensor([i]), -torch.tensor([1]), -opt_x * 1.01) for i in range(dim)
        ] + [
            # x[i] * 1 >= opt_x * .99
            (torch.tensor([i]), torch.tensor([1]), opt_x * 0.99)
            for i in range(dim)
        ]
        q = 1

        ic_shapes = [(1, 2, dim), (2, 1, dim), (1, dim)]

        torch.manual_seed(0)
        for shape in ic_shapes:
            with self.subTest(shape=shape):
                # start near one (of many) optima
                initial_conditions = (opt_x * 1.01) * torch.ones(shape)
                batch_candidates, acq_value_list = optimize_acqf(
                    acq_function=SinOneOverXAcqusitionFunction(),
                    bounds=torch.stack([-1 * torch.ones(dim), torch.ones(dim)]),
                    q=q,
                    num_restarts=num_restarts,
                    raw_samples=raw_samples,
                    batch_initial_conditions=initial_conditions,
                    inequality_constraints=inequality_constraints,
                )
                self.assertAllClose(
                    batch_candidates,
                    opt_x * torch.ones_like(batch_candidates),
                    # must be at least 50% closer to the optimum than it started
                    atol=0.004,
                    rtol=0.005,
                )
                self.assertAlmostEqual(acq_value_list.item(), 1, places=3)

    def test_optimize_acqf_wrong_ic_shape_inequality_constraints(self) -> None:
        dim = 3
        ic_shapes = [(1, 2, dim + 1), (1, 2, dim, 1), (1, dim + 1), (1, 1), (dim,)]

        for shape in ic_shapes:
            with self.subTest(shape=shape):
                initial_conditions = torch.ones(shape)
                expected_error = (
                    rf"batch_initial_conditions.shape\[-1\] must be {dim}\."
                    if len(shape) in (2, 3)
                    else r"batch_initial_conditions must be 2\-dimensional or "
                )
                with self.assertRaisesRegex(ValueError, expected_error):
                    optimize_acqf(
                        acq_function=MockAcquisitionFunction(),
                        bounds=torch.stack([-1 * torch.ones(dim), torch.ones(dim)]),
                        q=4,
                        batch_initial_conditions=initial_conditions,
                        num_restarts=1,
                    )

    def test_optimize_acqf_warns_on_opt_failure(self):
        """
        Test error handling in `scipy.optimize.minimize`.

        Expected behavior is that a warning is raised when optimization fails
        in `scipy.optimize.minimize`, and then it restarts and tries again.

        This is a test case cooked up to fail. It is trying to optimize
        sin(1/x), which is pathological near zero, given a starting point near
        zero.
        """
        num_restarts, raw_samples, dim = 1, 1, 1

        initial_conditions = 1e-8 * torch.ones((num_restarts, raw_samples, dim))
        torch.manual_seed(0)
        with warnings.catch_warnings(record=True) as ws:
            batch_candidates, acq_value_list = optimize_acqf(
                acq_function=SinOneOverXAcqusitionFunction(),
                bounds=torch.stack([-1 * torch.ones(dim), torch.ones(dim)]),
                q=1,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                batch_initial_conditions=initial_conditions,
            )

        message = (
            "Optimization failed in `gen_candidates_scipy` with the following "
            "warning(s):\n[OptimizationWarning('Optimization failed within "
            "`scipy.optimize.minimize` with status 2 and message ABNORMAL_TERMINATION"
            "_IN_LNSRCH.')]\nBecause you specified `batch_initial_conditions`, "
            "optimization will not be retried with new initial conditions and will "
            "proceed with the current solution. Suggested remediation: Try again with "
            "different `batch_initial_conditions`, or don't provide "
            "`batch_initial_conditions.`"
        )
        expected_warning_raised = any(
            (
                issubclass(w.category, RuntimeWarning) and message in str(w.message)
                for w in ws
            )
        )
        self.assertTrue(expected_warning_raised)

    def test_optimize_acqf_successfully_restarts_on_opt_failure(self):
        """
        Test that `optimize_acqf` can succeed after restarting on opt failure.

        With the given seed (5), `optimize_acqf` will choose an initial
        condition that causes failure in the first run of
        `gen_candidates_scipy`, then re-tries with a new starting point and
        succeed.

        Also tests that this can be turned off by setting
        `retry_on_optimization_warning = False`.
        """
        num_restarts, raw_samples, dim = 1, 1, 1

        bounds = torch.stack(
            [
                -1 * torch.ones(dim, dtype=torch.double),
                torch.ones(dim, dtype=torch.double),
            ]
        )
        torch.manual_seed(5)

        with warnings.catch_warnings(record=True) as ws:
            batch_candidates, acq_value_list = optimize_acqf(
                acq_function=SinOneOverXAcqusitionFunction(),
                bounds=bounds,
                q=1,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                # shorten the line search to make it faster and make failure
                # more likely
                options={"maxls": 2},
            )
        message = (
            "Optimization failed in `gen_candidates_scipy` with the following "
            "warning(s):\n[OptimizationWarning('Optimization failed within "
            "`scipy.optimize.minimize` with status 2 and message ABNORMAL_TERMINATION"
            "_IN_LNSRCH.')]\nTrying again with a new set of initial conditions."
        )
        expected_warning_raised = any(
            (
                issubclass(w.category, RuntimeWarning) and message in str(w.message)
                for w in ws
            )
        )
        self.assertTrue(expected_warning_raised)
        # check if it succeeded on restart -- the maximum value of sin(1/x) is 1
        self.assertAlmostEqual(acq_value_list.item(), 1.0)

        # Test with retry_on_optimization_warning = False.
        torch.manual_seed(5)
        with warnings.catch_warnings(record=True) as ws:
            batch_candidates, acq_value_list = optimize_acqf(
                acq_function=SinOneOverXAcqusitionFunction(),
                bounds=bounds,
                q=1,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                # shorten the line search to make it faster and make failure
                # more likely
                options={"maxls": 2},
                retry_on_optimization_warning=False,
            )
        expected_warning_raised = any(
            (
                issubclass(w.category, RuntimeWarning) and message in str(w.message)
                for w in ws
            )
        )
        self.assertFalse(expected_warning_raised)

    def test_optimize_acqf_warns_on_second_opt_failure(self):
        """
        Test that `optimize_acqf` warns if it fails on a second optimization try.

        With the given seed (230), `optimize_acqf` will choose an initial
        condition that causes failure in the first run of
        `gen_candidates_scipy`, then re-tries and still does not succeed. Since
        this doesn't happen with seeds 0 - 229, this test might be broken by
        future refactorings affecting calls to `torch`.
        """
        num_restarts, raw_samples, dim = 1, 1, 1

        bounds = torch.stack(
            [
                -1 * torch.ones(dim, dtype=torch.double),
                torch.ones(dim, dtype=torch.double),
            ]
        )

        with warnings.catch_warnings(record=True) as ws:
            torch.manual_seed(230)
            batch_candidates, acq_value_list = optimize_acqf(
                acq_function=SinOneOverXAcqusitionFunction(),
                bounds=bounds,
                q=1,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                # shorten the line search to make it faster and make failure
                # more likely
                options={"maxls": 2},
            )

        message_1 = (
            "Optimization failed in `gen_candidates_scipy` with the following "
            "warning(s):\n[OptimizationWarning('Optimization failed within "
            "`scipy.optimize.minimize` with status 2 and message ABNORMAL_TERMINATION"
            "_IN_LNSRCH.')]\nTrying again with a new set of initial conditions."
        )

        message_2 = (
            "Optimization failed on the second try, after generating a new set "
            "of initial conditions."
        )
        first_expected_warning_raised = any(
            (
                issubclass(w.category, RuntimeWarning) and message_1 in str(w.message)
                for w in ws
            )
        )
        second_expected_warning_raised = any(
            (
                issubclass(w.category, RuntimeWarning) and message_2 in str(w.message)
                for w in ws
            )
        )
        self.assertTrue(first_expected_warning_raised)
        self.assertTrue(second_expected_warning_raised)

    def test_optimize_acqf_nonlinear_constraints(self):
        num_restarts = 2
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            mock_acq_function = SquaredAcquisitionFunction()
            bounds = torch.stack(
                [torch.zeros(3, **tkwargs), 4 * torch.ones(3, **tkwargs)]
            )

            # Make sure we find the global optimum [4, 4, 4] without constraints
            with torch.random.fork_rng():
                torch.manual_seed(0)
                candidates, acq_value = optimize_acqf(
                    acq_function=mock_acq_function,
                    bounds=bounds,
                    q=1,
                    num_restarts=num_restarts,
                    sequential=True,
                    raw_samples=16,
                )
            self.assertAllClose(candidates, 4 * torch.ones(1, 3, **tkwargs))

            # Constrain the sum to be <= 4 in which case the solution is a
            # permutation of [4, 0, 0]
            def nlc1(x):
                return 4 - x.sum(dim=-1)

            batch_initial_conditions = torch.tensor([[[0.5, 0.5, 3]]], **tkwargs)
            candidates, acq_value = optimize_acqf(
                acq_function=mock_acq_function,
                bounds=bounds,
                q=1,
                nonlinear_inequality_constraints=[nlc1],
                batch_initial_conditions=batch_initial_conditions,
                num_restarts=1,
            )
            self.assertTrue(
                torch.allclose(
                    torch.sort(candidates).values,
                    torch.tensor([[0, 0, 4]], **tkwargs),
                )
            )
            self.assertTrue(
                torch.allclose(acq_value, torch.tensor([4], **tkwargs), atol=1e-3)
            )

            # Make sure we return the initial solution if SLSQP fails to return
            # a feasible point.
            with mock.patch(
                "botorch.generation.gen.minimize_with_timeout"
            ) as mock_minimize:
                # By setting "success" to True and "status" to 0, we prevent a
                # warning that `minimize` failed, which isn't the behavior
                # we're looking to test here.
                mock_minimize.return_value = OptimizeResult(
                    x=np.array([4, 4, 4]), success=True, status=0
                )
                candidates, acq_value = optimize_acqf(
                    acq_function=mock_acq_function,
                    bounds=bounds,
                    q=1,
                    nonlinear_inequality_constraints=[nlc1],
                    batch_initial_conditions=batch_initial_conditions,
                    num_restarts=1,
                )
                self.assertAllClose(candidates, batch_initial_conditions[0, ...])

            # Constrain all variables to be >= 1. The global optimum is 2.45 and
            # is attained by some permutation of [1, 1, 2]
            def nlc2(x):
                return x[..., 0] - 1

            def nlc3(x):
                return x[..., 1] - 1

            def nlc4(x):
                return x[..., 2] - 1

            with torch.random.fork_rng():
                torch.manual_seed(0)
                batch_initial_conditions = 1 + 0.33 * torch.rand(
                    num_restarts, 1, 3, **tkwargs
                )
            candidates, acq_value = optimize_acqf(
                acq_function=mock_acq_function,
                bounds=bounds,
                q=1,
                nonlinear_inequality_constraints=[nlc1, nlc2, nlc3, nlc4],
                batch_initial_conditions=batch_initial_conditions,
                num_restarts=num_restarts,
            )
            self.assertTrue(
                torch.allclose(
                    torch.sort(candidates).values,
                    torch.tensor([[1, 1, 2]], **tkwargs),
                )
            )
            self.assertTrue(
                torch.allclose(acq_value, torch.tensor(2.45, **tkwargs), atol=1e-3)
            )

            # Test that an ic_generator object with the same API as
            # gen_batch_initial_conditions returns candidates of the
            # required shape.
            with mock.patch(
                "botorch.optim.optimize.gen_batch_initial_conditions"
            ) as ic_generator:
                ic_generator.return_value = batch_initial_conditions
                candidates, acq_value = optimize_acqf(
                    acq_function=mock_acq_function,
                    bounds=bounds,
                    q=3,
                    nonlinear_inequality_constraints=[nlc1],
                    num_restarts=1,
                    ic_generator=ic_generator,
                )
                self.assertEqual(candidates.size(), torch.Size([1, 3]))

            # Make sure fixed features aren't supported
            with self.assertRaisesRegex(
                NotImplementedError,
                "Fixed features are not supported when non-linear inequality "
                "constraints are given.",
            ):
                optimize_acqf(
                    acq_function=mock_acq_function,
                    bounds=bounds,
                    q=1,
                    nonlinear_inequality_constraints=[nlc1, nlc2, nlc3, nlc4],
                    batch_initial_conditions=batch_initial_conditions,
                    num_restarts=num_restarts,
                    fixed_features={0: 0.1},
                )

            # Constraints must be passed in as lists
            with self.assertRaisesRegex(
                ValueError,
                "`nonlinear_inequality_constraints` must be a list of callables, "
                "got <class 'function'>.",
            ):
                optimize_acqf(
                    acq_function=mock_acq_function,
                    bounds=bounds,
                    q=1,
                    nonlinear_inequality_constraints=nlc1,
                    num_restarts=num_restarts,
                    batch_initial_conditions=batch_initial_conditions,
                )

            # batch_initial_conditions must be feasible
            with self.assertRaisesRegex(
                ValueError,
                "`batch_initial_conditions` must satisfy the non-linear "
                "inequality constraints.",
            ):
                optimize_acqf(
                    acq_function=mock_acq_function,
                    bounds=bounds,
                    q=1,
                    nonlinear_inequality_constraints=[nlc1],
                    num_restarts=num_restarts,
                    batch_initial_conditions=4 * torch.ones(1, 1, 3, **tkwargs),
                )
            # Explicitly setting batch_limit to be >1 should raise
            with self.assertRaisesRegex(
                ValueError,
                "`batch_limit` must be 1 when non-linear inequality constraints "
                "are given.",
            ):
                optimize_acqf(
                    acq_function=mock_acq_function,
                    bounds=bounds,
                    q=1,
                    nonlinear_inequality_constraints=[nlc1],
                    batch_initial_conditions=torch.rand(5, 1, 3, **tkwargs),
                    num_restarts=5,
                    options={"batch_limit": 5},
                )
            # If there are non-linear inequality constraints an initial condition
            # generator object `ic_generator` must be supplied.
            with self.assertRaisesRegex(
                RuntimeError,
                "`ic_generator` must be given if "
                "there are non-linear inequality constraints.",
            ):
                optimize_acqf(
                    acq_function=mock_acq_function,
                    bounds=bounds,
                    q=1,
                    nonlinear_inequality_constraints=[nlc1],
                    num_restarts=1,
                    raw_samples=16,
                )

    @mock.patch("botorch.generation.gen.gen_candidates_torch")
    @mock.patch("botorch.optim.optimize.gen_batch_initial_conditions")
    @mock.patch("botorch.optim.optimize.gen_candidates_scipy")
    @mock.patch("botorch.optim.utils.common.signature")
    def test_optimize_acqf_non_linear_constraints_sequential(
        self,
        mock_signature,
        mock_gen_candidates_scipy,
        mock_gen_batch_initial_conditions,
        mock_gen_candidates_torch,
    ):
        def nlc(x):
            return 4 * x[..., 2] - 5

        q = 3
        num_restarts = 2
        raw_samples = 10
        options = {}
        for mock_gen_candidates in (
            mock_gen_candidates_torch,
            mock_gen_candidates_scipy,
        ):
            if mock_gen_candidates == mock_gen_candidates_torch:
                mock_signature.return_value = signature(gen_candidates_torch)
            else:
                mock_signature.return_value = signature(gen_candidates_scipy)
            for dtype in (torch.float, torch.double):

                mock_acq_function = MockAcquisitionFunction()
                mock_gen_batch_initial_conditions.side_effect = [
                    torch.zeros(num_restarts, 1, 3, device=self.device, dtype=dtype)
                    for _ in range(q)
                ]
                gcs_return_vals = [
                    (
                        torch.tensor(
                            [[[1.0, 2.0, 3.0]]], device=self.device, dtype=dtype
                        ),
                        torch.tensor([i], device=self.device, dtype=dtype),
                    )
                    # for nonlinear inequality constraints the batch_limit variable is
                    # currently set to 1 by default and hence gen_candidates_scipy is
                    # called num_restarts*q times
                    for i in range(num_restarts * q)
                ]
                mock_gen_candidates.side_effect = gcs_return_vals
                expected_candidates = torch.cat(
                    [cands[0] for cands, _ in gcs_return_vals[::num_restarts]], dim=-2
                )
                bounds = torch.stack(
                    [
                        torch.zeros(3, device=self.device, dtype=dtype),
                        4 * torch.ones(3, device=self.device, dtype=dtype),
                    ]
                )
                with warnings.catch_warnings(record=True) as ws:
                    candidates, acq_value = optimize_acqf(
                        acq_function=mock_acq_function,
                        bounds=bounds,
                        q=q,
                        num_restarts=num_restarts,
                        raw_samples=raw_samples,
                        options=options,
                        nonlinear_inequality_constraints=[nlc],
                        sequential=True,
                        ic_generator=mock_gen_batch_initial_conditions,
                        gen_candidates=mock_gen_candidates,
                    )
                if mock_gen_candidates == mock_gen_candidates_torch:
                    self.assertEqual(len(ws), 3)
                    message = (
                        "Keyword arguments ['nonlinear_inequality_constraints',"
                        " 'equality_constraints', 'inequality_constraints'] will"
                        " be ignored because they are not allowed parameters for"
                        " function gen_candidates. Allowed parameters are "
                        " ['initial_conditions', 'acquisition_function', "
                        "'lower_bounds', 'upper_bounds', 'optimizer', 'options',"
                        " 'callback', 'fixed_features', 'timeout_sec']."
                    )
                    expected_warning_raised = (
                        issubclass(w.category, UserWarning)
                        and message == str(w.message)
                        for w in ws
                    )
                    self.assertTrue(expected_warning_raised)
                    # check message
                else:
                    self.assertEqual(len(ws), 0)
                self.assertTrue(torch.equal(candidates, expected_candidates))
                # Extract the relevant entries from gcs_return_vals to
                # perform comparison with.
                self.assertTrue(
                    torch.equal(
                        acq_value,
                        torch.cat(
                            [
                                expected_acq_value
                                for _, expected_acq_value in gcs_return_vals[
                                    num_restarts - 1 :: num_restarts
                                ]
                            ]
                        ),
                    ),
                )

    def test_constraint_caching(self):
        def nlc(x):
            return 4 - x.sum(dim=-1)

        class FunWrapperWithCallCount:
            def __init__(self):
                self.call_count = 0

            def __call__(self, x, f):
                self.call_count += 1
                X = torch.from_numpy(x).view(-1).contiguous().requires_grad_(True)
                loss = f(X).sum()
                gradf = _arrayify(torch.autograd.grad(loss, X)[0].contiguous().view(-1))
                return loss.item(), gradf

        f_np_wrapper = FunWrapperWithCallCount()
        f_obj, f_grad = _make_f_and_grad_nonlinear_inequality_constraints(
            f_np_wrapper=f_np_wrapper, nlc=nlc
        )
        x1, x2 = np.array([1.0, 0.5, 0.25]), np.array([1.0, 0.5, 0.5])
        # Call f_obj once, this requires calling f_np_wrapper
        self.assertEqual(f_obj(x1), 2.25)
        self.assertEqual(f_np_wrapper.call_count, 1)
        # Call f_obj again, we should use the cached value this time
        self.assertEqual(f_obj(x1), 2.25)
        self.assertEqual(f_np_wrapper.call_count, 1)
        # Call f_grad, we should use the cached value here as well
        self.assertTrue(np.array_equal(f_grad(x1), -np.ones(3)))
        self.assertEqual(f_np_wrapper.call_count, 1)
        # Call f_grad with a new input
        self.assertTrue(np.array_equal(f_grad(x2), -np.ones(3)))
        self.assertEqual(f_np_wrapper.call_count, 2)
        # Call f_obj on the new input, should use the cache
        self.assertEqual(f_obj(x2), 2.0)
        self.assertEqual(f_np_wrapper.call_count, 2)


class TestOptimizeAcqfCyclic(BotorchTestCase):
    @mock.patch("botorch.optim.optimize._optimize_acqf")  # noqa: C901
    # TODO: make sure this runs without mock
    def test_optimize_acqf_cyclic(self, mock_optimize_acqf):
        num_restarts = 2
        raw_samples = 10
        num_cycles = 2
        options = {}
        tkwargs = {"device": self.device}
        bounds = torch.stack([torch.zeros(3), 4 * torch.ones(3)])
        inequality_constraints = [
            [torch.tensor([2], dtype=int), torch.tensor([4.0]), torch.tensor(5.0)]
        ]
        mock_acq_function = MockAcquisitionFunction()
        for q, dtype in itertools.product([1, 3], (torch.float, torch.double)):
            tkwargs["dtype"] = dtype
            inequality_constraints = [
                (
                    # indices can't be floats or doubles
                    inequality_constraints[0][0],
                    inequality_constraints[0][1].to(**tkwargs),
                    inequality_constraints[0][2].to(**tkwargs),
                )
            ]
            mock_optimize_acqf.reset_mock()
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


class TestOptimizeAcqfList(BotorchTestCase):
    @mock.patch("botorch.optim.optimize.optimize_acqf")  # noqa: C901
    @mock.patch("botorch.optim.optimize.optimize_acqf_mixed")
    def test_optimize_acqf_list(self, mock_optimize_acqf, mock_optimize_acqf_mixed):
        num_restarts = 2
        raw_samples = 10
        options = {}
        tkwargs = {"device": self.device}
        bounds = torch.stack([torch.zeros(3), 4 * torch.ones(3)])
        inequality_constraints = [
            [torch.tensor([3]), torch.tensor([4]), torch.tensor(5)]
        ]
        # reinitialize so that dtype
        mock_acq_function_1 = MockAcquisitionFunction()
        mock_acq_function_2 = MockAcquisitionFunction()
        mock_acq_function_list = [mock_acq_function_1, mock_acq_function_2]
        fixed_features_list = [None, [{0: 0.5}]]
        for ffl in fixed_features_list:
            for num_acqf, dtype in itertools.product(
                [1, 2], (torch.float, torch.double)
            ):
                for m in mock_acq_function_list:
                    # clear previous X_pending
                    m.set_X_pending(None)
                tkwargs["dtype"] = dtype
                inequality_constraints[0] = [
                    t.to(**tkwargs) for t in inequality_constraints[0]
                ]
                mock_optimize_acqf.reset_mock()
                mock_optimize_acqf_mixed.reset_mock()
                bounds = bounds.to(**tkwargs)
                candidate_rvs = []
                acq_val_rvs = []
                gcs_return_vals = [
                    (torch.rand(1, 3, **tkwargs), torch.rand(1, **tkwargs))
                    for _ in range(num_acqf)
                ]
                for rv in gcs_return_vals:
                    candidate_rvs.append(rv[0])
                    acq_val_rvs.append(rv[1])
                side_effect = list(zip(candidate_rvs, acq_val_rvs))
                mock_optimize_acqf.side_effect = side_effect
                mock_optimize_acqf_mixed.side_effect = side_effect
                orig_candidates = candidate_rvs[0].clone()
                # Wrap the set_X_pending method for checking that call arguments
                with mock.patch.object(
                    MockAcquisitionFunction,
                    "set_X_pending",
                    wraps=mock_acq_function_1.set_X_pending,
                ) as mock_set_X_pending_1, mock.patch.object(
                    MockAcquisitionFunction,
                    "set_X_pending",
                    wraps=mock_acq_function_2.set_X_pending,
                ) as mock_set_X_pending_2:
                    candidates, _ = optimize_acqf_list(
                        acq_function_list=mock_acq_function_list[:num_acqf],
                        bounds=bounds,
                        num_restarts=num_restarts,
                        raw_samples=raw_samples,
                        options=options,
                        inequality_constraints=inequality_constraints,
                        post_processing_func=rounding_func,
                        fixed_features_list=ffl,
                    )
                    # check that X_pending is set correctly in sequential optimization
                    if num_acqf > 1:
                        x_pending_call_args_list = mock_set_X_pending_2.call_args_list
                        idxr = torch.ones(
                            num_acqf, dtype=torch.bool, device=self.device
                        )
                        for i in range(len(x_pending_call_args_list) - 1):
                            idxr[i] = 0
                            self.assertTrue(
                                torch.equal(
                                    x_pending_call_args_list[i][0][0],
                                    orig_candidates[idxr],
                                )
                            )
                            idxr[i] = 1
                            orig_candidates[i] = candidate_rvs[i + 1]
                    else:
                        mock_set_X_pending_1.assert_not_called()
                # check final candidates
                expected_candidates = (
                    torch.cat(candidate_rvs[-num_acqf:], dim=0)
                    if num_acqf > 1
                    else candidate_rvs[0]
                )
                self.assertTrue(torch.equal(candidates, expected_candidates))
                # check call arguments for optimize_acqf
                if ffl is None:
                    call_args_list = mock_optimize_acqf.call_args_list
                    expected_call_args = {
                        "acq_function": None,
                        "bounds": bounds,
                        "q": 1,
                        "num_restarts": num_restarts,
                        "raw_samples": raw_samples,
                        "options": options,
                        "inequality_constraints": inequality_constraints,
                        "equality_constraints": None,
                        "fixed_features": None,
                        "post_processing_func": rounding_func,
                        "batch_initial_conditions": None,
                        "return_best_only": True,
                        "sequential": False,
                    }
                else:
                    call_args_list = mock_optimize_acqf_mixed.call_args_list
                    expected_call_args = {
                        "acq_function": None,
                        "bounds": bounds,
                        "q": 1,
                        "num_restarts": num_restarts,
                        "raw_samples": raw_samples,
                        "options": options,
                        "inequality_constraints": inequality_constraints,
                        "equality_constraints": None,
                        "post_processing_func": rounding_func,
                        "batch_initial_conditions": None,
                        "fixed_features_list": ffl,
                    }
                for i in range(len(call_args_list)):
                    expected_call_args["acq_function"] = mock_acq_function_list[i]
                    for k, v in call_args_list[i][1].items():
                        if torch.is_tensor(v):
                            self.assertTrue(torch.equal(expected_call_args[k], v))
                        elif k == "acq_function":
                            self.assertIsInstance(
                                mock_acq_function_list[i], MockAcquisitionFunction
                            )
                        else:
                            self.assertEqual(expected_call_args[k], v)

    def test_optimize_acqf_list_empty_list(self):
        with self.assertRaises(ValueError):
            optimize_acqf_list(
                acq_function_list=[],
                bounds=torch.stack([torch.zeros(3), 4 * torch.ones(3)]),
                num_restarts=2,
                raw_samples=10,
            )

    def test_optimize_acqf_list_fixed_features(self):
        with self.assertRaises(ValueError):
            optimize_acqf_list(
                acq_function_list=[
                    MockAcquisitionFunction(),
                    MockAcquisitionFunction(),
                ],
                bounds=torch.stack([torch.zeros(3), 4 * torch.ones(3)]),
                num_restarts=2,
                raw_samples=10,
                fixed_features_list=[{0: 0.5}],
                fixed_features={0: 0.5},
            )


class TestOptimizeAcqfMixed(BotorchTestCase):
    @mock.patch("botorch.optim.optimize.optimize_acqf")  # noqa: C901
    def test_optimize_acqf_mixed_q1(self, mock_optimize_acqf):
        num_restarts = 2
        raw_samples = 10
        q = 1
        options = {}
        tkwargs = {"device": self.device}
        bounds = torch.stack([torch.zeros(3), 4 * torch.ones(3)])
        mock_acq_function = MockAcquisitionFunction()
        for num_ff, dtype in itertools.product([1, 3], (torch.float, torch.double)):
            tkwargs["dtype"] = dtype
            mock_optimize_acqf.reset_mock()
            bounds = bounds.to(**tkwargs)

            candidate_rvs = []
            acq_val_rvs = []
            for _ in range(num_ff):
                candidate_rvs.append(torch.rand(1, 3, **tkwargs))
                acq_val_rvs.append(torch.rand(1, **tkwargs))
            fixed_features_list = [{i: i * 0.1} for i in range(num_ff)]
            side_effect = list(zip(candidate_rvs, acq_val_rvs))
            mock_optimize_acqf.side_effect = side_effect

            candidates, acq_value = optimize_acqf_mixed(
                acq_function=mock_acq_function,
                q=q,
                fixed_features_list=fixed_features_list,
                bounds=bounds,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                options=options,
                post_processing_func=rounding_func,
            )
            # compute expected output
            ff_acq_values = torch.stack(acq_val_rvs)
            best = torch.argmax(ff_acq_values)
            expected_candidates = candidate_rvs[best]
            expected_acq_value = ff_acq_values[best]
            self.assertTrue(torch.equal(candidates, expected_candidates))
            self.assertTrue(torch.equal(acq_value, expected_acq_value))
            # check call arguments for optimize_acqf
            call_args_list = mock_optimize_acqf.call_args_list
            expected_call_args = {
                "acq_function": None,
                "bounds": bounds,
                "q": q,
                "num_restarts": num_restarts,
                "raw_samples": raw_samples,
                "options": options,
                "inequality_constraints": None,
                "equality_constraints": None,
                "fixed_features": None,
                "post_processing_func": rounding_func,
                "batch_initial_conditions": None,
                "return_best_only": True,
                "sequential": False,
            }
            for i in range(len(call_args_list)):
                expected_call_args["fixed_features"] = fixed_features_list[i]
                for k, v in call_args_list[i][1].items():
                    if torch.is_tensor(v):
                        self.assertTrue(torch.equal(expected_call_args[k], v))
                    elif k == "acq_function":
                        self.assertIsInstance(v, MockAcquisitionFunction)
                    else:
                        self.assertEqual(expected_call_args[k], v)

    @mock.patch("botorch.optim.optimize.optimize_acqf")  # noqa: C901
    def test_optimize_acqf_mixed_q2(self, mock_optimize_acqf):
        num_restarts = 2
        raw_samples = 10
        q = 2
        options = {}
        tkwargs = {"device": self.device}
        bounds = torch.stack([torch.zeros(3), 4 * torch.ones(3)])
        mock_acq_functions = [
            MockAcquisitionFunction(),
            MockOneShotEvaluateAcquisitionFunction(),
        ]
        for num_ff, dtype, mock_acq_function in itertools.product(
            [1, 3], (torch.float, torch.double), mock_acq_functions
        ):
            tkwargs["dtype"] = dtype
            mock_optimize_acqf.reset_mock()
            bounds = bounds.to(**tkwargs)

            fixed_features_list = [{i: i * 0.1} for i in range(num_ff)]
            candidate_rvs, exp_candidates, acq_val_rvs = [], [], []
            # generate mock side effects and compute expected outputs
            for _ in range(q):
                candidate_rvs_q = [torch.rand(1, 3, **tkwargs) for _ in range(num_ff)]
                acq_val_rvs_q = [torch.rand(1, **tkwargs) for _ in range(num_ff)]
                best = torch.argmax(torch.stack(acq_val_rvs_q))
                exp_candidates.append(candidate_rvs_q[best])
                candidate_rvs += candidate_rvs_q
                acq_val_rvs += acq_val_rvs_q
            side_effect = list(zip(candidate_rvs, acq_val_rvs))
            mock_optimize_acqf.side_effect = side_effect

            candidates, acq_value = optimize_acqf_mixed(
                acq_function=mock_acq_function,
                q=q,
                fixed_features_list=fixed_features_list,
                bounds=bounds,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                options=options,
                post_processing_func=rounding_func,
            )

            expected_candidates = torch.cat(exp_candidates, dim=-2)
            if isinstance(mock_acq_function, MockOneShotEvaluateAcquisitionFunction):
                expected_acq_value = mock_acq_function.evaluate(
                    expected_candidates, bounds=bounds
                )
            else:
                expected_acq_value = mock_acq_function(expected_candidates)
            self.assertTrue(torch.equal(candidates, expected_candidates))
            self.assertTrue(torch.equal(acq_value, expected_acq_value))

    def test_optimize_acqf_mixed_empty_ff(self):
        with self.assertRaises(ValueError):
            mock_acq_function = MockAcquisitionFunction()
            optimize_acqf_mixed(
                acq_function=mock_acq_function,
                q=1,
                fixed_features_list=[],
                bounds=torch.stack([torch.zeros(3), 4 * torch.ones(3)]),
                num_restarts=2,
                raw_samples=10,
            )

    def test_optimize_acqf_one_shot_large_q(self):
        with self.assertRaises(ValueError):
            mock_acq_function = MockOneShotAcquisitionFunction()
            fixed_features_list = [{i: i * 0.1} for i in range(2)]
            optimize_acqf_mixed(
                acq_function=mock_acq_function,
                q=2,
                fixed_features_list=fixed_features_list,
                bounds=torch.stack([torch.zeros(3), 4 * torch.ones(3)]),
                num_restarts=2,
                raw_samples=10,
            )


class TestOptimizeAcqfDiscrete(BotorchTestCase):
    def test_optimize_acqf_discrete(self):

        for q, dtype in itertools.product((1, 2), (torch.float, torch.double)):
            tkwargs = {"device": self.device, "dtype": dtype}

            mock_acq_function = SquaredAcquisitionFunction()
            mock_acq_function.set_X_pending(None)
            # ensure proper raising of errors if no choices
            with self.assertRaisesRegex(InputDataError, "`choices` must be non-emtpy."):
                optimize_acqf_discrete(
                    acq_function=mock_acq_function,
                    q=q,
                    choices=torch.empty(0, 2),
                )

            choices = torch.rand(5, 2, **tkwargs)

            # warning for unsupported keyword arguments
            with self.assertWarnsRegex(
                DeprecationWarning,
                r"`optimize_acqf_discrete` does not support arguments "
                r"\['num_restarts'\]. In the future, this will become an error.",
            ):
                optimize_acqf_discrete(
                    acq_function=mock_acq_function, q=q, choices=choices, num_restarts=8
                )

            exp_acq_vals = mock_acq_function(choices)

            # test unique
            candidates, acq_value = optimize_acqf_discrete(
                acq_function=mock_acq_function,
                q=q,
                choices=choices,
            )
            best_idcs = torch.topk(exp_acq_vals, q).indices
            expected_candidates = choices[best_idcs]
            expected_acq_value = exp_acq_vals[best_idcs].reshape_as(acq_value)
            self.assertAllClose(acq_value, expected_acq_value)
            self.assertAllClose(candidates, expected_candidates)

            # test non-unique (test does not properly use pending points)
            candidates, acq_value = optimize_acqf_discrete(
                acq_function=mock_acq_function, q=q, choices=choices, unique=False
            )
            best_idx = torch.argmax(exp_acq_vals)
            expected_candidates = choices[best_idx].repeat(q, 1)
            expected_acq_value = exp_acq_vals[best_idx].repeat(q).reshape_as(acq_value)
            self.assertAllClose(acq_value, expected_acq_value)
            self.assertAllClose(candidates, expected_candidates)

            # test max_batch_limit
            candidates, acq_value = optimize_acqf_discrete(
                acq_function=mock_acq_function, q=q, choices=choices, max_batch_size=3
            )
            best_idcs = torch.topk(exp_acq_vals, q).indices
            expected_candidates = choices[best_idcs]
            expected_acq_value = exp_acq_vals[best_idcs].reshape_as(acq_value)
            self.assertAllClose(acq_value, expected_acq_value)
            self.assertAllClose(candidates, expected_candidates)

            # test max_batch_limit & unique
            candidates, acq_value = optimize_acqf_discrete(
                acq_function=mock_acq_function,
                q=q,
                choices=choices,
                unique=False,
                max_batch_size=3,
            )
            best_idx = torch.argmax(exp_acq_vals)
            expected_candidates = choices[best_idx].repeat(q, 1)
            expected_acq_value = exp_acq_vals[best_idx].repeat(q).reshape_as(acq_value)
            self.assertAllClose(acq_value, expected_acq_value)
            self.assertAllClose(candidates, expected_candidates)

        with self.assertRaises(UnsupportedError):
            acqf = MockOneShotAcquisitionFunction()
            optimize_acqf_discrete(
                acq_function=acqf,
                q=1,
                choices=torch.tensor([[0.5], [0.2]]),
            )

    def test_optimize_acqf_discrete_local_search(self):
        for q, dtype in itertools.product((1, 2), (torch.float, torch.double)):
            tkwargs = {"device": self.device, "dtype": dtype}

            mock_acq_function = SquaredAcquisitionFunction()
            mock_acq_function.set_X_pending(None)
            discrete_choices = [
                torch.tensor([0, 1, 6], **tkwargs),
                torch.tensor([2, 3, 4], **tkwargs),
                torch.tensor([5, 6, 9], **tkwargs),
            ]

            # make sure we can find the global optimum
            candidates, acq_value = optimize_acqf_discrete_local_search(
                acq_function=mock_acq_function,
                q=q,
                discrete_choices=discrete_choices,
                raw_samples=1,
                num_restarts=1,
            )
            self.assertTrue(
                torch.allclose(candidates[0], torch.tensor([6, 4, 9], **tkwargs))
            )
            if q > 1:  # there are three local minima
                self.assertTrue(
                    torch.allclose(candidates[1], torch.tensor([6, 3, 9], **tkwargs))
                    or torch.allclose(candidates[1], torch.tensor([1, 4, 9], **tkwargs))
                    or torch.allclose(candidates[1], torch.tensor([6, 4, 6], **tkwargs))
                )

            # same but with unique=False
            candidates, acq_value = optimize_acqf_discrete_local_search(
                acq_function=mock_acq_function,
                q=q,
                discrete_choices=discrete_choices,
                raw_samples=1,
                num_restarts=1,
                unique=False,
            )
            expected_candidates = torch.tensor([[6, 4, 9], [6, 4, 9]], **tkwargs)
            self.assertAllClose(candidates, expected_candidates[:q])

            # test X_avoid and batch_initial_conditions
            candidates, acq_value = optimize_acqf_discrete_local_search(
                acq_function=mock_acq_function,
                q=q,
                discrete_choices=discrete_choices,
                X_avoid=torch.tensor([[6, 4, 9]], **tkwargs),
                batch_initial_conditions=torch.tensor([[0, 2, 5]], **tkwargs).unsqueeze(
                    1
                ),
            )
            self.assertTrue(
                torch.allclose(candidates[0], torch.tensor([6, 3, 9], **tkwargs))
            )
            if q > 1:  # there are two local minima
                self.assertTrue(
                    torch.allclose(candidates[1], torch.tensor([6, 2, 9], **tkwargs))
                )

            # test inequality constraints
            inequality_constraints = [
                (
                    torch.tensor([2], device=self.device),
                    -1 * torch.ones(1, **tkwargs),
                    -6 * torch.ones(1, **tkwargs),
                )
            ]
            candidates, acq_value = optimize_acqf_discrete_local_search(
                acq_function=mock_acq_function,
                q=q,
                discrete_choices=discrete_choices,
                raw_samples=1,
                num_restarts=1,
                inequality_constraints=inequality_constraints,
            )
            self.assertTrue(
                torch.allclose(candidates[0], torch.tensor([6, 4, 6], **tkwargs))
            )
            if q > 1:  # there are three local minima
                self.assertTrue(
                    torch.allclose(candidates[1], torch.tensor([6, 4, 5], **tkwargs))
                    or torch.allclose(candidates[1], torch.tensor([6, 3, 6], **tkwargs))
                    or torch.allclose(candidates[1], torch.tensor([1, 4, 6], **tkwargs))
                )

            # make sure we break if there are no neighbors
            optimize_acqf_discrete_local_search(
                acq_function=mock_acq_function,
                q=q,
                discrete_choices=[
                    torch.tensor([0, 1], **tkwargs),
                    torch.tensor([1], **tkwargs),
                ],
                raw_samples=1,
                num_restarts=1,
            )

            # test _filter_infeasible
            X = torch.tensor([[0, 2, 5], [0, 2, 6], [0, 2, 9]], **tkwargs)
            X_filtered = _filter_infeasible(
                X=X, inequality_constraints=inequality_constraints
            )
            self.assertAllClose(X[:2], X_filtered)

            # test _filter_invalid
            X_filtered = _filter_invalid(X=X, X_avoid=X[1].unsqueeze(0))
            self.assertAllClose(X[[0, 2]], X_filtered)
            X_filtered = _filter_invalid(X=X, X_avoid=X[[0, 2]])
            self.assertAllClose(X[1].unsqueeze(0), X_filtered)

            # test _generate_neighbors
            X_loc = _generate_neighbors(
                x=torch.tensor([0, 2, 6], **tkwargs).unsqueeze(0),
                discrete_choices=discrete_choices,
                X_avoid=torch.tensor([[0, 3, 6], [0, 2, 5]], **tkwargs),
                inequality_constraints=inequality_constraints,
            )
            self.assertTrue(
                torch.allclose(
                    X_loc, torch.tensor([[1, 2, 6], [6, 2, 6], [0, 4, 6]], **tkwargs)
                )
            )

            # test _gen_batch_initial_conditions_local_search
            with self.assertRaisesRegex(RuntimeError, "Failed to generate"):
                _gen_batch_initial_conditions_local_search(
                    discrete_choices=discrete_choices,
                    raw_samples=1,
                    X_avoid=torch.zeros(0, 3, **tkwargs),
                    inequality_constraints=[],
                    min_points=30,
                )

            X = _gen_batch_initial_conditions_local_search(
                discrete_choices=discrete_choices,
                raw_samples=1,
                X_avoid=torch.zeros(0, 3, **tkwargs),
                inequality_constraints=[],
                min_points=20,
            )
            self.assertEqual(len(X), 20)
            self.assertAllClose(torch.unique(X, dim=0), X)
