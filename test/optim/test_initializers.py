#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from contextlib import ExitStack
from itertools import product
from random import random
from unittest import mock

import torch
from botorch import settings
from botorch.acquisition.analytic import PosteriorMean
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.acquisition.monte_carlo import (
    qExpectedImprovement,
    qNoisyExpectedImprovement,
)
from botorch.acquisition.multi_objective.monte_carlo import (
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.exceptions import BadInitialCandidatesWarning, SamplingWarning
from botorch.exceptions.errors import BotorchTensorDimensionError, UnsupportedError
from botorch.exceptions.warnings import BotorchWarning
from botorch.models import SingleTaskGP
from botorch.optim import initialize_q_batch, initialize_q_batch_nonneg
from botorch.optim.initializers import (
    gen_batch_initial_conditions,
    gen_one_shot_kg_initial_conditions,
    gen_value_function_initial_conditions,
    sample_perturbed_subset_dims,
    sample_points_around_best,
    sample_q_batches_from_polytope,
    sample_truncated_normal_perturbations,
    transform_constraints,
    transform_inter_point_constraint,
    transform_intra_point_constraint,
)
from botorch.sampling.normal import IIDNormalSampler
from botorch.utils.sampling import draw_sobol_samples
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
    def test_gen_batch_initial_inf_bounds(self):
        bounds = torch.rand(2, 2)
        bounds[0, 1] = float("inf")
        with self.assertRaisesRegex(
            NotImplementedError,
            r"Currently only finite values in `bounds` are supported for "
            r"generating initial conditions for optimization.",
        ):
            gen_batch_initial_conditions(
                acq_function=mock.Mock(),
                bounds=bounds,
                q=1,
                num_restarts=2,
                raw_samples=2,
            )

    def test_gen_batch_initial_conditions(self):
        bounds = torch.stack([torch.zeros(2), torch.ones(2)])
        mock_acqf = MockAcquisitionFunction()
        mock_acqf.objective = lambda y: y.squeeze(-1)
        for dtype in (torch.float, torch.double):
            bounds = bounds.to(device=self.device, dtype=dtype)
            mock_acqf.X_baseline = bounds  # for testing sample_around_best
            mock_acqf.model = MockModel(MockPosterior(mean=bounds[:, :1]))
            for nonnegative, seed, init_batch_limit, ffs, sample_around_best in product(
                [True, False], [None, 1234], [None, 1], [None, {0: 0.5}], [True, False]
            ):
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
                        fixed_features=ffs,
                        options={
                            "nonnegative": nonnegative,
                            "eta": 0.01,
                            "alpha": 0.1,
                            "seed": seed,
                            "init_batch_limit": init_batch_limit,
                            "sample_around_best": sample_around_best,
                        },
                    )
                    expected_shape = torch.Size([2, 1, 2])
                    self.assertEqual(batch_initial_conditions.shape, expected_shape)
                    self.assertEqual(batch_initial_conditions.device, bounds.device)
                    self.assertEqual(batch_initial_conditions.dtype, bounds.dtype)
                    batch_shape = (
                        torch.Size([])
                        if init_batch_limit is None
                        else torch.Size([init_batch_limit])
                    )
                    raw_samps = mock_acqf_call.call_args[0][0]
                    batch_shape = (
                        torch.Size([20 if sample_around_best else 10])
                        if init_batch_limit is None
                        else torch.Size([init_batch_limit])
                    )
                    expected_raw_samps_shape = batch_shape + torch.Size([1, 2])
                    self.assertEqual(raw_samps.shape, expected_raw_samps_shape)

                    if ffs is not None:
                        for idx, val in ffs.items():
                            self.assertTrue(
                                torch.all(batch_initial_conditions[..., idx] == val)
                            )

    def test_gen_batch_initial_conditions_highdim(self):
        d = 2200  # 2200 * 10 (q) > 21201 (sobol max dim)
        bounds = torch.stack([torch.zeros(d), torch.ones(d)])
        ffs_map = {i: random() for i in range(0, d, 2)}
        mock_acqf = MockAcquisitionFunction()
        mock_acqf.objective = lambda y: y.squeeze(-1)
        for dtype in (torch.float, torch.double):
            bounds = bounds.to(device=self.device, dtype=dtype)
            mock_acqf.X_baseline = bounds  # for testing sample_around_best
            mock_acqf.model = MockModel(MockPosterior(mean=bounds[:, :1]))

            for nonnegative, seed, ffs, sample_around_best in product(
                [True, False], [None, 1234], [None, ffs_map], [True, False]
            ):
                with warnings.catch_warnings(record=True) as ws, settings.debug(True):
                    batch_initial_conditions = gen_batch_initial_conditions(
                        acq_function=MockAcquisitionFunction(),
                        bounds=bounds,
                        q=10,
                        num_restarts=1,
                        raw_samples=2,
                        fixed_features=ffs,
                        options={
                            "nonnegative": nonnegative,
                            "eta": 0.01,
                            "alpha": 0.1,
                            "seed": seed,
                            "sample_around_best": sample_around_best,
                        },
                    )
                    self.assertTrue(
                        any(issubclass(w.category, SamplingWarning) for w in ws)
                    )
                expected_shape = torch.Size([1, 10, d])
                self.assertEqual(batch_initial_conditions.shape, expected_shape)
                self.assertEqual(batch_initial_conditions.device, bounds.device)
                self.assertEqual(batch_initial_conditions.dtype, bounds.dtype)
                if ffs is not None:
                    for idx, val in ffs.items():
                        self.assertTrue(
                            torch.all(batch_initial_conditions[..., idx] == val)
                        )

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

    def test_gen_batch_initial_conditions_transform_intra_point_constraint(self):
        for dtype in (torch.float, torch.double):
            constraint = (
                torch.tensor([0, 1], dtype=torch.int64, device=self.device),
                torch.tensor([-1, -1]).to(dtype=dtype, device=self.device),
                -1.0,
            )
            constraints = transform_intra_point_constraint(
                constraint=constraint, d=3, q=3
            )
            self.assertEqual(len(constraints), 3)
            self.assertAllClose(
                constraints[0][0],
                torch.tensor([0, 1], dtype=torch.int64, device=self.device),
            )
            self.assertAllClose(
                constraints[1][0],
                torch.tensor([3, 4], dtype=torch.int64, device=self.device),
            )
            self.assertAllClose(
                constraints[2][0],
                torch.tensor([6, 7], dtype=torch.int64, device=self.device),
            )
            for constraint in constraints:
                self.assertAllClose(
                    torch.tensor([-1, -1], dtype=dtype, device=self.device),
                    constraint[1],
                )
                self.assertEqual(constraint[2], -1.0)
            # test failure on invalid d
            constraint = (
                torch.tensor([[0, 3]], dtype=torch.int64, device=self.device),
                torch.tensor([-1.0, -1.0], dtype=dtype, device=self.device),
                0,
            )
            with self.assertRaisesRegex(
                ValueError,
                "Constraint indices cannot exceed the problem dimension d=3.",
            ):
                transform_intra_point_constraint(constraint=constraint, d=3, q=2)

    def test_gen_batch_intial_conditions_transform_inter_point_constraint(self):
        for dtype in (torch.float, torch.double):
            constraint = (
                torch.tensor([[0, 1], [1, 1]], dtype=torch.int64, device=self.device),
                torch.tensor([1.0, -1.0], dtype=dtype, device=self.device),
                0,
            )
            transformed = transform_inter_point_constraint(constraint=constraint, d=3)
            self.assertAllClose(
                transformed[0],
                torch.tensor([1, 4], dtype=torch.int64, device=self.device),
            )
            self.assertAllClose(
                transformed[1],
                torch.tensor([1.0, -1.0]).to(dtype=dtype, device=self.device),
            )
            self.assertEqual(constraint[2], 0.0)
            # test failure on invalid d
            constraint = (
                torch.tensor([[0, 1], [1, 3]], dtype=torch.int64, device=self.device),
                torch.tensor([1.0, -1.0], dtype=dtype, device=self.device),
                0,
            )
            with self.assertRaisesRegex(
                ValueError,
                "Constraint indices cannot exceed the problem dimension d=3.",
            ):
                transform_inter_point_constraint(constraint=constraint, d=3)

    def test_gen_batch_initial_conditions_transform_constraints(self):
        for dtype in (torch.float, torch.double):
            # test with None
            self.assertIsNone(transform_constraints(constraints=None, d=3, q=3))
            constraints = [
                (
                    torch.tensor([0, 1], dtype=torch.int64, device=self.device),
                    torch.tensor([-1.0, -1.0], dtype=dtype, device=self.device),
                    -1.0,
                ),
                (
                    torch.tensor(
                        [[0, 1], [1, 1]], device=self.device, dtype=torch.int64
                    ),
                    torch.tensor([1.0, -1.0], dtype=dtype, device=self.device),
                    0,
                ),
            ]
            transformed = transform_constraints(constraints=constraints, d=3, q=3)
            self.assertEqual(len(transformed), 4)
            self.assertAllClose(
                transformed[0][0],
                torch.tensor([0, 1], dtype=torch.int64, device=self.device),
            )
            self.assertAllClose(
                transformed[1][0],
                torch.tensor([3, 4], dtype=torch.int64, device=self.device),
            )
            self.assertAllClose(
                transformed[2][0],
                torch.tensor([6, 7], dtype=torch.int64, device=self.device),
            )
            for constraint in transformed[:3]:
                self.assertAllClose(
                    torch.tensor([-1, -1], dtype=dtype, device=self.device),
                    constraint[1],
                )
                self.assertEqual(constraint[2], -1.0)
            self.assertAllClose(
                transformed[-1][0],
                torch.tensor([1, 4], dtype=torch.int64, device=self.device),
            )
            self.assertAllClose(
                transformed[-1][1],
                torch.tensor([1.0, -1.0], dtype=dtype, device=self.device),
            )
            self.assertEqual(transformed[-1][2], 0.0)

    def test_gen_batch_initial_conditions_sample_q_batches_from_polytope(self):
        for dtype in (torch.float, torch.double):
            bounds = torch.tensor(
                [[0, 0, 0], [1, 1, 1]], device=self.device, dtype=dtype
            )
            inequality_constraints = [
                (
                    torch.tensor([0, 1], device=self.device, dtype=torch.int64),
                    torch.tensor([-1, 1], device=self.device, dtype=dtype),
                    torch.tensor(-0.5, device=self.device, dtype=dtype),
                )
            ]
            inter_point_inequality_constraints = [
                (
                    torch.tensor([0, 1], device=self.device, dtype=torch.int64),
                    torch.tensor([-1, 1], device=self.device, dtype=dtype),
                    torch.tensor(-0.5, device=self.device, dtype=dtype),
                ),
                (
                    torch.tensor(
                        [[0, 1], [1, 1]], device=self.device, dtype=torch.int64
                    ),
                    torch.tensor([1, 1], device=self.device, dtype=dtype),
                    torch.tensor(0.3, device=self.device, dtype=dtype),
                ),
            ]
            equality_constraints = [
                (
                    torch.tensor([0, 1, 2], device=self.device, dtype=torch.int64),
                    torch.tensor([1, 1, 1], device=self.device, dtype=dtype),
                    torch.tensor(1, device=self.device, dtype=dtype),
                )
            ]
            inter_point_equality_constraints = [
                (
                    torch.tensor([0, 1, 2], device=self.device, dtype=torch.int64),
                    torch.tensor([1, 1, 1], device=self.device, dtype=dtype),
                    torch.tensor(1, device=self.device, dtype=dtype),
                ),
                (
                    torch.tensor(
                        [[0, 0], [1, 0]], device=self.device, dtype=torch.int64
                    ),
                    torch.tensor([1.0, -1.0], device=self.device, dtype=dtype),
                    0,
                ),
            ]
            for equalities, inequalities in product(
                [None, equality_constraints, inter_point_equality_constraints],
                [None, inequality_constraints, inter_point_inequality_constraints],
            ):
                samples = sample_q_batches_from_polytope(
                    n=5,
                    q=3,
                    bounds=bounds,
                    n_burnin=10000,
                    thinning=32,
                    seed=42,
                    inequality_constraints=inequalities,
                    equality_constraints=equalities,
                )
                self.assertEqual(samples.shape, torch.Size((5, 3, 3)))

    def test_gen_batch_initial_conditions_constraints(self):
        for dtype in (torch.float, torch.double):
            bounds = torch.tensor([[0, 0], [1, 1]], device=self.device, dtype=dtype)
            inequality_constraints = [
                (
                    torch.tensor([1], device=self.device, dtype=torch.int64),
                    torch.tensor([-4], device=self.device, dtype=dtype),
                    torch.tensor(-3, device=self.device, dtype=dtype),
                )
            ]
            equality_constraints = [
                (
                    torch.tensor([0], device=self.device, dtype=torch.int64),
                    torch.tensor([1], device=self.device, dtype=dtype),
                    torch.tensor(0.5, device=self.device, dtype=dtype),
                )
            ]
            for nonnegative, seed, init_batch_limit, ffs in product(
                [True, False], [None, 1234], [None, 1], [None, {0: 0.5}]
            ):
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
                            "thinning": 2,
                            "n_burnin": 3,
                        },
                        inequality_constraints=inequality_constraints,
                        equality_constraints=equality_constraints,
                    )
                    expected_shape = torch.Size([2, 1, 2])
                    self.assertEqual(batch_initial_conditions.shape, expected_shape)
                    self.assertEqual(batch_initial_conditions.device, bounds.device)
                    self.assertEqual(batch_initial_conditions.dtype, bounds.dtype)
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
                    self.assertTrue((raw_samps[..., 0] == 0.5).all())
                    self.assertTrue((-4 * raw_samps[..., 1] >= -3).all())
                    if ffs is not None:
                        for idx, val in ffs.items():
                            self.assertTrue(
                                torch.all(batch_initial_conditions[..., idx] == val)
                            )

    def test_gen_batch_initial_conditions_interpoint_constraints(self):
        for dtype in (torch.float, torch.double):
            bounds = torch.tensor([[0, 0], [1, 1]], device=self.device, dtype=dtype)
            inequality_constraints = [
                (
                    torch.tensor([0, 1], device=self.device, dtype=torch.int64),
                    torch.tensor([-1, -1.0], device=self.device, dtype=dtype),
                    torch.tensor(-1.0, device=self.device, dtype=dtype),
                )
            ]
            equality_constraints = [
                (
                    torch.tensor(
                        [[0, 0], [1, 0]], device=self.device, dtype=torch.int64
                    ),
                    torch.tensor([1.0, -1.0], device=self.device, dtype=dtype),
                    0,
                ),
                (
                    torch.tensor(
                        [[0, 0], [2, 0]], device=self.device, dtype=torch.int64
                    ),
                    torch.tensor([1.0, -1.0], device=self.device, dtype=dtype),
                    0,
                ),
            ]

            for nonnegative, seed in product([True, False], [None, 1234]):
                mock_acqf = MockAcquisitionFunction()
                with mock.patch.object(
                    MockAcquisitionFunction,
                    "__call__",
                    wraps=mock_acqf.__call__,
                ):
                    batch_initial_conditions = gen_batch_initial_conditions(
                        acq_function=mock_acqf,
                        bounds=bounds,
                        q=3,
                        num_restarts=2,
                        raw_samples=10,
                        options={
                            "nonnegative": nonnegative,
                            "eta": 0.01,
                            "alpha": 0.1,
                            "seed": seed,
                            "init_batch_limit": None,
                            "thinning": 2,
                            "n_burnin": 3,
                        },
                        inequality_constraints=inequality_constraints,
                        equality_constraints=equality_constraints,
                    )
                    expected_shape = torch.Size([2, 3, 2])
                    self.assertEqual(batch_initial_conditions.shape, expected_shape)
                    self.assertEqual(batch_initial_conditions.device, bounds.device)
                    self.assertEqual(batch_initial_conditions.dtype, bounds.dtype)

                    self.assertTrue((batch_initial_conditions.sum(dim=-1) <= 1).all())

                    self.assertAllClose(
                        batch_initial_conditions[0, 0, 0],
                        batch_initial_conditions[0, 1, 0],
                        batch_initial_conditions[0, 2, 0],
                    )

                    self.assertAllClose(
                        batch_initial_conditions[1, 0, 0],
                        batch_initial_conditions[1, 1, 0],
                        batch_initial_conditions[1, 2, 0],
                    )

    def test_error_equality_constraints_with_sample_around_best(self):
        tkwargs = {"device": self.device, "dtype": torch.double}
        # this will give something that does not respect the constraints
        # TODO: it would be good to have a utils function to check if the
        # constraints are obeyed
        with self.assertRaises(UnsupportedError) as e:
            gen_batch_initial_conditions(
                MockAcquisitionFunction(),
                bounds=torch.tensor([[0, 0], [1, 1]], **tkwargs),
                q=1,
                num_restarts=1,
                raw_samples=1,
                equality_constraints=[
                    (
                        torch.tensor([0], **tkwargs),
                        torch.tensor([1], **tkwargs),
                        torch.tensor(0.5, **tkwargs),
                    )
                ],
                options={"sample_around_best": True},
            )
        self.assertTrue(
            "Option 'sample_around_best' is not supported when equality"
            "constraints are present." in str(e.exception)
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
        fantasy_model = model.fantasize(
            fant_X, IIDNormalSampler(sample_shape=torch.Size([num_fantasies]))
        )
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
        fantasy_model = model.fantasize(
            fant_X, IIDNormalSampler(sample_shape=torch.Size([num_fantasies]))
        )
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


class TestSampleAroundBest(BotorchTestCase):
    def test_sample_truncated_normal_perturbations(self):
        tkwargs = {"device": self.device}
        n_discrete_points = 5
        _bounds = torch.ones(2, 4)
        _bounds[1] = 2
        for dtype in (torch.float, torch.double):
            tkwargs["dtype"] = dtype
            bounds = _bounds.to(**tkwargs)
            for n_best in (1, 2):
                X = 1 + torch.rand(n_best, 4, **tkwargs)
                # basic test
                perturbed_X = sample_truncated_normal_perturbations(
                    X=X,
                    n_discrete_points=n_discrete_points,
                    sigma=4,
                    bounds=bounds,
                    qmc=False,
                )
                self.assertEqual(perturbed_X.shape, torch.Size([n_discrete_points, 4]))
                self.assertTrue((perturbed_X >= 1).all())
                self.assertTrue((perturbed_X <= 2).all())
                # test qmc
                with mock.patch(
                    "botorch.optim.initializers.draw_sobol_samples",
                    wraps=draw_sobol_samples,
                ) as mock_sobol:
                    perturbed_X = sample_truncated_normal_perturbations(
                        X=X,
                        n_discrete_points=n_discrete_points,
                        sigma=4,
                        bounds=bounds,
                        qmc=True,
                    )
                    mock_sobol.assert_called_once()
                self.assertEqual(perturbed_X.shape, torch.Size([n_discrete_points, 4]))
                self.assertTrue((perturbed_X >= 1).all())
                self.assertTrue((perturbed_X <= 2).all())

    def test_sample_perturbed_subset_dims(self):
        tkwargs = {"device": self.device}
        n_discrete_points = 5

        # test that errors are raised
        with self.assertRaises(BotorchTensorDimensionError):
            sample_perturbed_subset_dims(
                X=torch.zeros(1, 1),
                n_discrete_points=1,
                sigma=1e-3,
                bounds=torch.zeros(1, 2, 1),
            )
        with self.assertRaises(BotorchTensorDimensionError):
            sample_perturbed_subset_dims(
                X=torch.zeros(1, 1, 1),
                n_discrete_points=1,
                sigma=1e-3,
                bounds=torch.zeros(2, 1),
            )
        for dtype in (torch.float, torch.double):
            for n_best in (1, 2):
                tkwargs["dtype"] = dtype
                bounds = torch.zeros(2, 21, **tkwargs)
                bounds[1] = 1
                X = torch.rand(n_best, 21, **tkwargs)
                # basic test
                with mock.patch(
                    "botorch.optim.initializers.draw_sobol_samples",
                ) as mock_sobol:
                    perturbed_X = sample_perturbed_subset_dims(
                        X=X,
                        n_discrete_points=n_discrete_points,
                        qmc=False,
                        sigma=1e-3,
                        bounds=bounds,
                    )
                    mock_sobol.assert_not_called()
                self.assertEqual(perturbed_X.shape, torch.Size([n_discrete_points, 21]))
                self.assertTrue((perturbed_X >= 0).all())
                self.assertTrue((perturbed_X <= 1).all())
                # test qmc
                with mock.patch(
                    "botorch.optim.initializers.draw_sobol_samples",
                    wraps=draw_sobol_samples,
                ) as mock_sobol:
                    perturbed_X = sample_perturbed_subset_dims(
                        X=X,
                        n_discrete_points=n_discrete_points,
                        sigma=1e-3,
                        bounds=bounds,
                    )
                    mock_sobol.assert_called_once()
                self.assertEqual(perturbed_X.shape, torch.Size([n_discrete_points, 21]))
                self.assertTrue((perturbed_X >= 0).all())
                self.assertTrue((perturbed_X <= 1).all())
                # for each point in perturbed_X compute the number of
                # dimensions it has in common with each point in X
                # and take the maximum number
                max_equal_dims = (
                    (perturbed_X.unsqueeze(0) == X.unsqueeze(1))
                    .sum(dim=-1)
                    .max(dim=0)
                    .values
                )
                # check that at least one dimension is perturbed
                self.assertTrue((20 - max_equal_dims >= 1).all())

    def test_sample_points_around_best(self):
        tkwargs = {"device": self.device}
        _bounds = torch.ones(2, 2)
        _bounds[1] = 2
        for dtype in (torch.float, torch.double):
            tkwargs["dtype"] = dtype
            bounds = _bounds.to(**tkwargs)
            X_train = 1 + torch.rand(20, 2, **tkwargs)
            model = MockModel(
                MockPosterior(mean=(2 * X_train + 1).sum(dim=-1, keepdim=True))
            )
            # test NEI with X_baseline
            acqf = qNoisyExpectedImprovement(
                model, X_baseline=X_train, cache_root=False
            )
            with mock.patch(
                "botorch.optim.initializers.sample_perturbed_subset_dims"
            ) as mock_subset_dims:
                X_rnd = sample_points_around_best(
                    acq_function=acqf,
                    n_discrete_points=4,
                    sigma=1e-3,
                    bounds=bounds,
                )
                mock_subset_dims.assert_not_called()
            self.assertTrue(X_rnd.shape, torch.Size([4, 2]))
            self.assertTrue((X_rnd >= 1).all())
            self.assertTrue((X_rnd <= 2).all())
            # test model that returns a batched mean
            model = MockModel(
                MockPosterior(
                    mean=(2 * X_train + 1).sum(dim=-1, keepdim=True).unsqueeze(0)
                )
            )
            acqf = qNoisyExpectedImprovement(
                model, X_baseline=X_train, cache_root=False
            )
            X_rnd = sample_points_around_best(
                acq_function=acqf,
                n_discrete_points=4,
                sigma=1e-3,
                bounds=bounds,
            )
            self.assertTrue(X_rnd.shape, torch.Size([4, 2]))
            self.assertTrue((X_rnd >= 1).all())
            self.assertTrue((X_rnd <= 2).all())

            # test EI without X_baseline
            acqf = qExpectedImprovement(model, best_f=0.0)

            with warnings.catch_warnings(record=True) as w, settings.debug(True):

                X_rnd = sample_points_around_best(
                    acq_function=acqf,
                    n_discrete_points=4,
                    sigma=1e-3,
                    bounds=bounds,
                )
                self.assertEqual(len(w), 1)
                self.assertTrue(issubclass(w[-1].category, BotorchWarning))
                self.assertIsNone(X_rnd)

            # set train inputs
            model.train_inputs = (X_train,)
            X_rnd = sample_points_around_best(
                acq_function=acqf,
                n_discrete_points=4,
                sigma=1e-3,
                bounds=bounds,
            )
            self.assertTrue(X_rnd.shape, torch.Size([4, 2]))
            self.assertTrue((X_rnd >= 1).all())
            self.assertTrue((X_rnd <= 2).all())

            # test an acquisition function that has no posterior_transform
            # and maximize=False
            pm = PosteriorMean(model, maximize=False)
            self.assertIsNone(pm.posterior_transform)
            self.assertFalse(pm.maximize)
            X_rnd = sample_points_around_best(
                acq_function=pm,
                n_discrete_points=4,
                sigma=0,
                bounds=bounds,
                best_pct=1e-8,  # ensures that we only use best value
            )
            idx = (-model.posterior(X_train).mean).argmax()
            self.assertTrue((X_rnd == X_train[idx : idx + 1]).all(dim=-1).all())

            # test acquisition function that has no model
            ff = FixedFeatureAcquisitionFunction(pm, d=2, columns=[0], values=[0])
            # set X_baseline for testing purposes
            ff.X_baseline = X_train
            with warnings.catch_warnings(record=True) as w, settings.debug(True):
                X_rnd = sample_points_around_best(
                    acq_function=ff,
                    n_discrete_points=4,
                    sigma=1e-3,
                    bounds=bounds,
                )
                self.assertEqual(len(w), 1)
                self.assertTrue(issubclass(w[-1].category, BotorchWarning))
                self.assertIsNone(X_rnd)

            # test constraints with NEHVI
            constraints = [lambda Y: Y[..., 0]]
            ref_point = torch.zeros(2, **tkwargs)
            # test cases when there are and are not any feasible points
            for any_feas in (True, False):
                Y_train = torch.stack(
                    [
                        torch.linspace(-0.5, 0.5, X_train.shape[0], **tkwargs)
                        if any_feas
                        else torch.ones(X_train.shape[0], **tkwargs),
                        X_train.sum(dim=-1),
                    ],
                    dim=-1,
                )
                moo_model = MockModel(MockPosterior(mean=Y_train, samples=Y_train))
                acqf = qNoisyExpectedHypervolumeImprovement(
                    moo_model,
                    ref_point=ref_point,
                    X_baseline=X_train,
                    constraints=constraints,
                    cache_root=False,
                    sampler=IIDNormalSampler(sample_shape=torch.Size([2])),
                )
                X_rnd = sample_points_around_best(
                    acq_function=acqf,
                    n_discrete_points=4,
                    sigma=0.0,
                    bounds=bounds,
                )
                self.assertTrue(X_rnd.shape, torch.Size([4, 2]))
                # this should be true since sigma=0
                # and we should only be returning feasible points
                violation = constraints[0](Y_train)
                neg_violation = -violation.clamp_min(0.0)
                feas = neg_violation == 0
                eq_mask = (X_train.unsqueeze(1) == X_rnd.unsqueeze(0)).all(dim=-1)
                if feas.any():
                    # determine
                    # create n_train x n_rnd tensor of booleans
                    eq_mask = (X_train.unsqueeze(1) == X_rnd.unsqueeze(0)).all(dim=-1)
                    # check that all X_rnd correspond to feasible points
                    self.assertEqual(eq_mask[feas].sum(), 4)
                else:
                    idcs = torch.topk(neg_violation, k=2).indices
                    self.assertEqual(eq_mask[idcs].sum(), 4)
                self.assertTrue((X_rnd >= 1).all())
                self.assertTrue((X_rnd <= 2).all())
            # test that subset_dims is called if d>=20
            X_train = 1 + torch.rand(10, 20, **tkwargs)
            model = MockModel(
                MockPosterior(mean=(2 * X_train + 1).sum(dim=-1, keepdim=True))
            )
            bounds = torch.ones(2, 20, **tkwargs)
            bounds[1] = 2
            # test NEI with X_baseline
            acqf = qNoisyExpectedImprovement(
                model, X_baseline=X_train, cache_root=False
            )
            with mock.patch(
                "botorch.optim.initializers.sample_perturbed_subset_dims",
                wraps=sample_perturbed_subset_dims,
            ) as mock_subset_dims:
                X_rnd = sample_points_around_best(
                    acq_function=acqf, n_discrete_points=5, sigma=1e-3, bounds=bounds
                )
            self.assertEqual(X_rnd.shape, torch.Size([5, 20]))
            self.assertTrue((X_rnd >= 1).all())
            self.assertTrue((X_rnd <= 2).all())
            mock_subset_dims.assert_called_once()
            # test tiny prob_perturb to make sure we perturb at least one dimension
            X_rnd = sample_points_around_best(
                acq_function=acqf,
                n_discrete_points=5,
                sigma=1e-3,
                bounds=bounds,
                prob_perturb=1e-8,
            )
            self.assertTrue(
                ((X_rnd.unsqueeze(0) == X_train.unsqueeze(1)).all(dim=-1)).sum() == 0
            )
