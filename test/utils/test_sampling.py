#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import itertools
import warnings
from typing import Any, Dict, Type
from unittest import mock

import numpy as np
import torch
from botorch.exceptions.errors import BotorchError
from botorch.models import FixedNoiseGP
from botorch.utils.sampling import (
    _convert_bounds_to_inequality_constraints,
    batched_multinomial,
    DelaunayPolytopeSampler,
    draw_sobol_samples,
    find_interior_point,
    get_polytope_samples,
    HitAndRunPolytopeSampler,
    manual_seed,
    normalize_linear_constraints,
    PolytopeSampler,
    sample_hypersphere,
    sample_simplex,
    sparse_to_dense_constraints,
    optimize_posterior_samples,
)
from botorch.sampling.pathwise import draw_matheron_paths
from botorch.utils.testing import BotorchTestCase


class TestManualSeed(BotorchTestCase):
    def test_manual_seed(self):
        initial_state = torch.random.get_rng_state()
        with manual_seed():
            self.assertTrue(torch.all(torch.random.get_rng_state() == initial_state))
        with manual_seed(1234):
            self.assertFalse(torch.all(torch.random.get_rng_state() == initial_state))
        self.assertTrue(torch.all(torch.random.get_rng_state() == initial_state))


class TestSampleUtils(BotorchTestCase):
    def test_draw_sobol_samples(self):
        batch_shapes = [None, [3, 5], torch.Size([2]), (5, 3, 2, 3), []]
        for d, q, n, batch_shape, seed, dtype in itertools.product(
            (1, 3),
            (1, 2),
            (2, 5),
            batch_shapes,
            (None, 1234),
            (torch.float, torch.double),
        ):
            tkwargs = {"device": self.device, "dtype": dtype}
            bounds = torch.stack([torch.rand(d), 1 + torch.rand(d)]).to(**tkwargs)
            samples = draw_sobol_samples(
                bounds=bounds, n=n, q=q, batch_shape=batch_shape, seed=seed
            )
            batch_shape = batch_shape or torch.Size()
            self.assertEqual(samples.shape, torch.Size([n, *batch_shape, q, d]))
            self.assertTrue(torch.all(samples >= bounds[0]))
            self.assertTrue(torch.all(samples <= bounds[1]))
            self.assertEqual(samples.device.type, self.device.type)
            self.assertEqual(samples.dtype, dtype)

    def test_sample_simplex(self):
        for d, n, qmc, seed, dtype in itertools.product(
            (1, 2, 3), (2, 5), (False, True), (None, 1234), (torch.float, torch.double)
        ):
            samples = sample_simplex(
                d=d, n=n, qmc=qmc, seed=seed, device=self.device, dtype=dtype
            )
            self.assertEqual(samples.shape, torch.Size([n, d]))
            self.assertTrue(torch.all(samples >= 0))
            self.assertTrue(torch.all(samples <= 1))
            self.assertTrue(torch.max((samples.sum(dim=-1) - 1).abs()) < 1e-5)
            self.assertEqual(samples.device.type, self.device.type)
            self.assertEqual(samples.dtype, dtype)

    def test_sample_hypersphere(self):
        for d, n, qmc, seed, dtype in itertools.product(
            (1, 2, 3), (2, 5), (False, True), (None, 1234), (torch.float, torch.double)
        ):
            samples = sample_hypersphere(
                d=d, n=n, qmc=qmc, seed=seed, device=self.device, dtype=dtype
            )
            self.assertEqual(samples.shape, torch.Size([n, d]))
            self.assertTrue(torch.max((samples.pow(2).sum(dim=-1) - 1).abs()) < 1e-5)
            self.assertEqual(samples.device.type, self.device.type)
            self.assertEqual(samples.dtype, dtype)

    def test_batched_multinomial(self):
        num_categories = 5
        num_samples = 4
        Trulse = (True, False)
        for batch_shape, dtype, replacement, use_gen, use_out in itertools.product(
            ([], [3], [2, 3]), (torch.float, torch.double), Trulse, Trulse, Trulse
        ):
            weights = torch.rand(*batch_shape, num_categories, dtype=dtype)
            out = None
            if use_out:
                out = torch.empty(*batch_shape, num_samples, dtype=torch.long)
            samples = batched_multinomial(
                weights,
                num_samples,
                replacement=replacement,
                generator=torch.Generator() if use_gen else None,
                out=out,
            )
            self.assertEqual(samples.shape, torch.Size([*batch_shape, num_samples]))
            if use_out:
                self.assertTrue(torch.equal(samples, out))
            if not replacement:
                for s in samples.view(-1, num_samples):
                    self.assertTrue(torch.unique(s).size(0), num_samples)

    def test_convert_bounds_to_inequality_constraints(self):
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            # test basic case with no indefinite bounds
            lower_bounds = torch.rand(3, **tkwargs)
            upper_bounds = torch.rand_like(lower_bounds) + lower_bounds
            bounds = torch.stack([lower_bounds, upper_bounds], dim=0)
            A, b = _convert_bounds_to_inequality_constraints(bounds=bounds)
            identity = torch.eye(3, **tkwargs)
            self.assertTrue(torch.equal(A[:3], -identity))
            self.assertTrue(torch.equal(A[3:], identity))
            self.assertTrue(torch.equal(b[:3], -bounds[:1].t()))
            self.assertTrue(torch.equal(b[3:], bounds[1:].t()))
            # test filtering of indefinite bounds
            inf = float("inf")
            bounds = torch.tensor(
                [[-3.0, -inf, -inf], [inf, 2.0, inf]],
                **tkwargs,
            )
            A, b = _convert_bounds_to_inequality_constraints(bounds=bounds)
            A_xpct = torch.tensor([[-1.0, -0.0, -0.0], [0.0, 1.0, 0.0]], **tkwargs)
            b_xpct = torch.tensor([[3.0], [2.0]], **tkwargs)
            self.assertTrue(torch.equal(A, A_xpct))
            self.assertTrue(torch.equal(b, b_xpct))

    def test_sparse_to_dense_constraints(self):
        tkwargs = {"device": self.device}
        for dtype in (torch.float, torch.double):
            tkwargs["dtype"] = dtype
            inequality_constraints = [
                (
                    torch.tensor([3], **tkwargs),
                    torch.tensor([4], **tkwargs),
                    3,
                )
            ]
            (A, b) = sparse_to_dense_constraints(
                d=4, constraints=inequality_constraints
            )
            expected_A = torch.tensor([[0.0, 0.0, 0.0, 4.0]], **tkwargs)
            self.assertTrue(torch.equal(A, expected_A))
            expected_b = torch.tensor([[3.0]], **tkwargs)
            self.assertTrue(torch.equal(b, expected_b))

    def test_normalize_linear_constraints(self):
        tkwargs = {"device": self.device}
        for dtype in (torch.float, torch.double):
            tkwargs["dtype"] = dtype
            constraints = [
                (
                    torch.tensor([1, 2, 0], dtype=torch.int64, device=self.device),
                    torch.tensor([1.0, 1.0, 1.0], **tkwargs),
                    1.0,
                )
            ]
            bounds = torch.tensor(
                [[0.1, 0.3, 0.1, 30.0], [0.6, 0.7, 0.7, 700.0]], **tkwargs
            )
            new_constraints = normalize_linear_constraints(bounds, constraints)
            expected_coefficients = torch.tensor([0.4000, 0.6000, 0.5000], **tkwargs)
            self.assertTrue(
                torch.allclose(new_constraints[0][1], expected_coefficients)
            )
            expected_rhs = 0.5
            self.assertAlmostEqual(new_constraints[0][-1], expected_rhs)

    def test_normalize_linear_constraints_wrong_dtype(self):
        for dtype in (torch.float, torch.double):
            with self.subTest(dtype=dtype):
                tkwargs = {"device": self.device, "dtype": dtype}
                constraints = [
                    (
                        torch.ones(3, dtype=torch.float, device=self.device),
                        torch.ones(3, **tkwargs),
                        1.0,
                    )
                ]
                bounds = torch.zeros(2, 4, **tkwargs)
                msg = "tensors used as indices must be long, byte or bool tensors"
                with self.assertRaises(IndexError, msg=msg):
                    normalize_linear_constraints(bounds, constraints)

    def test_find_interior_point(self):
        # basic problem: 1 <= x_1 <= 2, 2 <= x_2 <= 3
        A = np.concatenate([np.eye(2), -np.eye(2)], axis=0)
        b = np.array([2.0, 3.0, -1.0, -2.0])
        x = find_interior_point(A=A, b=b)
        self.assertTrue(np.allclose(x, np.array([1.5, 2.5])))
        # problem w/ negatives variables: -2 <= x_1 <= -1, -3 <= x_2 <= -2
        b = np.array([-1.0, -2.0, 2.0, 3.0])
        x = find_interior_point(A=A, b=b)
        self.assertTrue(np.allclose(x, np.array([-1.5, -2.5])))
        # problem with bound on a single variable: x_1 <= 0
        A = np.array([[1.0, 0.0]])
        b = np.zeros(1)
        x = find_interior_point(A=A, b=b)
        self.assertLessEqual(x[0].item(), 0.0)
        # unbounded problem: x >= 3
        A = np.array([[-1.0]])
        b = np.array([-3.0])
        x = find_interior_point(A=A, b=b)
        self.assertAlmostEqual(x.item(), 5.0, places=4)

    def test_get_polytope_samples_wrong_inequality_constraints_dtype(self):
        for dtype in (torch.float, torch.double):
            with self.subTest(dtype=dtype):
                tkwargs = {"device": self.device, "dtype": dtype}
                bounds = torch.zeros(2, 4, **tkwargs)
                inequality_constraints = [
                    (
                        torch.tensor([3], dtype=torch.float, device=self.device),
                        torch.tensor([-4], **tkwargs),
                        -3,
                    )
                ]

                msg = (
                    "Normalizing `inequality_constraints` failed. Check that the first "
                    "element of `inequality_constraints` is the correct dtype following"
                    " the previous IndexError."
                )
                msg_orig = "tensors used as indices must be long, byte or bool tensors"

                with self.assertRaisesRegex(ValueError, msg), self.assertRaisesRegex(
                    IndexError, msg_orig
                ):
                    get_polytope_samples(
                        n=5,
                        bounds=bounds,
                        inequality_constraints=inequality_constraints,
                    )

    def test_get_polytope_samples_wrong_equality_constraints_dtype(self):
        for dtype in (torch.float, torch.double):
            with self.subTest(dtype=dtype):
                tkwargs = {"device": self.device, "dtype": dtype}
                bounds = torch.zeros(2, 4, **tkwargs)

                equality_constraints = [
                    (
                        torch.tensor([0], dtype=torch.float, device=self.device),
                        torch.tensor([1], **tkwargs),
                        0.5,
                    )
                ]
                msg = (
                    "Normalizing `equality_constraints` failed. Check that the first "
                    "element of `equality_constraints` is the correct dtype following "
                    "the previous IndexError."
                )
                msg_orig = "tensors used as indices must be long, byte or bool tensors"

                with self.assertRaisesRegex(ValueError, msg), self.assertRaisesRegex(
                    IndexError, msg_orig
                ):
                    get_polytope_samples(
                        n=5,
                        bounds=bounds,
                        equality_constraints=equality_constraints,
                    )

    def test_get_polytope_samples(self):
        tkwargs = {"device": self.device}
        for dtype in (torch.float, torch.double):
            tkwargs["dtype"] = dtype
            bounds = torch.zeros(2, 4, **tkwargs)
            bounds[1] = 1
            inequality_constraints = [
                (
                    torch.tensor([3], dtype=torch.int64, device=self.device),
                    torch.tensor([-4], **tkwargs),
                    -3,
                )
            ]
            equality_constraints = [
                (
                    torch.tensor([0], dtype=torch.int64, device=self.device),
                    torch.tensor([1], **tkwargs),
                    0.5,
                )
            ]
            dense_equality_constraints = sparse_to_dense_constraints(
                d=4, constraints=equality_constraints
            )
            with manual_seed(0):
                samps = get_polytope_samples(
                    n=5,
                    bounds=bounds,
                    inequality_constraints=inequality_constraints,
                    equality_constraints=equality_constraints,
                    seed=0,
                    thinning=3,
                    n_burnin=2,
                )
            (A, b) = sparse_to_dense_constraints(
                d=4, constraints=inequality_constraints
            )
            dense_inequality_constraints = (-A, -b)
            with manual_seed(0):
                expected_samps = HitAndRunPolytopeSampler(
                    bounds=bounds,
                    inequality_constraints=dense_inequality_constraints,
                    equality_constraints=dense_equality_constraints,
                    n_burnin=2,
                ).draw(15, seed=0)[::3]
            self.assertTrue(torch.equal(samps, expected_samps))

            # test no equality constraints
            with manual_seed(0):
                samps = get_polytope_samples(
                    n=5,
                    bounds=bounds,
                    inequality_constraints=inequality_constraints,
                    seed=0,
                    thinning=3,
                    n_burnin=2,
                )
            with manual_seed(0):
                expected_samps = HitAndRunPolytopeSampler(
                    bounds=bounds,
                    inequality_constraints=dense_inequality_constraints,
                    n_burnin=2,
                ).draw(15, seed=0)[::3]
            self.assertTrue(torch.equal(samps, expected_samps))

            # test no inequality constraints
            with manual_seed(0):
                samps = get_polytope_samples(
                    n=5,
                    bounds=bounds,
                    equality_constraints=equality_constraints,
                    seed=0,
                    thinning=3,
                    n_burnin=2,
                )
            with manual_seed(0):
                expected_samps = HitAndRunPolytopeSampler(
                    bounds=bounds,
                    equality_constraints=dense_equality_constraints,
                    n_burnin=2,
                ).draw(15, seed=0)[::3]
            self.assertTrue(torch.equal(samps, expected_samps))


class PolytopeSamplerTestBase:
    sampler_class: Type[PolytopeSampler]
    sampler_kwargs: Dict[str, Any] = {}

    def setUp(self):
        self.bounds = torch.zeros(2, 3, device=self.device)
        self.bounds[1] = 1
        self.A = torch.tensor(
            [
                [-1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 4.0, 1.0],
            ],
            device=self.device,
        )
        self.b = torch.tensor([[0.0], [1.0], [0.0], [0.0], [1.0]], device=self.device)
        self.x0 = torch.tensor([0.1, 0.1, 0.1], device=self.device).unsqueeze(-1)

    def test_sample_polytope(self):
        for dtype in (torch.float, torch.double):
            A = self.A.to(dtype)
            b = self.b.to(dtype)
            x0 = self.x0.to(dtype)
            bounds = self.bounds.to(dtype)
            for interior_point in [x0, None]:
                sampler = self.sampler_class(
                    inequality_constraints=(A, b),
                    bounds=bounds,
                    interior_point=interior_point,
                    **self.sampler_kwargs,
                )
                samples = sampler.draw(n=10, seed=1)
                self.assertEqual(((A @ samples.t() - b) > 0).sum().item(), 0)
                self.assertTrue((samples <= bounds[1]).all())
                self.assertTrue((samples >= bounds[0]).all())
                # make sure we can draw mulitple samples
                more_samples = sampler.draw(n=5)
                self.assertEqual(((A @ more_samples.t() - b) > 0).sum().item(), 0)
                self.assertTrue((more_samples <= bounds[1]).all())
                self.assertTrue((more_samples >= bounds[0]).all())

    def test_sample_polytope_with_eq_constraints(self):
        for dtype in (torch.float, torch.double):
            A = self.A.to(dtype)
            b = self.b.to(dtype)
            x0 = self.x0.to(dtype)
            bounds = self.bounds.to(dtype)
            C = torch.tensor([[1.0, -1, 0.0]], device=self.device, dtype=dtype)
            d = torch.zeros(1, 1, device=self.device, dtype=dtype)

            for interior_point in [x0, None]:
                sampler = self.sampler_class(
                    inequality_constraints=(A, b),
                    equality_constraints=(C, d),
                    bounds=bounds,
                    interior_point=interior_point,
                    **self.sampler_kwargs,
                )
                samples = sampler.draw(n=10, seed=1)
                inequality_satisfied = ((A @ samples.t() - b) > 0).sum().item() == 0
                equality_satisfied = (C @ samples.t() - d).abs().sum().item() < 1e-6
                self.assertTrue(inequality_satisfied)
                self.assertTrue(equality_satisfied)
                self.assertTrue((samples <= bounds[1]).all())
                self.assertTrue((samples >= bounds[0]).all())
                # test no inequality constraints
                sampler = self.sampler_class(
                    equality_constraints=(C, d),
                    bounds=bounds,
                    interior_point=interior_point,
                    **self.sampler_kwargs,
                )
                samples = sampler.draw(n=10, seed=1)
                equality_satisfied = (C @ samples.t() - d).abs().sum().item() < 1e-6
                self.assertTrue(equality_satisfied)
                self.assertTrue((samples <= bounds[1]).all())
                self.assertTrue((samples >= bounds[0]).all())
                # test no inequality constraints or bounds
                with self.assertRaises(BotorchError):
                    self.sampler_class(
                        equality_constraints=(C, d),
                        interior_point=interior_point,
                        **self.sampler_kwargs,
                    )

    def test_sample_polytope_1d(self):
        for dtype in (torch.float, torch.double):
            A = torch.tensor(
                [[-1.0, 0.0], [0.0, -1.0], [1.0, 1.0]], device=self.device, dtype=dtype
            )
            b = torch.tensor([[0.0], [0.0], [1.0]], device=self.device, dtype=dtype)
            C = torch.tensor([[1.0, -1.0]], device=self.device, dtype=dtype)
            x0 = torch.tensor([[0.1], [0.1]], device=self.device, dtype=dtype)
            C = torch.tensor([[1.0, -1.0]], device=self.device, dtype=dtype)
            d = torch.tensor([[0.0]], device=self.device, dtype=dtype)
            bounds = self.bounds[:, :2].to(dtype=dtype)
            for interior_point in [x0, None]:
                sampler = self.sampler_class(
                    inequality_constraints=(A, b),
                    equality_constraints=(C, d),
                    bounds=bounds,
                    interior_point=interior_point,
                    **self.sampler_kwargs,
                )
                samples = sampler.draw(n=10, seed=1)
                inequality_satisfied = ((A @ samples.t() - b) > 0).sum().item() == 0
                equality_satisfied = (C @ samples.t() - d).abs().sum().item() < 1e-6
                self.assertTrue(inequality_satisfied)
                self.assertTrue(equality_satisfied)
                self.assertTrue((samples <= bounds[1]).all())
                self.assertTrue((samples >= bounds[0]).all())

    def test_initial_point(self):
        for dtype in (torch.float, torch.double):
            A = torch.tensor(
                [[0.0, -1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 4.0, 0.0]],
                device=self.device,
                dtype=dtype,
            )
            b = torch.tensor([[0.0], [-1.0], [1.0]], device=self.device, dtype=dtype)
            x0 = self.x0.to(dtype)

            # testing for infeasibility of the initial point and
            # infeasibility of the original LP (status 2 of the linprog output).
            for interior_point in [x0, None]:
                with self.assertRaises(ValueError):
                    self.sampler_class(
                        inequality_constraints=(A, b), interior_point=interior_point
                    )

            class Result:
                status = 1
                message = "mock status 1"

            # testing for only status 1 of the LP
            with mock.patch("scipy.optimize.linprog") as mock_linprog:
                mock_linprog.return_value = Result()
                with self.assertRaises(ValueError):
                    self.sampler_class(inequality_constraints=(A, b))


class TestHitAndRunPolytopeSampler(PolytopeSamplerTestBase, BotorchTestCase):
    sampler_class = HitAndRunPolytopeSampler
    sampler_kwargs = {"n_burnin": 2}


class TestDelaunayPolytopeSampler(PolytopeSamplerTestBase, BotorchTestCase):
    sampler_class = DelaunayPolytopeSampler

    def test_sample_polytope_unbounded(self):
        A = torch.tensor(
            [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [0.0, 4.0, 1.0]],
            device=self.device,
        )
        b = torch.tensor([[0.0], [0.0], [0.0], [1.0]], device=self.device)
        with self.assertRaises(ValueError):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.sampler_class(
                    inequality_constraints=(A, b),
                    interior_point=self.x0,
                    **self.sampler_kwargs,
                )


class TestOptimizePosteriorSamples(BotorchTestCase):
    def test_optimize_posterior_samples(self):
        dtypes = (torch.float32, torch.float64)
        dims = 2
        dtype = torch.float64
        eps = 1e-6
        for_testing_speed_kwargs = {"raw_samples": 250, "num_restarts": 3}
        nums_optima = (1, 7)
        batch_shapes = ((), (3,), (5, 2))
        for num_optima, batch_shape in itertools.product(nums_optima, batch_shapes):
            bounds = torch.Tensor([[0, 1]] * dims, dtype=dtype).T
            X = torch.rand(*batch_shape, 52, dims, dtype=dtype)
            Y = torch.pow(X - 0.5, 2).sum(dim=-1, keepdim=True)

            # having a noiseless model all but guarantees that the found optima
            # will be better than the observations
            model = FixedNoiseGP(X, Y, torch.full_like(Y, eps))
            paths = draw_matheron_paths(
                model=model, sample_shape=torch.Size([num_optima])
            )
            X_opt, f_opt = optimize_posterior_samples(
                paths, bounds, **for_testing_speed_kwargs
            )

            correct_X_shape = (num_optima,) + batch_shape + (dims,)
            correct_f_shape = (num_optima,) + batch_shape + (1,)

            self.assertEqual(X_opt.shape, correct_X_shape)
            self.assertEqual(f_opt.shape, correct_f_shape)
            self.assertTrue(torch.all(X_opt >= bounds[0]))
            self.assertTrue(torch.all(X_opt <= bounds[1]))

            # Check that the all found optima are larger than the observations
            # This is not 100% deterministic, but just about.
            self.assertTrue(torch.all((f_opt > Y.max(dim=-2).values)))
