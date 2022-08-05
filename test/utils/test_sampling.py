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
from botorch import settings
from botorch.exceptions.errors import BotorchError
from botorch.exceptions.warnings import SamplingWarning
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.utils.sampling import (
    _convert_bounds_to_inequality_constraints,
    batched_multinomial,
    construct_base_samples,
    construct_base_samples_from_posterior,
    DelaunayPolytopeSampler,
    draw_sobol_samples,
    find_interior_point,
    get_polytope_samples,
    HitAndRunPolytopeSampler,
    manual_seed,
    PolytopeSampler,
    sample_hypersphere,
    sample_simplex,
    sparse_to_dense_constraints,
    normalize_linear_constraints,
)
from botorch.utils.testing import BotorchTestCase
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from torch.quasirandom import SobolEngine


class TestManualSeed(BotorchTestCase):
    def test_manual_seed(self):
        initial_state = torch.random.get_rng_state()
        with manual_seed():
            self.assertTrue(torch.all(torch.random.get_rng_state() == initial_state))
        with manual_seed(1234):
            self.assertFalse(torch.all(torch.random.get_rng_state() == initial_state))
        self.assertTrue(torch.all(torch.random.get_rng_state() == initial_state))


class TestConstructBaseSamples(BotorchTestCase):
    def test_construct_base_samples(self):
        test_shapes = [
            {"batch": [2], "output": [4, 3], "sample": [5]},
            {"batch": [1], "output": [5, 3], "sample": [5, 6]},
            {"batch": [2, 3], "output": [2, 3], "sample": [5]},
        ]
        for tshape, qmc, seed, dtype in itertools.product(
            test_shapes, (False, True), (None, 1234), (torch.float, torch.double)
        ):
            batch_shape = torch.Size(tshape["batch"])
            output_shape = torch.Size(tshape["output"])
            sample_shape = torch.Size(tshape["sample"])
            expected_shape = sample_shape + batch_shape + output_shape
            samples = construct_base_samples(
                batch_shape=batch_shape,
                output_shape=output_shape,
                sample_shape=sample_shape,
                qmc=qmc,
                seed=seed,
                device=self.device,
                dtype=dtype,
            )
            self.assertEqual(samples.shape, expected_shape)
            self.assertEqual(samples.device.type, self.device.type)
            self.assertEqual(samples.dtype, dtype)
        # check that warning is issued if dimensionality is too large
        with warnings.catch_warnings(record=True) as w, settings.debug(True):
            construct_base_samples(
                batch_shape=torch.Size(),
                output_shape=torch.Size([2000, 11]),  # 200 * 11 = 22000 > 21201
                sample_shape=torch.Size([1]),
                qmc=True,
            )
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[-1].category, SamplingWarning))
            exp_str = f"maximum supported by qmc ({SobolEngine.MAXDIM})"
            self.assertTrue(exp_str in str(w[-1].message))

    def test_construct_base_samples_from_posterior(self):  # noqa: C901
        for dtype in (torch.float, torch.double):
            # single-output
            mean = torch.zeros(2, device=self.device, dtype=dtype)
            cov = torch.eye(2, device=self.device, dtype=dtype)
            mvn = MultivariateNormal(mean=mean, covariance_matrix=cov)
            posterior = GPyTorchPosterior(mvn=mvn)
            for sample_shape, qmc, seed in itertools.product(
                (torch.Size([5]), torch.Size([5, 3])), (False, True), (None, 1234)
            ):
                expected_shape = sample_shape + torch.Size([2, 1])
                samples = construct_base_samples_from_posterior(
                    posterior=posterior, sample_shape=sample_shape, qmc=qmc, seed=seed
                )
                self.assertEqual(samples.shape, expected_shape)
                self.assertEqual(samples.device.type, self.device.type)
                self.assertEqual(samples.dtype, dtype)
            # single-output, batch mode
            mean = torch.zeros(2, 2, device=self.device, dtype=dtype)
            cov = torch.eye(2, device=self.device, dtype=dtype).expand(2, 2, 2)
            mvn = MultivariateNormal(mean=mean, covariance_matrix=cov)
            posterior = GPyTorchPosterior(mvn=mvn)
            for sample_shape, qmc, seed, collapse_batch_dims in itertools.product(
                (torch.Size([5]), torch.Size([5, 3])),
                (False, True),
                (None, 1234),
                (False, True),
            ):
                if collapse_batch_dims:
                    expected_shape = sample_shape + torch.Size([1, 2, 1])
                else:
                    expected_shape = sample_shape + torch.Size([2, 2, 1])
                samples = construct_base_samples_from_posterior(
                    posterior=posterior,
                    sample_shape=sample_shape,
                    qmc=qmc,
                    collapse_batch_dims=collapse_batch_dims,
                    seed=seed,
                )
                self.assertEqual(samples.shape, expected_shape)
                self.assertEqual(samples.device.type, self.device.type)
                self.assertEqual(samples.dtype, dtype)
            # multi-output
            mean = torch.zeros(2, 2, device=self.device, dtype=dtype)
            cov = torch.eye(4, device=self.device, dtype=dtype)
            mtmvn = MultitaskMultivariateNormal(mean=mean, covariance_matrix=cov)
            posterior = GPyTorchPosterior(mvn=mtmvn)
            for sample_shape, qmc, seed in itertools.product(
                (torch.Size([5]), torch.Size([5, 3])), (False, True), (None, 1234)
            ):
                expected_shape = sample_shape + torch.Size([2, 2])
                samples = construct_base_samples_from_posterior(
                    posterior=posterior, sample_shape=sample_shape, qmc=qmc, seed=seed
                )
                self.assertEqual(samples.shape, expected_shape)
                self.assertEqual(samples.device.type, self.device.type)
                self.assertEqual(samples.dtype, dtype)
            # multi-output, batch mode
            mean = torch.zeros(2, 2, 2, device=self.device, dtype=dtype)
            cov = torch.eye(4, device=self.device, dtype=dtype).expand(2, 4, 4)
            mtmvn = MultitaskMultivariateNormal(mean=mean, covariance_matrix=cov)
            posterior = GPyTorchPosterior(mvn=mtmvn)
            for sample_shape, qmc, seed, collapse_batch_dims in itertools.product(
                (torch.Size([5]), torch.Size([5, 3])),
                (False, True),
                (None, 1234),
                (False, True),
            ):
                if collapse_batch_dims:
                    expected_shape = sample_shape + torch.Size([1, 2, 2])
                else:
                    expected_shape = sample_shape + torch.Size([2, 2, 2])
                samples = construct_base_samples_from_posterior(
                    posterior=posterior,
                    sample_shape=sample_shape,
                    qmc=qmc,
                    collapse_batch_dims=collapse_batch_dims,
                    seed=seed,
                )
                self.assertEqual(samples.shape, expected_shape)
                self.assertEqual(samples.device.type, self.device.type)
                self.assertEqual(samples.dtype, dtype)


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
                    torch.tensor([1, 2, 0], **tkwargs),
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
            self.assertTrue(np.allclose(new_constraints[0][-1], expected_rhs))

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

    def test_get_polytope_samples(self):
        tkwargs = {"device": self.device}
        for dtype in (torch.float, torch.double):
            tkwargs["dtype"] = dtype
            bounds = torch.zeros(2, 4, **tkwargs)
            bounds[1] = 1
            inequality_constraints = [
                (
                    torch.tensor([3], **tkwargs),
                    torch.tensor([-4], **tkwargs),
                    -3,
                )
            ]
            equality_constraints = [
                (
                    torch.tensor([0], **tkwargs),
                    torch.tensor([1], **tkwargs),
                    0.5,
                )
            ]
            dense_equality_constraints = sparse_to_dense_constraints(
                d=4, constraints=equality_constraints
            )
            for normalize_bounds in [True, False]:
                with manual_seed(0):
                    samps = get_polytope_samples(
                        n=5,
                        bounds=bounds,
                        inequality_constraints=inequality_constraints,
                        equality_constraints=equality_constraints,
                        seed=0,
                        thinning=3,
                        n_burnin=2,
                        normalize_bounds=normalize_bounds,
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
                        normalize_bounds=normalize_bounds,
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
                        normalize_bounds=normalize_bounds,
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
