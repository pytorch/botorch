#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math

from unittest.mock import patch

import torch
from botorch.exceptions.errors import BotorchError
from botorch.utils.probability.lin_ess import LinearEllipticalSliceSampler
from botorch.utils.testing import BotorchTestCase


class TestLinearEllipticalSliceSampler(BotorchTestCase):
    def test_univariate(self):
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            # test input validation
            with self.assertRaises(BotorchError) as e:
                LinearEllipticalSliceSampler()
                self.assertTrue(
                    "requires either inequality constraints or bounds" in str(e)
                )
            # special case: N(0, 1) truncated to negative numbers
            A = torch.ones(1, 1, **tkwargs)
            b = torch.zeros(1, 1, **tkwargs)
            x0 = -torch.rand(1, 1, **tkwargs)
            sampler = LinearEllipticalSliceSampler(
                inequality_constraints=(A, b), interior_point=x0
            )
            self.assertIsNone(sampler._mean)
            self.assertIsNone(sampler._covariance_root)
            self.assertTrue(torch.equal(sampler._x, x0))
            self.assertTrue(torch.equal(sampler.x0, x0))
            samples = sampler.draw(n=3)
            self.assertEqual(samples.shape, torch.Size([3, 1]))
            self.assertLessEqual(samples.max().item(), 0.0)
            self.assertFalse(torch.equal(sampler._x, x0))
            # same case as above, but instantiated with bounds
            sampler = LinearEllipticalSliceSampler(
                bounds=torch.tensor([[-float("inf")], [0.0]], **tkwargs),
                interior_point=x0,
            )
            self.assertIsNone(sampler._mean)
            self.assertIsNone(sampler._covariance_root)
            self.assertTrue(torch.equal(sampler._x, x0))
            self.assertTrue(torch.equal(sampler.x0, x0))
            samples = sampler.draw(n=3)
            self.assertEqual(samples.shape, torch.Size([3, 1]))
            self.assertLessEqual(samples.max().item(), 0.0)
            self.assertFalse(torch.equal(sampler._x, x0))
            # same case as above, but with redundant constraints
            sampler = LinearEllipticalSliceSampler(
                inequality_constraints=(A, b),
                bounds=torch.tensor([[-float("inf")], [1.0]], **tkwargs),
                interior_point=x0,
            )
            self.assertIsNone(sampler._mean)
            self.assertIsNone(sampler._covariance_root)
            self.assertTrue(torch.equal(sampler._x, x0))
            self.assertTrue(torch.equal(sampler.x0, x0))
            samples = sampler.draw(n=3)
            self.assertEqual(samples.shape, torch.Size([3, 1]))
            self.assertLessEqual(samples.max().item(), 0.0)
            self.assertFalse(torch.equal(sampler._x, x0))
            # narrow feasible region, automatically find interior point
            sampler = LinearEllipticalSliceSampler(
                inequality_constraints=(A, b),
                bounds=torch.tensor([[-0.25], [float("inf")]], **tkwargs),
            )
            self.assertIsNone(sampler._mean)
            self.assertIsNone(sampler._covariance_root)
            self.assertTrue(torch.all(sampler._is_feasible(sampler.x0)))
            samples = sampler.draw(n=3)
            self.assertEqual(samples.shape, torch.Size([3, 1]))
            self.assertLessEqual(samples.max().item(), 0.0)
            self.assertGreaterEqual(samples.min().item(), -0.25)
            self.assertFalse(torch.equal(sampler._x, x0))
            # non-standard mean / variance
            mean = torch.tensor([[0.25]], **tkwargs)
            covariance_matrix = torch.tensor([[4.0]], **tkwargs)
            error_msg = ".*either covariance_matrix or covariance_root, not both.*"
            with self.assertRaisesRegex(ValueError, error_msg):
                LinearEllipticalSliceSampler(
                    bounds=torch.tensor([[0.0], [float("inf")]], **tkwargs),
                    covariance_matrix=covariance_matrix,
                    covariance_root=covariance_matrix.sqrt(),
                )
            error_msg = ".*Covariance matrix is not positive definite.*"
            with self.assertRaisesRegex(ValueError, error_msg):
                LinearEllipticalSliceSampler(
                    bounds=torch.tensor([[0.0], [float("inf")]], **tkwargs),
                    covariance_matrix=-covariance_matrix,
                )
            sampler = LinearEllipticalSliceSampler(
                bounds=torch.tensor([[0.0], [float("inf")]], **tkwargs),
                mean=mean,
                covariance_matrix=covariance_matrix,
            )
            self.assertTrue(torch.equal(sampler._mean, mean))
            self.assertTrue(
                torch.equal(sampler._covariance_root, covariance_matrix.sqrt())
            )
            self.assertTrue(torch.all(sampler._is_feasible(sampler.x0)))
            samples = sampler.draw(n=4)
            self.assertEqual(samples.shape, torch.Size([4, 1]))
            self.assertGreaterEqual(samples.min().item(), 0.0)
            self.assertFalse(torch.equal(sampler._x, x0))

    def test_bivariate(self):
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            # special case: N(0, I) truncated to positive numbers
            A = -torch.eye(2, **tkwargs)
            b = torch.zeros(2, 1, **tkwargs)
            sampler = LinearEllipticalSliceSampler(inequality_constraints=(A, b))
            self.assertIsNone(sampler._mean)
            self.assertIsNone(sampler._covariance_root)
            self.assertTrue(torch.all(sampler._is_feasible(sampler.x0)))
            samples = sampler.draw(n=3)
            self.assertEqual(samples.shape, torch.Size([3, 2]))
            self.assertGreaterEqual(samples.min().item(), 0.0)
            self.assertFalse(torch.equal(sampler._x, sampler.x0))
            # same case as above, but instantiated with bounds
            sampler = LinearEllipticalSliceSampler(
                bounds=torch.tensor(
                    [[0.0, 0.0], [float("inf"), float("inf")]], **tkwargs
                ),
            )
            self.assertIsNone(sampler._mean)
            self.assertIsNone(sampler._covariance_root)
            self.assertTrue(torch.all(sampler._is_feasible(sampler.x0)))
            samples = sampler.draw(n=3)
            self.assertEqual(samples.shape, torch.Size([3, 2]))
            self.assertGreaterEqual(samples.min().item(), 0.0)
            self.assertFalse(torch.equal(sampler._x, sampler.x0))
            # A case with bounded domain and non-standard mean and covariance
            mean = -3.0 * torch.ones(2, 1, **tkwargs)
            covariance_matrix = torch.tensor([[4.0, 2.0], [2.0, 2.0]], **tkwargs)
            bounds = torch.tensor(
                [[-float("inf"), -float("inf")], [0.0, 0.0]], **tkwargs
            )
            A = torch.ones(1, 2, **tkwargs)
            b = torch.tensor([[-2.0]], **tkwargs)
            sampler = LinearEllipticalSliceSampler(
                inequality_constraints=(A, b),
                bounds=bounds,
                mean=mean,
                covariance_matrix=covariance_matrix,
            )
            self.assertTrue(torch.equal(sampler._mean, mean))
            covar_root_xpct = torch.tensor([[2.0, 0.0], [1.0, 1.0]], **tkwargs)
            self.assertTrue(torch.equal(sampler._covariance_root, covar_root_xpct))
            samples = sampler.draw(n=3)
            self.assertEqual(samples.shape, torch.Size([3, 2]))
            self.assertTrue(sampler._is_feasible(samples.t()).all())
            self.assertFalse(torch.equal(sampler._x, sampler.x0))

    def test_multivariate(self):
        for dtype in (torch.float, torch.double):
            d = 3
            tkwargs = {"device": self.device, "dtype": dtype}
            # special case: N(0, I) truncated to greater than lower_bound
            A = -torch.eye(d, **tkwargs)
            lower_bound = 1
            b = -torch.full((d, 1), lower_bound, **tkwargs)
            sampler = LinearEllipticalSliceSampler(
                inequality_constraints=(A, b), check_feasibility=True
            )
            self.assertIsNone(sampler._mean)
            self.assertIsNone(sampler._covariance_root)
            self.assertTrue(torch.all(sampler._is_feasible(sampler.x0)))
            samples = sampler.draw(n=3)
            self.assertEqual(samples.shape, torch.Size([3, d]))
            self.assertGreaterEqual(samples.min().item(), lower_bound)
            self.assertFalse(torch.equal(sampler._x, sampler.x0))
            # same case as above, but instantiated with bounds
            sampler = LinearEllipticalSliceSampler(
                bounds=torch.tensor(
                    [[lower_bound for _ in range(d)], [float("inf") for _ in range(d)]],
                    **tkwargs,
                ),
            )
            self.assertIsNone(sampler._mean)
            self.assertIsNone(sampler._covariance_root)
            self.assertTrue(torch.all(sampler._is_feasible(sampler.x0)))
            num_samples = 3
            samples = sampler.draw(n=num_samples)
            self.assertEqual(samples.shape, torch.Size([3, d]))
            self.assertGreaterEqual(samples.min().item(), lower_bound)
            self.assertFalse(torch.equal(sampler._x, sampler.x0))
            self.assertEqual(sampler.lifetime_samples, num_samples)
            samples = sampler.draw(n=num_samples)
            self.assertEqual(sampler.lifetime_samples, 2 * num_samples)

            # checking sampling from non-standard normal
            lower_bound = -2
            b = -torch.full((d, 1), lower_bound, **tkwargs)
            mean = torch.arange(d, **tkwargs).view(d, 1)
            cov_matrix = torch.randn(d, d, **tkwargs)
            cov_matrix = cov_matrix @ cov_matrix.T
            # normalizing to maximal unit variance so that sem math below applies
            cov_matrix /= cov_matrix.max()
            interior_point = torch.ones_like(mean)
            for mean_i, cov_i in [
                (None, None),
                (mean, None),
                (None, cov_matrix),
                (mean, cov_matrix),
            ]:
                with self.subTest(mean=mean_i, cov=cov_i):
                    sampler = LinearEllipticalSliceSampler(
                        inequality_constraints=(A, b),
                        interior_point=interior_point,
                        check_feasibility=True,
                        mean=mean_i,
                        covariance_matrix=cov_i,
                    )
                    # checking standardized system of constraints
                    mean_i = torch.zeros_like(mean) if mean_i is None else mean_i
                    cov_root_i = (
                        torch.eye(d, **tkwargs)
                        if cov_i is None
                        else torch.linalg.cholesky_ex(cov_i)[0]
                    )
                    self.assertAllClose(sampler._Az, A @ cov_root_i)
                    self.assertAllClose(sampler._bz, b - A @ mean_i)

                    x = torch.randn_like(mean_i)
                    z = sampler._standardize(x)
                    self.assertAllClose(
                        z,
                        torch.linalg.solve_triangular(
                            cov_root_i, x - mean_i, upper=False
                        ),
                    )
                    self.assertAllClose(sampler._unstandardize(z), x)

                    # checking rejection-free property
                    num_samples = 32
                    samples = sampler.draw(num_samples)
                    self.assertEqual(len(samples.unique(dim=0)), num_samples)

                    # checking mean is approximately equal to unconstrained distribution
                    # mean if the constraint boundary is far away from the unconstrained
                    # mean. NOTE: Expected failure rate due to statistical fluctuations
                    # of 5 sigma is 1 in 1.76 million.
                    # sem ~ 0.7 -> can differentiate from zero mean
                    sem = 5 / math.sqrt(num_samples)
                    sample_mean = samples.mean(dim=0)
                    self.assertAllClose(sample_mean, mean_i.squeeze(-1), atol=sem)

                    # checking that standardization does not change feasibility values
                    X_test = 3 * torch.randn(d, num_samples, **tkwargs)
                    self.assertAllClose(
                        sampler._Az @ sampler._standardize(X_test) - sampler._bz,
                        A @ X_test - b,
                        atol=1e-5,
                    )
                    self.assertAllClose(
                        sampler._is_feasible(
                            sampler._standardize(X_test), standardized=True
                        ),
                        sampler._is_feasible(X_test, standardized=False),
                        atol=1e-5,
                    )

            # thining and burn-in tests
            burnin = 7
            thinning = 2
            sampler = LinearEllipticalSliceSampler(
                inequality_constraints=(A, b),
                check_feasibility=True,
                burnin=burnin,
                thinning=thinning,
            )
            self.assertEqual(sampler.lifetime_samples, burnin)
            num_samples = 2
            samples = sampler.draw(n=num_samples)
            self.assertEqual(samples.shape, torch.Size([num_samples, d]))
            self.assertEqual(
                sampler.lifetime_samples, burnin + num_samples * (thinning + 1)
            )
            samples = sampler.draw(n=num_samples)
            self.assertEqual(
                sampler.lifetime_samples, burnin + 2 * num_samples * (thinning + 1)
            )

            # two special cases of _find_intersection_angles below:
            # 1) testing _find_intersection_angles with a proposal "nu"
            # that ensures that the full ellipse is feasible
            # setting lower bound below the mean to ensure there's no intersection
            lower_bound = -2
            b = -torch.full((d, 1), lower_bound, **tkwargs)
            nu = torch.full((d, 1), lower_bound + 1, **tkwargs)
            sampler = LinearEllipticalSliceSampler(
                interior_point=nu,
                inequality_constraints=(A, b),
                check_feasibility=True,
            )
            nu = torch.full((d, 1), lower_bound + 2, **tkwargs)
            theta_active = sampler._find_active_intersections(nu)
            self.assertTrue(
                torch.equal(theta_active, sampler._full_angular_range.view(-1))
            )
            rot_angle, slices = sampler._find_rotated_intersections(nu)
            self.assertEqual(rot_angle, 0.0)
            self.assertAllClose(
                slices, torch.tensor([[0.0, 2 * torch.pi]], **tkwargs), atol=1e-6
            )

            # 2) testing tangential intersection of ellipse with constraint
            nu = torch.full((d, 1), lower_bound, **tkwargs)
            sampler = LinearEllipticalSliceSampler(
                interior_point=nu,
                inequality_constraints=(A, b),
                check_feasibility=True,
            )
            nu = torch.full((d, 1), lower_bound + 1, **tkwargs)
            # nu[1] += 1
            theta_active = sampler._find_active_intersections(nu)
            self.assertTrue(theta_active.numel() % 2 == 0)

            # testing error message for infeasible sample
            sampler.check_feasibility = True
            infeasible_x = torch.full((d, 1), lower_bound - 1, **tkwargs)
            with patch.object(
                sampler, "_draw_angle", return_value=torch.tensor(0.0, **tkwargs)
            ):
                with patch.object(
                    sampler,
                    "_get_cart_coords",
                    return_value=infeasible_x,
                ):
                    with self.assertRaisesRegex(
                        RuntimeError, "Sampling resulted in infeasible point"
                    ):
                        sampler.step()

            # high dimensional test case
            d = 128
            # this encodes order constraints on all d variables: Ax < b
            # x[i] < x[i + 1]
            A = torch.zeros(d - 1, d, **tkwargs)
            for i in range(d - 1):
                A[i, i] = 1
                A[i, i + 1] = -1
            b = torch.zeros(d - 1, 1, **tkwargs)

            interior_point = torch.arange(d, **tkwargs).unsqueeze(-1) / d - 1 / 2
            sampler = LinearEllipticalSliceSampler(
                inequality_constraints=(A, b),
                interior_point=interior_point,
                check_feasibility=True,
            )
            num_samples = 16
            X_high_d = sampler.draw(n=num_samples)
            self.assertEqual(X_high_d.shape, torch.Size([16, d]))
            self.assertTrue(sampler._is_feasible(X_high_d.T).all())
            self.assertEqual(sampler.lifetime_samples, num_samples)
