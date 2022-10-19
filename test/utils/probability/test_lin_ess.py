#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

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
            with self.assertRaises(ValueError) as e:
                LinearEllipticalSliceSampler(
                    bounds=torch.tensor([[0.0], [float("inf")]], **tkwargs),
                    covariance_matrix=covariance_matrix,
                    covariance_root=covariance_matrix.sqrt(),
                )
                self.assertTrue(
                    "either covariance_matrix or covariance_root, not both" in str(e)
                )
                with self.assertRaises(ValueError) as e:
                    LinearEllipticalSliceSampler(
                        bounds=torch.tensor([[0.0], [float("inf")]], **tkwargs),
                        covariance_matrix=-covariance_matrix,
                    )
                    self.assertTrue(
                        "Covariance matrix is not positive definite" in str(e)
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
