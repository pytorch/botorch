#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import warnings

import torch
from botorch import settings
from botorch.acquisition.objective import (
    ConstrainedMCObjective,
    GenericMCObjective,
    IdentityMCObjective,
    LinearMCObjective,
    MCAcquisitionObjective,
    ScalarizedObjective,
)
from botorch.utils import apply_constraints
from botorch.utils.testing import BotorchTestCase, _get_test_posterior
from torch import Tensor


def generic_obj_deprecated(samples: Tensor) -> Tensor:
    return torch.log(torch.sum(samples ** 2, dim=-1))


def generic_obj(samples: Tensor, X=None) -> Tensor:
    return generic_obj_deprecated(samples)


def infeasible_con(samples: Tensor) -> Tensor:
    return torch.ones(samples.shape[0:-1], device=samples.device, dtype=samples.dtype)


def feasible_con(samples: Tensor) -> Tensor:
    return -(
        torch.ones(samples.shape[0:-1], device=samples.device, dtype=samples.dtype)
    )


class TestScalarizedObjective(BotorchTestCase):
    def test_affine_acquisition_objective(self):
        for batch_shape, m, dtype in itertools.product(
            ([], [3]), (1, 2), (torch.float, torch.double)
        ):
            offset = torch.rand(1).item()
            weights = torch.randn(m, device=self.device, dtype=dtype)
            obj = ScalarizedObjective(weights=weights, offset=offset)
            posterior = _get_test_posterior(
                batch_shape, m=m, device=self.device, dtype=dtype
            )
            mean, covar = posterior.mvn.mean, posterior.mvn.covariance_matrix
            new_posterior = obj(posterior)
            exp_size = torch.Size(batch_shape + [1, 1])
            self.assertEqual(new_posterior.mean.shape, exp_size)
            new_mean_exp = offset + mean @ weights
            self.assertTrue(torch.allclose(new_posterior.mean[..., -1], new_mean_exp))
            self.assertEqual(new_posterior.variance.shape, exp_size)
            new_covar_exp = ((covar @ weights) @ weights).unsqueeze(-1)
            self.assertTrue(
                torch.allclose(new_posterior.variance[..., -1], new_covar_exp)
            )
            # test error
            with self.assertRaises(ValueError):
                ScalarizedObjective(weights=torch.rand(2, m))


class TestMCAcquisitionObjective(BotorchTestCase):
    def test_abstract_raises(self):
        with self.assertRaises(TypeError):
            MCAcquisitionObjective()


class TestGenericMCObjective(BotorchTestCase):
    def test_generic_mc_objective(self):
        for dtype in (torch.float, torch.double):
            obj = GenericMCObjective(generic_obj)
            samples = torch.randn(1, device=self.device, dtype=dtype)
            self.assertTrue(torch.equal(obj(samples), generic_obj(samples)))
            samples = torch.randn(2, device=self.device, dtype=dtype)
            self.assertTrue(torch.equal(obj(samples), generic_obj(samples)))
            samples = torch.randn(3, 1, device=self.device, dtype=dtype)
            self.assertTrue(torch.equal(obj(samples), generic_obj(samples)))
            samples = torch.randn(3, 2, device=self.device, dtype=dtype)
            self.assertTrue(torch.equal(obj(samples), generic_obj(samples)))

    def test_generic_mc_objective_deprecated(self):
        for dtype in (torch.float, torch.double):
            with warnings.catch_warnings(record=True) as ws, settings.debug(True):
                obj = GenericMCObjective(generic_obj_deprecated)
                warning_msg = (
                    "The `objective` callable of `GenericMCObjective` is expected to "
                    "take two arguments. Passing a callable that expects a single "
                    "argument will result in an error in future versions."
                )
                self.assertTrue(
                    any(issubclass(w.category, DeprecationWarning) for w in ws)
                )
                self.assertTrue(any(warning_msg in str(w.message) for w in ws))
            samples = torch.randn(1, device=self.device, dtype=dtype)
            self.assertTrue(torch.equal(obj(samples), generic_obj(samples)))
            samples = torch.randn(2, device=self.device, dtype=dtype)
            self.assertTrue(torch.equal(obj(samples), generic_obj(samples)))
            samples = torch.randn(3, 1, device=self.device, dtype=dtype)
            self.assertTrue(torch.equal(obj(samples), generic_obj(samples)))
            samples = torch.randn(3, 2, device=self.device, dtype=dtype)
            self.assertTrue(torch.equal(obj(samples), generic_obj(samples)))


class TestConstrainedMCObjective(BotorchTestCase):
    def test_constrained_mc_objective(self):
        for dtype in (torch.float, torch.double):
            # one feasible constraint
            obj = ConstrainedMCObjective(
                objective=generic_obj, constraints=[feasible_con]
            )
            samples = torch.randn(1, device=self.device, dtype=dtype)
            constrained_obj = generic_obj(samples)
            constrained_obj = apply_constraints(
                obj=constrained_obj,
                constraints=[feasible_con],
                samples=samples,
                infeasible_cost=0.0,
            )
            self.assertTrue(torch.equal(obj(samples), constrained_obj))
            # one infeasible constraint
            obj = ConstrainedMCObjective(
                objective=generic_obj, constraints=[infeasible_con]
            )
            samples = torch.randn(2, device=self.device, dtype=dtype)
            constrained_obj = generic_obj(samples)
            constrained_obj = apply_constraints(
                obj=constrained_obj,
                constraints=[infeasible_con],
                samples=samples,
                infeasible_cost=0.0,
            )
            self.assertTrue(torch.equal(obj(samples), constrained_obj))
            # one feasible, one infeasible
            obj = ConstrainedMCObjective(
                objective=generic_obj, constraints=[feasible_con, infeasible_con]
            )
            samples = torch.randn(2, 1, device=self.device, dtype=dtype)
            constrained_obj = generic_obj(samples)
            constrained_obj = apply_constraints(
                obj=constrained_obj,
                constraints=[feasible_con, infeasible_con],
                samples=samples,
                infeasible_cost=0.0,
            )
            self.assertTrue(torch.equal(obj(samples), constrained_obj))
            # one feasible, one infeasible, infeasible_cost
            obj = ConstrainedMCObjective(
                objective=generic_obj,
                constraints=[feasible_con, infeasible_con],
                infeasible_cost=5.0,
            )
            samples = torch.randn(3, 2, device=self.device, dtype=dtype)
            constrained_obj = generic_obj(samples)
            constrained_obj = apply_constraints(
                obj=constrained_obj,
                constraints=[feasible_con, infeasible_con],
                samples=samples,
                infeasible_cost=5.0,
            )
            self.assertTrue(torch.equal(obj(samples), constrained_obj))
            # one feasible, one infeasible, infeasible_cost, higher dimension
            obj = ConstrainedMCObjective(
                objective=generic_obj,
                constraints=[feasible_con, infeasible_con],
                infeasible_cost=5.0,
            )
            samples = torch.randn(4, 3, 2, device=self.device, dtype=dtype)
            constrained_obj = generic_obj(samples)
            constrained_obj = apply_constraints(
                obj=constrained_obj,
                constraints=[feasible_con, infeasible_con],
                samples=samples,
                infeasible_cost=5.0,
            )
            self.assertTrue(torch.equal(obj(samples), constrained_obj))


class TestIdentityMCObjective(BotorchTestCase):
    def test_identity_mc_objective(self):
        for dtype in (torch.float, torch.double):
            obj = IdentityMCObjective()
            # single-element tensor
            samples = torch.randn(1, device=self.device, dtype=dtype)
            self.assertTrue(torch.equal(obj(samples), samples[0]))
            # single-dimensional non-squeezable tensor
            samples = torch.randn(2, device=self.device, dtype=dtype)
            self.assertTrue(torch.equal(obj(samples), samples))
            # two-dimensional squeezable tensor
            samples = torch.randn(3, 1, device=self.device, dtype=dtype)
            self.assertTrue(torch.equal(obj(samples), samples.squeeze(-1)))
            # two-dimensional non-squeezable tensor
            samples = torch.randn(3, 2, device=self.device, dtype=dtype)
            self.assertTrue(torch.equal(obj(samples), samples))


class TestLinearMCObjective(BotorchTestCase):
    def test_linear_mc_objective(self):
        for dtype in (torch.float, torch.double):
            weights = torch.rand(3, device=self.device, dtype=dtype)
            obj = LinearMCObjective(weights=weights)
            samples = torch.randn(4, 2, 3, device=self.device, dtype=dtype)
            self.assertTrue(
                torch.allclose(obj(samples), (samples * weights).sum(dim=-1))
            )
            samples = torch.randn(5, 4, 2, 3, device=self.device, dtype=dtype)
            self.assertTrue(
                torch.allclose(obj(samples), (samples * weights).sum(dim=-1))
            )
            # make sure this errors if sample output dimensions are incompatible
            with self.assertRaises(RuntimeError):
                obj(samples=torch.randn(2, device=self.device, dtype=dtype))
            with self.assertRaises(RuntimeError):
                obj(samples=torch.randn(1, device=self.device, dtype=dtype))
            # make sure we can't construct objectives with multi-dim. weights
            with self.assertRaises(ValueError):
                LinearMCObjective(
                    weights=torch.rand(2, 3, device=self.device, dtype=dtype)
                )
            with self.assertRaises(ValueError):
                LinearMCObjective(
                    weights=torch.tensor(1.0, device=self.device, dtype=dtype)
                )
