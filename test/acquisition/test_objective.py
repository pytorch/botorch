#! /usr/bin/env python3

import unittest

import torch
from botorch.acquisition.objective import (
    ConstrainedMCObjective,
    GenericMCObjective,
    IdentityMCObjective,
    LinearMCObjective,
    MCAcquisitionObjective,
)
from botorch.utils import apply_constraints
from torch import Tensor


def generic_obj(samples: Tensor) -> Tensor:
    return torch.log(torch.sum(samples ** 2, dim=-1))


def infeasible_con(samples: Tensor) -> Tensor:
    return torch.ones(samples.shape[0:-1], device=samples.device, dtype=samples.dtype)


def feasible_con(samples: Tensor) -> Tensor:
    return -torch.ones(samples.shape[0:-1], device=samples.device, dtype=samples.dtype)


class TestMCAcquisitionObjective(unittest.TestCase):
    def test_abstract_raises(self):
        with self.assertRaises(TypeError):
            MCAcquisitionObjective()


class TestGenericMCObjective(unittest.TestCase):
    def test_generic_mc_objective(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            obj = GenericMCObjective(generic_obj)
            samples = torch.randn(1, device=device, dtype=dtype)
            self.assertTrue(torch.equal(obj(samples), generic_obj(samples)))
            samples = torch.randn(2, device=device, dtype=dtype)
            self.assertTrue(torch.equal(obj(samples), generic_obj(samples)))
            samples = torch.randn(3, 1, device=device, dtype=dtype)
            self.assertTrue(torch.equal(obj(samples), generic_obj(samples)))
            samples = torch.randn(3, 2, device=device, dtype=dtype)
            self.assertTrue(torch.equal(obj(samples), generic_obj(samples)))

    def test_generic_mc_objective_cuda(self, cuda=False):
        if torch.cuda.is_available():
            self.test_generic_mc_objective(cuda=True)


class TestConstrainedMCObjective(unittest.TestCase):
    def test_constrained_mc_objective(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            # one feasible constraint
            obj = ConstrainedMCObjective(
                objective=generic_obj, constraints=[feasible_con]
            )
            samples = torch.randn(1, device=device, dtype=dtype)
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
            samples = torch.randn(2, device=device, dtype=dtype)
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
            samples = torch.randn(2, 1, device=device, dtype=dtype)
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
            samples = torch.randn(3, 2, device=device, dtype=dtype)
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
            samples = torch.randn(4, 3, 2, device=device, dtype=dtype)
            constrained_obj = generic_obj(samples)
            constrained_obj = apply_constraints(
                obj=constrained_obj,
                constraints=[feasible_con, infeasible_con],
                samples=samples,
                infeasible_cost=5.0,
            )
            self.assertTrue(torch.equal(obj(samples), constrained_obj))

    def test_constrained_mc_objective_cuda(self, cuda=False):
        if torch.cuda.is_available():
            self.test_constrained_mc_objective(cuda=True)


class TestIdentityMCObjective(unittest.TestCase):
    def test_identity_mc_objective(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            obj = IdentityMCObjective()
            # single-element tensor
            samples = torch.randn(1, device=device, dtype=dtype)
            self.assertTrue(torch.equal(obj(samples), samples[0]))
            # single-dimensional non-squeezable tensor
            samples = torch.randn(2, device=device, dtype=dtype)
            self.assertTrue(torch.equal(obj(samples), samples))
            # two-dimensional squeezable tensor
            samples = torch.randn(3, 1, device=device, dtype=dtype)
            self.assertTrue(torch.equal(obj(samples), samples.squeeze(-1)))
            # two-dimensional non-squeezable tensor
            samples = torch.randn(3, 2, device=device, dtype=dtype)
            self.assertTrue(torch.equal(obj(samples), samples))

    def test_identity_mc_objective_cuda(self, cuda=False):
        if torch.cuda.is_available():
            self.test_identity_mc_objective(cuda=True)


class TestLinearMCObjective(unittest.TestCase):
    def test_linear_mc_objective(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            weights = torch.rand(3, device=device, dtype=dtype)
            obj = LinearMCObjective(weights=weights)
            samples = torch.randn(4, 2, 3, device=device, dtype=dtype)
            self.assertTrue(
                torch.allclose(obj(samples), (samples * weights).sum(dim=-1))
            )
            samples = torch.randn(5, 4, 2, 3, device=device, dtype=dtype)
            self.assertTrue(
                torch.allclose(obj(samples), (samples * weights).sum(dim=-1))
            )
            # make sure this errors if sample output dimensions are incompatible
            with self.assertRaises(RuntimeError):
                obj(samples=torch.randn(2, device=device, dtype=dtype))
            with self.assertRaises(RuntimeError):
                obj(samples=torch.randn(1, device=device, dtype=dtype))
            # make sure we can't construct objectives with multi-dim. weights
            with self.assertRaises(ValueError):
                LinearMCObjective(weights=torch.rand(2, 3, device=device, dtype=dtype))
            with self.assertRaises(ValueError):
                LinearMCObjective(weights=torch.tensor(1.0, device=device, dtype=dtype))

    def test_linear_mc_objective_cuda(self, cuda=False):
        if torch.cuda.is_available():
            self.test_linear_mc_objective(cuda=True)
