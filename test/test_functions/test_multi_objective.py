#! /usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
from botorch.test_functions.multi_objective import (
    BNH,
    C2DTLZ2,
    CONSTR,
    DTLZ1,
    DTLZ2,
    SRN,
    ZDT1,
    ZDT2,
    ZDT3,
    BraninCurrin,
    ConstrainedBraninCurrin,
    MultiObjectiveTestProblem,
    VehicleSafety,
    OSY,
)
from botorch.utils.testing import (
    BotorchTestCase,
    ConstrainedMultiObjectiveTestProblemBaseTestCase,
    MultiObjectiveTestProblemBaseTestCase,
)


class DummyMOProblem(MultiObjectiveTestProblem):
    _ref_point = [0.0, 0.0]
    _num_objectives = 2
    _bounds = [(0.0, 1.0)] * 2

    def evaluate_true(self, X):
        f_X = X + 2
        return -f_X if self.negate else f_X


class TestBaseTestMultiObjectiveProblem(BotorchTestCase):
    def test_base_mo_problem(self):
        for negate in (True, False):
            for noise_std in (None, 1.0):
                f = DummyMOProblem(noise_std=noise_std, negate=negate)
                self.assertEqual(f.noise_std, noise_std)
                self.assertEqual(f.negate, negate)
                for dtype in (torch.float, torch.double):
                    f.to(dtype=dtype, device=self.device)
                    X = torch.rand(3, 2, dtype=dtype, device=self.device)
                    f_X = f.evaluate_true(X)
                    expected_f_X = -(X + 2) if negate else X + 2
                    self.assertTrue(torch.equal(f_X, expected_f_X))
                with self.assertRaises(NotImplementedError):
                    f.gen_pareto_front(1)


class TestBraninCurrin(MultiObjectiveTestProblemBaseTestCase, BotorchTestCase):
    functions = [BraninCurrin()]

    def test_init(self):
        for f in self.functions:
            self.assertEqual(f.num_objectives, 2)
            self.assertEqual(f.dim, 2)


class TestDTLZ(MultiObjectiveTestProblemBaseTestCase, BotorchTestCase):
    functions = [DTLZ1(dim=5, num_objectives=2), DTLZ2(dim=5, num_objectives=2)]

    def test_init(self):
        for f in self.functions:
            with self.assertRaises(ValueError):
                f.__class__(dim=1, num_objectives=2)
            self.assertEqual(f.num_objectives, 2)
            self.assertEqual(f.dim, 5)
            self.assertEqual(f.k, 4)

    def test_gen_pareto_front(self):
        for dtype in (torch.float, torch.double):
            for f in self.functions:
                for negate in (True, False):
                    f.negate = negate
                    f = f.to(dtype=dtype, device=self.device)
                    pareto_f = f.gen_pareto_front(n=10)
                    if negate:
                        pareto_f *= -1
                    self.assertEqual(pareto_f.dtype, dtype)
                    self.assertEqual(pareto_f.device.type, self.device.type)
                    self.assertTrue((pareto_f > 0).all())
                    if isinstance(f, DTLZ1):
                        # assert is the hyperplane sum_i (f(x_i)) = 0.5
                        self.assertTrue(
                            torch.allclose(
                                pareto_f.sum(dim=-1),
                                torch.full(
                                    pareto_f.shape[0:1],
                                    0.5,
                                    dtype=dtype,
                                    device=self.device,
                                ),
                            )
                        )
                    elif isinstance(f, DTLZ2):
                        # assert the points lie on the surface of the unit hypersphere
                        self.assertTrue(
                            torch.allclose(
                                pareto_f.pow(2).sum(dim=-1),
                                torch.ones(
                                    pareto_f.shape[0], dtype=dtype, device=self.device
                                ),
                            )
                        )


class TestZDT(MultiObjectiveTestProblemBaseTestCase, BotorchTestCase):
    functions = [
        ZDT1(dim=3, num_objectives=2),
        ZDT2(dim=3, num_objectives=2),
        ZDT3(dim=3, num_objectives=2),
    ]

    def test_init(self):
        for f in self.functions:
            with self.assertRaises(NotImplementedError):
                f.__class__(dim=3, num_objectives=3)
            with self.assertRaises(NotImplementedError):
                f.__class__(dim=3, num_objectives=1)
            with self.assertRaises(ValueError):
                f.__class__(dim=1, num_objectives=2)
            self.assertEqual(f.num_objectives, 2)
            self.assertEqual(f.dim, 3)

    def test_gen_pareto_front(self):
        for dtype in (torch.float, torch.double):
            for f in self.functions:
                for negate in (True, False):
                    f.negate = negate
                    f = f.to(dtype=dtype, device=self.device)
                    pareto_f = f.gen_pareto_front(n=11)
                    if negate:
                        pareto_f *= -1
                    self.assertEqual(pareto_f.dtype, dtype)
                    self.assertEqual(pareto_f.device.type, self.device.type)
                    if isinstance(f, ZDT1):
                        self.assertTrue(
                            torch.equal(pareto_f[:, 1], 1 - pareto_f[:, 0].sqrt())
                        )
                    elif isinstance(f, ZDT2):
                        self.assertTrue(
                            torch.equal(pareto_f[:, 1], 1 - pareto_f[:, 0].pow(2))
                        )
                    elif isinstance(f, ZDT3):
                        f_0 = pareto_f[:, 0]
                        f_1 = pareto_f[:, 1]
                        # check f_0 is in the expected discontinuous part of the pareto
                        # front
                        self.assertTrue(
                            (
                                (f_0[:3] >= f._parts[0][0])
                                & (f_0[:3] <= f._parts[0][1])
                            ).all()
                        )
                        for i in range(0, 4):
                            f_0_i = f_0[3 + 2 * i : 3 + 2 * (i + 1)]
                            comparison = f_0_i > torch.tensor(
                                f._parts[i + 1], dtype=dtype, device=self.device
                            )
                            self.assertTrue((comparison[..., 0]).all())
                            self.assertTrue((~comparison[..., 1]).all())
                            self.assertTrue(
                                ((comparison[..., 0]) & (~comparison[..., 1])).all()
                            )
                        # check f_1
                        self.assertTrue(
                            torch.equal(
                                f_1,
                                1 - f_0.sqrt() - f_0 * torch.sin(10 * math.pi * f_0),
                            )
                        )


class TestMultiObjectiveProblems(
    MultiObjectiveTestProblemBaseTestCase, BotorchTestCase
):
    functions = [VehicleSafety()]


class TestConstrainedMultiObjectiveProblems(
    ConstrainedMultiObjectiveTestProblemBaseTestCase, BotorchTestCase
):
    functions = [
        BNH(),
        SRN(),
        CONSTR(),
        ConstrainedBraninCurrin(),
        C2DTLZ2(dim=3, num_objectives=2),
        OSY(),
    ]

    def test_c2dtlz2_batch_exception(self):
        f = C2DTLZ2(dim=3, num_objectives=2)
        with self.assertRaises(NotImplementedError):
            f.evaluate_slack_true(torch.empty(1, 1, 3))
