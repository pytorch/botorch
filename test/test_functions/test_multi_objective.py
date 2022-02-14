#! /usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
from botorch.test_functions.multi_objective import (
    BNH,
    BraninCurrin,
    C2DTLZ2,
    CarSideImpact,
    CONSTR,
    ConstrainedBraninCurrin,
    DiscBrake,
    DH1,
    DH2,
    DH3,
    DH4,
    DTLZ1,
    DTLZ2,
    DTLZ3,
    DTLZ4,
    DTLZ5,
    DTLZ7,
    MultiObjectiveTestProblem,
    MW7,
    OSY,
    Penicillin,
    SRN,
    ToyRobust,
    VehicleSafety,
    WeldedBeam,
    ZDT1,
    ZDT2,
    ZDT3,
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


class TestDH(MultiObjectiveTestProblemBaseTestCase, BotorchTestCase):
    functions = [DH1(dim=2), DH2(dim=3), DH3(dim=4), DH4(dim=5)]
    dims = [2, 3, 4, 5]
    bounds = [
        [[0.0, -1], [1, 1]],
        [[0.0, -1, -1], [1, 1, 1]],
        [[0.0, 0, -1, -1], [1, 1, 1, 1]],
        [[0.0, -0.15, -1, -1, -1], [1, 1, 1, 1, 1]],
    ]
    expected = [
        [[0.0, 1.0], [1.0, 1.0 / 1.2 + 1.0]],
        [[0.0, 1.0], [1.0, 2.0 / 1.2 + 20.0]],
        [[0.0, 1.88731], [1.0, 1.9990726 * 100]],
        [[0.0, 1.88731], [1.0, 150.0]],
    ]

    def test_init(self):
        for i, f in enumerate(self.functions):
            with self.assertRaises(ValueError):
                f.__class__(dim=1)
            self.assertEqual(f.num_objectives, 2)
            self.assertEqual(f.dim, self.dims[i])
            self.assertTrue(
                torch.equal(f.bounds, torch.tensor(self.bounds[i]).to(f.bounds))
            )

    def test_function_values(self):
        for i, f in enumerate(self.functions):
            test_X = torch.zeros(2, self.dims[i], device=self.device)
            test_X[1] = 1.0
            actual = f(test_X)
            expected = torch.tensor(self.expected[i], device=self.device)
            self.assertTrue(torch.allclose(actual, expected))


class TestDTLZ(MultiObjectiveTestProblemBaseTestCase, BotorchTestCase):
    functions = [
        DTLZ1(dim=5, num_objectives=2),
        DTLZ2(dim=5, num_objectives=2),
        DTLZ3(dim=5, num_objectives=2),
        DTLZ4(dim=5, num_objectives=2),
        DTLZ5(dim=5, num_objectives=2),
        DTLZ7(dim=5, num_objectives=2),
    ]

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
                    if isinstance(f, (DTLZ5, DTLZ7)):
                        with self.assertRaises(NotImplementedError):
                            f.gen_pareto_front(n=1)
                    else:
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
                        elif isinstance(f, (DTLZ2, DTLZ3, DTLZ4)):
                            # assert the points lie on the surface
                            # of the unit hypersphere
                            self.assertTrue(
                                torch.allclose(
                                    pareto_f.pow(2).sum(dim=-1),
                                    torch.ones(
                                        pareto_f.shape[0],
                                        dtype=dtype,
                                        device=self.device,
                                    ),
                                )
                            )


class TestMW7(ConstrainedMultiObjectiveTestProblemBaseTestCase, BotorchTestCase):
    functions = [MW7(dim=3)]

    def test_init(self):
        for f in self.functions:
            with self.assertRaises(ValueError):
                f.__class__(dim=1)
            self.assertEqual(f.num_objectives, 2)
            self.assertEqual(f.dim, 3)


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
    functions = [CarSideImpact(), Penicillin(), ToyRobust(), VehicleSafety()]


class TestConstrainedMultiObjectiveProblems(
    ConstrainedMultiObjectiveTestProblemBaseTestCase, BotorchTestCase
):
    functions = [
        BNH(),
        SRN(),
        CONSTR(),
        ConstrainedBraninCurrin(),
        C2DTLZ2(dim=3, num_objectives=2),
        DiscBrake(),
        WeldedBeam(),
        OSY(),
    ]

    def test_c2dtlz2_batch_exception(self):
        f = C2DTLZ2(dim=3, num_objectives=2)
        with self.assertRaises(NotImplementedError):
            f.evaluate_slack_true(torch.empty(1, 1, 3))
