#! /usr/bin/env python3

import unittest

import numpy as np
import torch
from botorch.exceptions.errors import UnsupportedError
from botorch.optim.parameter_constraints import (
    _arrayify,
    _make_linear_constraints,
    eval_lin_constraint,
    lin_constraint_jac,
    make_scipy_bounds,
    make_scipy_linear_constraints,
)
from scipy.optimize import Bounds


class TestParameterConstraints(unittest.TestCase):
    def test_arrayify(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double, torch.int, torch.long):
            t = torch.tensor([[1, 2], [3, 4]], device=device).type(dtype)
            t_np = _arrayify(t)
            self.assertIsInstance(t_np, np.ndarray)
            self.assertTrue(t_np.dtype == np.float64)

    def test_arrayify_cuda(self):
        if torch.cuda.is_available():
            self.test_arrayify(cuda=True)

    def test_eval_lin_constraint(self):
        res = eval_lin_constraint(
            flat_idxr=[0, 2],
            coeffs=np.array([1.0, -2.0]),
            rhs=0.5,
            x=np.array([1.0, 2.0, 3.0]),
        )
        self.assertEqual(res, -5.5)

    def test_lin_constraint_jac(self):
        dummy_array = np.array([1.0])
        res = lin_constraint_jac(
            dummy_array, flat_idxr=[0, 2], coeffs=np.array([1.0, -2.0]), n=3
        )
        self.assertTrue(all(np.equal(res, np.array([1.0, 0.0, -2.0]))))

    def test_make_linear_constraints(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        indices = torch.tensor([1, 2], dtype=torch.long, device=device)
        shapeX = torch.Size([3, 2, 4])
        for dtype in (torch.float, torch.double):
            coefficients = torch.tensor([1.0, 2.0], dtype=dtype, device=device)
            constraints = _make_linear_constraints(
                indices=indices,
                coefficients=coefficients,
                rhs=1.0,
                shapeX=shapeX,
                eq=True,
            )
            self.assertTrue(
                all(set(c.keys()) == {"fun", "jac", "type"} for c in constraints)
            )
            self.assertTrue(all(c["type"] == "eq" for c in constraints))
            self.assertEqual(len(constraints), shapeX[:-1].numel())
            x = np.random.rand(shapeX.numel())
            self.assertEqual(constraints[0]["fun"](x), x[1] + 2 * x[2] - 1.0)
            jac_exp = np.zeros(shapeX.numel())
            jac_exp[[1, 2]] = [1, 2]
            self.assertTrue(np.allclose(constraints[0]["jac"](x), jac_exp))
            self.assertEqual(constraints[-1]["fun"](x), x[-3] + 2 * x[-2] - 1.0)
            jac_exp = np.zeros(shapeX.numel())
            jac_exp[[-3, -2]] = [1, 2]
            self.assertTrue(np.allclose(constraints[-1]["jac"](x), jac_exp))
        # check inequality type
        lcs = _make_linear_constraints(
            indices=torch.tensor([1]),
            coefficients=torch.tensor([1.0]),
            rhs=1.0,
            shapeX=torch.Size([1, 1, 2]),
            eq=False,
        )
        self.assertEqual(len(lcs), 1)
        self.assertEqual(lcs[0]["type"], "ineq")

    def test_make_linear_constraints_cuda(self):
        if torch.cuda.is_available():
            self.test_make_linear_constraints(cuda=True)

    def test_make_scipy_linear_constraints(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        shapeX = torch.Size([2, 1, 4])
        res = make_scipy_linear_constraints(
            shapeX=shapeX, inequality_constraints=None, equality_constraints=None
        )
        self.assertEqual(res, [])
        indices = torch.tensor([0, 1], dtype=torch.long, device=device)
        coefficients = torch.tensor([1.5, -1.0], device=device)
        cs = make_scipy_linear_constraints(
            shapeX=shapeX,
            inequality_constraints=[(indices, coefficients, 1.0)],
            equality_constraints=[(indices, coefficients, 1.0)],
        )
        self.assertEqual(len(cs), 4)
        self.assertTrue({c["type"] for c in cs} == {"ineq", "eq"})
        cs = make_scipy_linear_constraints(
            shapeX=shapeX, inequality_constraints=[(indices, coefficients, 1.0)]
        )
        self.assertEqual(len(cs), 2)
        self.assertTrue(all(c["type"] == "ineq" for c in cs))
        cs = make_scipy_linear_constraints(
            shapeX=shapeX, equality_constraints=[(indices, coefficients, 1.0)]
        )
        self.assertEqual(len(cs), 2)
        self.assertTrue(all(c["type"] == "eq" for c in cs))

        # test that len(shapeX) < 3 raises an error
        with self.assertRaises(UnsupportedError):
            make_scipy_linear_constraints(
                shapeX=torch.Size([2, 1]),
                inequality_constraints=[(indices, coefficients, 1.0)],
                equality_constraints=[(indices, coefficients, 1.0)],
            )
        # test that 2-dim indices raises a NotImplementedError
        indices = indices.unsqueeze(0)
        with self.assertRaises(NotImplementedError):
            make_scipy_linear_constraints(
                shapeX=shapeX,
                inequality_constraints=[(indices, coefficients, 1.0)],
                equality_constraints=[(indices, coefficients, 1.0)],
            )
        # test that >2-dim indices raises an UnsupportedError
        indices = indices.unsqueeze(0)
        with self.assertRaises(UnsupportedError):
            make_scipy_linear_constraints(
                shapeX=shapeX,
                inequality_constraints=[(indices, coefficients, 1.0)],
                equality_constraints=[(indices, coefficients, 1.0)],
            )
        # test that out of bounds index raises an error
        indices = torch.tensor([0, 4], dtype=torch.long, device=device)
        with self.assertRaises(RuntimeError):
            make_scipy_linear_constraints(
                shapeX=shapeX,
                inequality_constraints=[(indices, coefficients, 1.0)],
                equality_constraints=[(indices, coefficients, 1.0)],
            )

    def test_make_scipy_linear_constraints_cuda(self):
        if torch.cuda.is_available():
            self.test_make_scipy_linear_constraints(cuda=True)


class TestMakeScipyBounds(unittest.TestCase):
    def test_make_scipy_bounds(self):
        X = torch.zeros(3, 1, 2)
        # both None
        self.assertIsNone(make_scipy_bounds(X=X, lower_bounds=None, upper_bounds=None))
        # lower None
        upper_bounds = torch.ones(2)
        bounds = make_scipy_bounds(X=X, lower_bounds=None, upper_bounds=upper_bounds)
        self.assertIsInstance(bounds, Bounds)
        self.assertTrue(
            np.all(np.equal(bounds.lb, np.full((3, 1, 2), float("-inf")).flatten()))
        )
        self.assertTrue(np.all(np.equal(bounds.ub, np.ones((3, 1, 2)).flatten())))
        # upper None
        lower_bounds = torch.zeros(2)
        bounds = make_scipy_bounds(X=X, lower_bounds=lower_bounds, upper_bounds=None)
        self.assertIsInstance(bounds, Bounds)
        self.assertTrue(np.all(np.equal(bounds.lb, np.zeros((3, 1, 2)).flatten())))
        self.assertTrue(
            np.all(np.equal(bounds.ub, np.full((3, 1, 2), float("inf")).flatten()))
        )
        # floats
        bounds = make_scipy_bounds(X=X, lower_bounds=0.0, upper_bounds=1.0)
        self.assertIsInstance(bounds, Bounds)
        self.assertTrue(np.all(np.equal(bounds.lb, np.zeros((3, 1, 2)).flatten())))
        self.assertTrue(np.all(np.equal(bounds.ub, np.ones((3, 1, 2)).flatten())))

        # 1-d tensors
        bounds = make_scipy_bounds(
            X=X, lower_bounds=lower_bounds, upper_bounds=upper_bounds
        )
        self.assertIsInstance(bounds, Bounds)
        self.assertTrue(np.all(np.equal(bounds.lb, np.zeros((3, 1, 2)).flatten())))
        self.assertTrue(np.all(np.equal(bounds.ub, np.ones((3, 1, 2)).flatten())))
