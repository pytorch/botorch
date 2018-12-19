#! /usr/bin/env python3

import unittest

import numpy as np
import torch
from botorch.optim.parameter_constraints import (
    _arrayify,
    _make_flat_indexer,
    _make_lin_constraint,
    eval_lin_constraint,
    lin_constraint_jac,
    make_scipy_linear_constraints,
)


class TestParameterConstraints(unittest.TestCase):
    def test_arraify(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double, torch.int, torch.long):
            t = torch.tensor([[1, 2], [3, 4]], device=device).type(dtype)
            t_np = _arrayify(t)
            self.assertIsInstance(t_np, np.ndarray)
            self.assertTrue(t_np.dtype == np.float64)

    def test_arraify_cuda(self):
        if torch.cuda.is_available():
            self.test_arraify(cuda=True)

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

    def test_make_flat_indexer(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        indices = torch.tensor(
            [[0, 1, 2, 1], [1, 1, 3, 2], [1, 2, 0, 4]], dtype=torch.long, device=device
        )
        shape = torch.Size([2, 3, 4, 5])
        flat_idxr = _make_flat_indexer(indices=indices, shape=shape)
        self.assertEqual(flat_idxr, [31, 97, 104])

    def test_make_flat_indexer_cuda(self):
        if torch.cuda.is_available():
            self.test_make_flat_indexer(cuda=True)

    def test_make_lin_constraint(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        indices = torch.tensor([[0, 1], [1, 2]], dtype=torch.long, device=device)
        shapeX = torch.Size([2, 3])
        x = np.array([0, 1, 2, 3, 4, 5])
        fun_expected = 10.0
        jac_expected = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 2.0])
        for dtype in (torch.float, torch.double, torch.int, torch.long):
            coefficients = torch.tensor([1.0, 2.0], dtype=dtype, device=device)
            constraint = _make_lin_constraint(
                indices=indices, coefficients=coefficients, rhs=1.0, shapeX=shapeX
            )
            self.assertTrue(set(constraint.keys()) == {"fun", "jac"})
            self.assertEqual(constraint["fun"](x), fun_expected)
            self.assertTrue(all(np.equal(constraint["jac"](x), jac_expected)))

    def test_make_scipy_linear_constraints(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        shapeX = torch.Size([2, 3])
        res = make_scipy_linear_constraints(
            shapeX=shapeX, inequality_constraints=None, equality_constraints=None
        )
        self.assertEqual(res, ())
        indices = torch.tensor([[0, 0], [1, 1]], dtype=torch.long, device=device)
        coefficients = torch.tensor([1.5, -1.0], device=device)
        res = make_scipy_linear_constraints(
            shapeX=shapeX,
            inequality_constraints=[(indices, coefficients, 1.0)],
            equality_constraints=[(indices, coefficients, 1.0)],
        )
        self.assertEqual(len(res), 2)
        self.assertTrue({c["type"] for c in res} == {"ineq", "eq"})
        res = make_scipy_linear_constraints(
            shapeX=shapeX, inequality_constraints=[(indices, coefficients, 1.0)]
        )
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["type"], "ineq")
        res = make_scipy_linear_constraints(
            shapeX=shapeX, equality_constraints=[(indices, coefficients, 1.0)]
        )
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["type"], "eq")

    def test_make_scipy_linear_constraints_cuda(self):
        if torch.cuda.is_available():
            self.test_make_scipy_linear_constraints(cuda=True)


if __name__ == "__main__":
    unittest.main()
