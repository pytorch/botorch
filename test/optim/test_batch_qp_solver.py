#! /usr/bin/env python3

import unittest

import torch
from botorch.optim.batch_qp_solver import batch_solve_lbfgs_qp

from .data import gen_test_data


# TODO: Write tests for helper functions (T35863287)
# class TestBatchSolveLBFGSQPHelpers(unittest.TestCase):
#     def test_expand_constraint_tensors(self):
#         raise NotImplementedError
#
#     def test_gen_Z0(self):
#         raise NotImplementedError
#
#     def test_gen_factor_Z0(self):
#         raise NotImplementedError
#
#     def test_find_alpha(self):
#         raise NotImplementedError
#
#     def test_compute_residuals(self):
#         raise NotImplementedError
#
#     def test_check_convergence(self):
#         raise NotImplementedError
#
#     def test_find_initial_condition(self):
#         raise NotImplementedError


class TestBatchSolveLBFGSQP(unittest.TestCase):
    def test_batch_solve_lbfgs_qp(self, cuda=False, double=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        dtype = torch.double if double else torch.float

        # inequality and equality constraints
        lbfgs, constraints, q, xopt = gen_test_data(
            device=device, dtype=dtype, ineq=True, eq=True
        )
        res = batch_solve_lbfgs_qp(lbfgs=lbfgs, q=q, constraints=constraints)
        self.assertTrue(res.success)
        self.assertTrue(torch.allclose(res.optimizer.x, xopt, atol=1e-4))

        # inequality constraints only
        lbfgs, constraints, q, xopt = gen_test_data(
            device=device, dtype=dtype, ineq=True, eq=False
        )
        res = batch_solve_lbfgs_qp(lbfgs=lbfgs, q=q, constraints=constraints)
        self.assertTrue(res.success)
        self.assertTrue(torch.allclose(res.optimizer.x, xopt, atol=1e-4))

        # unconstrained problem
        lbfgs, constraints, q, xopt = gen_test_data(
            device=device, dtype=dtype, ineq=False, eq=False
        )
        res = batch_solve_lbfgs_qp(lbfgs=lbfgs, q=q, constraints=constraints)
        self.assertTrue(res.success)
        self.assertTrue("Unconstrained" in res.message)
        self.assertTrue(torch.allclose(res.optimizer.x, xopt, atol=1e-4))
        self.assertTrue(all(all(r == 0) for r in res.residuals._asdict().values()))

    def test_batch_solve_lbfgs_qp_double(self):
        self.test_batch_solve_lbfgs_qp(double=True)

    def test_batch_solve_lbfgs_qp_cuda(self):
        if torch.cuda.is_available():
            self.test_batch_solve_lbfgs_qp(cuda=True)

    def test_batch_solve_lbfgs_qp_cuda_double(self):
        if torch.cuda.is_available():
            self.test_batch_solve_lbfgs_qp(cuda=True, double=True)
