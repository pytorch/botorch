#! /usr/bin/env python3

import unittest

import torch
from botorch.optim.outcome_constraints import soft_eval_constraint


class TestSoftEvalConstraint(unittest.TestCase):
    def test_soft_eval_scalar_constraint(self):
        lhs = torch.tensor([0.0])
        self.assertTrue(torch.equal(soft_eval_constraint(lhs), torch.tensor([0.5])))
        self.assertTrue(
            torch.equal(soft_eval_constraint(lhs, eta=1), torch.tensor([0.5]))
        )

    def test_soft_eval_tensor_constraint(self):
        eta = 0.1
        x = torch.tensor([[3, 9], [0.5, 0.1]], dtype=torch.float)
        res = soft_eval_constraint(eta * x.log(), eta=eta)
        res_expected = 1 / (1 + x)
        self.assertTrue(torch.allclose(res, res_expected))

    def test_invalid_temperature(self):
        with self.assertRaises(ValueError):
            soft_eval_constraint(torch.tensor([0.0]), -0.1)


if __name__ == "__main__":
    unittest.main()
