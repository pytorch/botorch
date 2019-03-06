#! /usr/bin/env python3

import unittest

import torch
from botorch.utils.transforms import standardize


class TestStandardize(unittest.TestCase):
    def test_standardize(self):
        X = torch.tensor([0.0, 0.0])
        self.assertTrue(torch.equal(X, standardize(X)))
        X2 = torch.tensor([0.0, 1.0, 1.0, 1.0])
        expected_X2_stdized = torch.tensor([-1.5, 0.5, 0.5, 0.5])
        self.assertTrue(torch.equal(expected_X2_stdized, standardize(X2)))
        X3 = torch.tensor([[0.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]]).transpose(1, 0)
        X3_stdized = standardize(X3)
        self.assertTrue(torch.equal(X3_stdized[:, 0], expected_X2_stdized))
        self.assertTrue(torch.equal(X3_stdized[:, 1], torch.zeros(4)))
