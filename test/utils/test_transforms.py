#! /usr/bin/env python3

import unittest

import torch
from botorch.acquisition.utils import batch_mode_transform, match_batch_shape
from botorch.utils.transforms import standardize
from torch import Tensor


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


class BMIMTestClass:
    @batch_mode_transform
    def method(self, X: Tensor) -> None:
        return X


class TestBatchModeTransform(unittest.TestCase):
    def test_batch_mode_transform(self):
        c = BMIMTestClass()
        # non-batch
        X = torch.rand(3, 2)
        Xout = c.method(X)
        self.assertTrue(torch.equal(Xout, X.unsqueeze(0)))
        # batch
        X = X.unsqueeze(0)
        Xout = c.method(X)
        self.assertTrue(torch.equal(Xout, X))


class TestMatchBatchShape(unittest.TestCase):
    def test_match_batch_shape(self):
        X = torch.rand(3, 2)
        Y = torch.rand(1, 3, 2)
        X_tf = match_batch_shape(X, Y)
        self.assertTrue(torch.equal(X_tf, X.unsqueeze(0)))

        X = torch.rand(1, 3, 2)
        Y = torch.rand(2, 3, 2)
        X_tf = match_batch_shape(X, Y)
        self.assertTrue(torch.equal(X_tf, X.repeat(2, 1, 1)))

        X = torch.rand(2, 3, 2)
        Y = torch.rand(1, 3, 2)
        with self.assertRaises(RuntimeError):
            match_batch_shape(X, Y)

    def test_match_batch_shape_multi_dim(self):
        X = torch.rand(1, 3, 2)
        Y = torch.rand(5, 4, 3, 2)
        X_tf = match_batch_shape(X, Y)
        self.assertTrue(torch.equal(X_tf, X.expand(5, 4, 3, 2)))

        X = torch.rand(4, 3, 2)
        Y = torch.rand(5, 4, 3, 2)
        X_tf = match_batch_shape(X, Y)
        self.assertTrue(torch.equal(X_tf, X.repeat(5, 1, 1, 1)))

        X = torch.rand(2, 1, 3, 2)
        Y = torch.rand(2, 4, 3, 2)
        X_tf = match_batch_shape(X, Y)
        self.assertTrue(torch.equal(X_tf, X.repeat(1, 4, 1, 1)))

        X = torch.rand(4, 2, 3, 2)
        Y = torch.rand(4, 3, 3, 2)
        with self.assertRaises(RuntimeError):
            match_batch_shape(X, Y)
