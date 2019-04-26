#! /usr/bin/env python3

import unittest

import torch
from botorch.utils.transforms import (
    match_batch_shape,
    normalize,
    squeeze_last_dim,
    standardize,
    t_batch_mode_transform,
    unnormalize,
)
from torch import Tensor


class TestStandardize(unittest.TestCase):
    def test_standardize(self, cuda=False):
        tkwargs = {"device": torch.device("cuda" if cuda else "cpu")}
        for dtype in (torch.float, torch.double):
            tkwargs["dtype"] = dtype
            X = torch.tensor([0.0, 0.0], **tkwargs)
            self.assertTrue(torch.equal(X, standardize(X)))
            X2 = torch.tensor([0.0, 1.0, 1.0, 1.0], **tkwargs)
            expected_X2_stdized = torch.tensor([-1.5, 0.5, 0.5, 0.5], **tkwargs)
            self.assertTrue(torch.equal(expected_X2_stdized, standardize(X2)))
            X3 = torch.tensor(
                [[0.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]], **tkwargs
            ).transpose(1, 0)
            X3_stdized = standardize(X3)
            self.assertTrue(torch.equal(X3_stdized[:, 0], expected_X2_stdized))
            self.assertTrue(torch.equal(X3_stdized[:, 1], torch.zeros(4, **tkwargs)))

    def test_standardize_cuda(self):
        if torch.cuda.is_available():
            self.test_standardize(cuda=True)


class TestNormalizeAndUnnormalize(unittest.TestCase):
    def test_normalize_unnormalize(self, cuda=False):
        tkwargs = {"device": torch.device("cuda" if cuda else "cpu")}
        for dtype in (torch.float, torch.double):
            tkwargs["dtype"] = dtype
            X = torch.tensor([0.0, 0.25, 0.5], **tkwargs).view(-1, 1)
            expected_X_normalized = torch.tensor([0.0, 0.5, 1.0], **tkwargs).view(-1, 1)
            bounds = torch.tensor([0.0, 0.5], **tkwargs).view(-1, 1)
            X_normalized = normalize(X, bounds=bounds)
            self.assertTrue(torch.equal(expected_X_normalized, X_normalized))
            self.assertTrue(torch.equal(X, unnormalize(X_normalized, bounds=bounds)))
            X2 = torch.tensor(
                [[0.25, 0.125, 0.0], [0.25, 0.0, 0.5]], **tkwargs
            ).transpose(1, 0)
            expected_X2_normalized = torch.tensor(
                [[1.0, 0.5, 0.0], [0.5, 0.0, 1.0]], **tkwargs
            ).transpose(1, 0)
            bounds2 = torch.tensor([[0.0, 0.0], [0.25, 0.5]], **tkwargs)
            X2_normalized = normalize(X2, bounds=bounds2)
            self.assertTrue(torch.equal(X2_normalized, expected_X2_normalized))
            self.assertTrue(torch.equal(X2, unnormalize(X2_normalized, bounds=bounds2)))

    def test_normalize_unnormalize_cuda(self):
        if torch.cuda.is_available():
            self.test_normalize_unnormalize(cuda=True)


class BMIMTestClass:
    @t_batch_mode_transform()
    def q_method(self, X: Tensor) -> None:
        return X

    @t_batch_mode_transform(expected_q=1)
    def q1_method(self, X: Tensor) -> None:
        return X


class TestBatchModeTransform(unittest.TestCase):
    def test_t_batch_mode_transform(self):
        c = BMIMTestClass()
        # test with q != 1
        # non-batch
        X = torch.rand(3, 2)
        Xout = c.q_method(X)
        self.assertTrue(torch.equal(Xout, X.unsqueeze(0)))
        # test with expected_q = 1
        with self.assertRaises(AssertionError):
            c.q1_method(X)
        # batch
        X = X.unsqueeze(0)
        Xout = c.q_method(X)
        self.assertTrue(torch.equal(Xout, X))
        # test with expected_q = 1
        with self.assertRaises(AssertionError):
            c.q1_method(X)

        # test with q = 1
        X = torch.rand(1, 2)
        Xout = c.q_method(X)
        self.assertTrue(torch.equal(Xout, X.unsqueeze(0)))
        # test with expected_q = 1
        Xout = c.q1_method(X)
        self.assertTrue(torch.equal(Xout, X.unsqueeze(0)))
        # batch
        X = X.unsqueeze(0)
        Xout = c.q_method(X)
        self.assertTrue(torch.equal(Xout, X))
        # test with expected_q = 1
        Xout = c.q1_method(X)
        self.assertTrue(torch.equal(Xout, X))

        # test single-dim
        X = torch.zeros(1)
        with self.assertRaises(ValueError):
            c.q_method(X)


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


class TestSqueezeLastDim(unittest.TestCase):
    def test_squeeze_last_dim(self):
        Y = torch.rand(2, 1, 1)
        Y_squeezed = squeeze_last_dim(Y=Y)
        self.assertTrue(torch.equal(Y_squeezed, Y.squeeze(-1)))
