#!/usr/bin/env python3

import unittest
from unittest.mock import Mock

import torch
from botorch.acquisition.batch_utils import batch_mode_transform


class testBatchModeTransform(unittest.TestCase):
    def setUp(self):
        self.X = torch.ones(6).view(3, 2)
        self.X2 = 2 * torch.ones(6).view(3, 2)
        self.s = "a"
        self.mock_identity_func = Mock()

        @batch_mode_transform
        def test_func(X, X2, s=""):
            self.mock_identity_func(X, X2, s)
            return X + X2

        self.test_func = test_func

    def test_non_batch_X(self):
        expected_output = self.X + self.X2
        # test args
        output = self.test_func(self.X, self.X2, self.s)
        args = self.mock_identity_func.call_args[0]
        self.assertTrue(torch.equal(args[0], self.X.unsqueeze(0)))
        self.assertTrue(torch.equal(args[1], self.X2.unsqueeze(0)))
        self.assertEqual(args[2], self.s)
        self.assertTrue(torch.equal(output, expected_output))
        # test kwargs
        output2 = self.test_func(self.X, X2=self.X2, s=self.s)
        args2 = self.mock_identity_func.call_args[0]
        self.assertTrue(torch.equal(args2[0], self.X.unsqueeze(0)))
        self.assertTrue(torch.equal(args2[1], self.X2.unsqueeze(0)))
        self.assertEqual(args2[2], self.s)
        self.assertTrue(torch.equal(output2, expected_output))

    def test_batch_X(self):
        batch_X = self.X.unsqueeze(0).expand(2, -1, -1)
        expected_batch_X2 = self.X2.unsqueeze(0).expand(2, -1, -1)
        # test args
        output = self.test_func(batch_X, self.X2, self.s)
        args = self.mock_identity_func.call_args[0]
        self.assertTrue(torch.equal(args[0], batch_X))
        self.assertTrue(torch.equal(args[1], expected_batch_X2))
        self.assertEqual(args[2], self.s)
        self.assertTrue(torch.equal(output, batch_X + expected_batch_X2))
        # test kwargs
        output2 = self.test_func(batch_X, X2=self.X2, s=self.s)
        args2 = self.mock_identity_func.call_args[0]
        self.assertTrue(torch.equal(args2[0], batch_X))
        self.assertTrue(torch.equal(args2[1], expected_batch_X2))
        self.assertEqual(args2[2], self.s)
        self.assertTrue(torch.equal(output2, batch_X + expected_batch_X2))
