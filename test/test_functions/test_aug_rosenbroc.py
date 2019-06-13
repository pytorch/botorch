#! /usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

import torch
from botorch.test_functions.aug_rosenbrock import GLOBAL_MAXIMUM, neg_aug_rosenbrock


class TestNegAugRosenbrock(unittest.TestCase):
    def test_single_eval_neg_aug_rosenbrock(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            X = torch.zeros(4, device=device, dtype=dtype)
            res = neg_aug_rosenbrock(X)
            self.assertEqual(res.dtype, dtype)
            self.assertEqual(res.device.type, device.type)
            self.assertEqual(res.shape, torch.Size())

    def test_single_eval_neg_aug_rosenbrock_cuda(self):
        if torch.cuda.is_available():
            self.test_single_eval_neg_aug_rosenbrock(cuda=True)

    def test_batch_eval_neg_aug_rosenbrock(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            X = torch.zeros(2, 4, device=device, dtype=dtype)
            res = neg_aug_rosenbrock(X)
            self.assertEqual(res.dtype, dtype)
            self.assertEqual(res.device.type, device.type)
            self.assertEqual(res.shape, torch.Size([2]))

    def test_batch_eval_neg_aug_rosenbrock_cuda(self):
        if torch.cuda.is_available():
            self.test_batch_eval_neg_aug_rosenbrock(cuda=True)

    def test_neg_aug_rosenbrock_global_maximum(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            X = torch.tensor(
                [[1, 1, 1, 1], [1, 0.95, 0.5, 1], [1, 0.91, 0.1, 1], [1, 0.99, 0.9, 1]],
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
            res = neg_aug_rosenbrock(X)
            for r in res:
                self.assertAlmostEqual(r.item(), GLOBAL_MAXIMUM, places=4)
            grad = torch.autograd.grad(res.sum(), X)[0]
            self.assertLess(grad.abs().max().item(), 1e-4)

    def test_neg_aug_rosenbrock_global_maximum_cuda(self):
        if torch.cuda.is_available():
            self.test_neg_aug_rosenbrock_global_maximum(cuda=False)
