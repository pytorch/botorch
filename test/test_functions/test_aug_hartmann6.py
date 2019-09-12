#! /usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
from botorch.test_functions.aug_hartmann6 import GLOBAL_MAXIMIZER, neg_aug_hartmann6
from botorch.test_functions.hartmann6 import GLOBAL_MAXIMUM
from botorch.utils.testing import BotorchTestCase


class TestNegAugHartmann6(BotorchTestCase):
    def test_single_eval_neg_aug_hartmann6(self):
        for dtype in (torch.float, torch.double):
            X = torch.zeros(7, device=self.device, dtype=dtype)
            res = neg_aug_hartmann6(X)
            self.assertEqual(res.dtype, dtype)
            self.assertEqual(res.device.type, self.device.type)
            self.assertEqual(res.shape, torch.Size())

    def test_batch_eval_neg_aug_hartmann6(self):
        for dtype in (torch.float, torch.double):
            X = torch.zeros(2, 7, device=self.device, dtype=dtype)
            res = neg_aug_hartmann6(X)
            self.assertEqual(res.dtype, dtype)
            self.assertEqual(res.device.type, self.device.type)
            self.assertEqual(res.shape, torch.Size([2]))

    def test_neg_aug_hartmann6_global_maximum(self):
        for dtype in (torch.float, torch.double):
            X = torch.tensor(GLOBAL_MAXIMIZER, device=self.device, dtype=dtype)
            res = neg_aug_hartmann6(X)
            self.assertAlmostEqual(res.item(), GLOBAL_MAXIMUM, places=4)
