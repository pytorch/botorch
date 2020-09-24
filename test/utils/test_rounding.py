#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
from botorch.utils.rounding import approximate_round
from botorch.utils.testing import BotorchTestCase


class TestApproximateRound(BotorchTestCase):
    def test_approximate_round(self):
        for dtype in (torch.float, torch.double):
            X = torch.linspace(-2.5, 2.5, 11, device=self.device, dtype=dtype)
            exact_rounded_X = X.round()
            approx_rounded_X = approximate_round(X)
            # check that approximate rounding is closer to rounded values than
            # the original inputs
            rounded_diffs = (approx_rounded_X - exact_rounded_X).abs()
            diffs = (X - exact_rounded_X).abs()
            self.assertTrue((rounded_diffs <= diffs).all())
            # check that not all gradients are zero
            X.requires_grad_(True)
            approximate_round(X).sum().backward()
            self.assertTrue((X.grad.abs() != 0).any())
