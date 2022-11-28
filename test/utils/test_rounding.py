#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
from botorch.utils.rounding import (
    approximate_round,
    IdentitySTEFunction,
    OneHotArgmaxSTE,
    RoundSTE,
)
from botorch.utils.testing import BotorchTestCase
from torch.nn.functional import one_hot


class DummySTEFunction(IdentitySTEFunction):
    @staticmethod
    def forward(ctx, X):
        return 2 * X


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


class TestIdentitySTEFunction(BotorchTestCase):
    def test_identity_ste(self):
        for dtype in (torch.float, torch.double):
            X = torch.rand(3, device=self.device, dtype=dtype)
            with self.assertRaises(NotImplementedError):
                IdentitySTEFunction.apply(X)
            X = X.requires_grad_(True)
            X_out = DummySTEFunction.apply(X)
            X_out.sum().backward()
            self.assertTrue(torch.equal(2 * X, X_out))
            self.assertTrue(torch.equal(X.grad, torch.ones_like(X)))


class TestRoundSTE(BotorchTestCase):
    def test_round_ste(self):
        for dtype in (torch.float, torch.double):
            # sample uniformly from the interval [-2.5,2.5]
            X = torch.rand(5, 2, device=self.device, dtype=dtype) * 5 - 2.5
            expected_rounded_X = X.round()
            rounded_X = RoundSTE.apply(X)
            # test forward
            self.assertTrue(torch.equal(expected_rounded_X, rounded_X))
            # test backward
            X = X.requires_grad_(True)
            output = RoundSTE.apply(X)
            # sample some weights to checked that gradients are passed
            # as intended
            w = torch.rand_like(X)
            (w * output).sum().backward()
            self.assertTrue(torch.equal(w, X.grad))


class TestOneHotArgmaxSTE(BotorchTestCase):
    def test_one_hot_argmax_ste(self):
        for dtype in (torch.float, torch.double):
            X = torch.rand(5, 4, device=self.device, dtype=dtype)
            expected_discretized_X = one_hot(
                X.argmax(dim=-1), num_classes=X.shape[-1]
            ).to(X)
            discretized_X = OneHotArgmaxSTE.apply(X)
            # test forward
            self.assertTrue(torch.equal(expected_discretized_X, discretized_X))
            # test backward
            X = X.requires_grad_(True)
            output = OneHotArgmaxSTE.apply(X)
            # sample some weights to checked that gradients are passed
            # as intended
            w = torch.rand_like(X)
            (w * output).sum().backward()
            self.assertTrue(torch.equal(w, X.grad))
