#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from abc import abstractmethod
from itertools import combinations, product
from typing import Callable

import torch
from botorch.exceptions import UnsupportedError
from botorch.utils import safe_math
from botorch.utils.constants import get_constants_like
from botorch.utils.objective import compute_smoothed_feasibility_indicator
from botorch.utils.safe_math import (
    cauchy,
    fatmax,
    fatmoid,
    fatplus,
    log_fatmoid,
    log_fatplus,
    log_softplus,
    logexpit,
    logmeanexp,
    sigmoid,
    smooth_amax,
)
from botorch.utils.testing import BotorchTestCase
from torch import finfo, Tensor
from torch.nn.functional import softplus

INF = float("inf")


def sum_constraint(samples: Tensor) -> Tensor:
    """Represents the constraint `samples.sum(dim=-1) > 0`.

    Args:
        samples: A `b x q x m`-dim Tensor.

    Returns:
        A `b x q`-dim Tensor representing constraint feasibility.
    """
    return -samples.sum(dim=-1)


class UnaryOpTestMixin:
    op: Callable[[Tensor], Tensor]
    safe_op: Callable[[Tensor], Tensor]

    def __init_subclass__(cls, op: Callable, safe_op: Callable):
        cls.op = staticmethod(op)
        cls.safe_op = staticmethod(safe_op)

    def test_generic(self, m: int = 3, n: int = 4):
        for dtype in (torch.float32, torch.float64):
            # Test forward
            x = torch.rand(n, m, dtype=dtype, requires_grad=True, device=self.device)
            y = self.safe_op(x)

            _x = x.detach().clone().requires_grad_(True)
            _y = self.op(_x)
            self.assertTrue(y.equal(_y))

            # Test backward
            y.sum().backward()
            _y.sum().backward()
            self.assertTrue(x.grad.equal(_x.grad))

            # Test passing in pre-allocated `out`
            with torch.no_grad():
                y.zero_()
                self.safe_op(x, out=y)
                self.assertTrue(y.equal(_y))

    @abstractmethod
    def test_special(self):
        pass  # pragma: no cover


class BinaryOpTestMixin:
    op: Callable[[Tensor, Tensor], Tensor]
    safe_op: Callable[[Tensor, Tensor], Tensor]

    def __init_subclass__(cls, op: Callable, safe_op: Callable):
        cls.op = staticmethod(op)
        cls.safe_op = staticmethod(safe_op)

    def test_generic(self, m: int = 3, n: int = 4):
        for dtype in (torch.float32, torch.float64):
            # Test equality for generic cases
            a = torch.rand(n, m, dtype=dtype, requires_grad=True, device=self.device)
            b = torch.rand(n, m, dtype=dtype, requires_grad=True, device=self.device)
            y = self.safe_op(a, b)

            _a = a.detach().clone().requires_grad_(True)
            _b = b.detach().clone().requires_grad_(True)
            _y = self.op(_a, _b)
            self.assertTrue(y.equal(_y))

            # Test backward
            y.sum().backward()
            _y.sum().backward()
            self.assertTrue(a.grad.equal(_a.grad))
            self.assertTrue(b.grad.equal(_b.grad))

    @abstractmethod
    def test_special(self):
        pass  # pragma: no cover


class TestSafeExp(
    BotorchTestCase, UnaryOpTestMixin, op=torch.exp, safe_op=safe_math.exp
):
    def test_special(self):
        for dtype in (torch.float32, torch.float64):
            x = torch.full([], INF, dtype=dtype, requires_grad=True, device=self.device)
            y = self.safe_op(x)
            self.assertEqual(
                y, get_constants_like(math.log(finfo(dtype).max) - 1e-4, x).exp()
            )

            y.backward()
            self.assertEqual(x.grad, 0)


class TestSafeLog(
    BotorchTestCase, UnaryOpTestMixin, op=torch.log, safe_op=safe_math.log
):
    def test_special(self):
        for dtype in (torch.float32, torch.float64):
            x = torch.zeros([], dtype=dtype, requires_grad=True, device=self.device)
            y = self.safe_op(x)
            self.assertEqual(y, math.log(finfo(dtype).tiny))

            y.backward()
            self.assertEqual(x.grad, 0)


class TestSafeAdd(
    BotorchTestCase, BinaryOpTestMixin, op=torch.add, safe_op=safe_math.add
):
    def test_special(self):
        for dtype in (torch.float32, torch.float64):
            for _a in (INF, -INF):
                a = torch.tensor(
                    _a, dtype=dtype, requires_grad=True, device=self.device
                )
                b = torch.tensor(
                    INF, dtype=dtype, requires_grad=True, device=self.device
                )

                out = self.safe_op(a, b)
                self.assertEqual(out, 0 if a != b else b)

                out.backward()
                self.assertEqual(a.grad, 0 if a != b else 1)
                self.assertEqual(b.grad, 0 if a != b else 1)


class TestSafeSub(
    BotorchTestCase, BinaryOpTestMixin, op=torch.sub, safe_op=safe_math.sub
):
    def test_special(self):
        for dtype in (torch.float32, torch.float64):
            for _a in (INF, -INF):
                a = torch.tensor(
                    _a, dtype=dtype, requires_grad=True, device=self.device
                )
                b = torch.tensor(
                    INF, dtype=dtype, requires_grad=True, device=self.device
                )

                out = self.safe_op(a, b)
                self.assertEqual(out, 0 if a == b else -b)

                out.backward()
                self.assertEqual(a.grad, 0 if a == b else 1)
                self.assertEqual(b.grad, 0 if a == b else -1)


class TestSafeMul(
    BotorchTestCase, BinaryOpTestMixin, op=torch.mul, safe_op=safe_math.mul
):
    def test_special(self):
        for dtype in (torch.float32, torch.float64):
            for _a, _b in product([0, 2], [INF, -INF]):
                a = torch.tensor(
                    _a, dtype=dtype, requires_grad=True, device=self.device
                )
                b = torch.tensor(
                    _b, dtype=dtype, requires_grad=True, device=self.device
                )

                out = self.safe_op(a, b)
                self.assertEqual(out, a if a == 0 else b)

                out.backward()
                self.assertEqual(a.grad, 0 if a == 0 else b)
                self.assertEqual(b.grad, 0 if a == 0 else a)


class TestSafeDiv(
    BotorchTestCase, BinaryOpTestMixin, op=torch.div, safe_op=safe_math.div
):
    def test_special(self):
        for dtype in (torch.float32, torch.float64):
            for _a, _b in combinations([0, INF, -INF], 2):
                a = torch.tensor(
                    _a, dtype=dtype, requires_grad=True, device=self.device
                )
                b = torch.tensor(
                    _b, dtype=dtype, requires_grad=True, device=self.device
                )

                out = self.safe_op(a, b)
                if a == b:
                    self.assertEqual(out, 1)
                elif a == -b:
                    self.assertEqual(out, -1)
                else:
                    self.assertEqual(out, a / b)

                out.backward()
                if ((a == 0) & (b == 0)) | (a.isinf() & b.isinf()):
                    self.assertEqual(a.grad, 0)
                    self.assertEqual(b.grad, 0)
                else:
                    self.assertEqual(a.grad, 1 / b)
                    self.assertEqual(b.grad, -a * b**-2)


class TestLogMeanExp(BotorchTestCase):
    def test_log_mean_exp(self):
        for dtype in (torch.float32, torch.float64):
            X = torch.rand(3, 2, 5, dtype=dtype, device=self.device) + 0.1

            # test single-dimension reduction
            self.assertAllClose(logmeanexp(X.log(), dim=-1).exp(), X.mean(dim=-1))
            self.assertAllClose(logmeanexp(X.log(), dim=-2).exp(), X.mean(dim=-2))
            # test tuple of dimensions
            self.assertAllClose(
                logmeanexp(X.log(), dim=(0, -1)).exp(), X.mean(dim=(0, -1))
            )

            # with keepdim
            self.assertAllClose(
                logmeanexp(X.log(), dim=-1, keepdim=True).exp(),
                X.mean(dim=-1, keepdim=True),
            )
            self.assertAllClose(
                logmeanexp(X.log(), dim=-2, keepdim=True).exp(),
                X.mean(dim=-2, keepdim=True),
            )
            self.assertAllClose(
                logmeanexp(X.log(), dim=(0, -1), keepdim=True).exp(),
                X.mean(dim=(0, -1), keepdim=True),
            )


class TestSmoothNonLinearities(BotorchTestCase):
    def test_smooth_non_linearities(self):
        for dtype in (torch.float, torch.double):
            tkwargs = {"dtype": dtype, "device": self.device}
            n = 17
            X = torch.randn(n, **tkwargs)
            self.assertAllClose(cauchy(X), 1 / (X.square() + 1))

            # testing softplus and fatplus
            tau = 1e-2
            fatplus_X = fatplus(X, tau=tau)
            self.assertAllClose(fatplus_X, X.clamp(0), atol=tau)
            self.assertTrue((fatplus_X > 0).all())
            self.assertAllClose(fatplus_X.log(), log_fatplus(X, tau=tau))
            self.assertAllClose(
                softplus(X, beta=1 / tau), log_softplus(X, tau=tau).exp()
            )

            # testing fatplus differentiability
            X = torch.randn(n, **tkwargs)
            X.requires_grad = True
            log_fatplus(X, tau=tau).sum().backward()
            self.assertFalse(X.grad.isinf().any())
            self.assertFalse(X.grad.isnan().any())
            # always increasing, could also test convexity (mathematically guaranteed)
            self.assertTrue((X.grad > 0).all())

            X_soft = X.detach().clone()
            X_soft.requires_grad = True
            log_softplus(X_soft, tau=tau).sum().backward()

            # for positive values away from zero, log_softplus and log_fatplus are close
            is_positive = X > 100 * tau  # i.e. 1 for tau = 1e-2
            self.assertAllClose(X.grad[is_positive], 1 / X[is_positive], atol=tau)
            self.assertAllClose(X_soft.grad[is_positive], 1 / X[is_positive], atol=tau)

            is_negative = X < -100 * tau  # i.e. -1
            # the softplus has very large gradients, which can saturate the smooth
            # approximation to the maximum over the q-batch.
            asym_val = torch.full_like(X_soft.grad[is_negative], 1 / tau)
            self.assertAllClose(X_soft.grad[is_negative], asym_val, atol=tau, rtol=tau)
            # the fatplus on the other hand has smaller, though non-vanishing gradients.
            self.assertTrue((X_soft.grad[is_negative] > X.grad[is_negative]).all())

            # testing smoothmax and fatmax
            d = 3
            X = torch.randn(n, d, **tkwargs)
            fatmax_X = fatmax(X, dim=-1, tau=tau)
            true_max = X.amax(dim=-1)
            self.assertAllClose(fatmax_X, true_max, atol=tau)
            self.assertAllClose(smooth_amax(X, dim=-1, tau=tau), true_max, atol=tau)

            # special case for d = 1
            d = 1
            X = torch.randn(n, d, **tkwargs)
            fatmax_X = fatmax(X, dim=-1, tau=tau)
            self.assertAllClose(fatmax_X, X[..., 0])

            # testing fatmax differentiability
            X = torch.randn(n, **tkwargs)
            X.requires_grad = True
            fatmax(X, dim=-1, tau=tau).sum().backward()

            self.assertFalse(X.grad.isinf().any())
            self.assertFalse(X.grad.isnan().any())
            self.assertTrue(X.grad.min() > 0)

            # testing fatmoid
            X = torch.randn(n, **tkwargs)
            fatmoid_X = fatmoid(X, tau=tau)
            # output is in [0, 1]
            self.assertTrue((fatmoid_X > 0).all())
            self.assertTrue((fatmoid_X < 1).all())
            # skew symmetry
            atol = 1e-6 if dtype == torch.float32 else 1e-12
            self.assertAllClose(1 - fatmoid_X, fatmoid(-X, tau=tau), atol=atol)
            zero = torch.tensor(0.0, **tkwargs)
            half = torch.tensor(0.5, **tkwargs)
            self.assertAllClose(fatmoid(zero), half, atol=atol)
            self.assertAllClose(fatmoid_X.log(), log_fatmoid(X, tau=tau))

            is_center = X.abs() < 100 * tau
            self.assertAllClose(
                fatmoid_X[~is_center], (X[~is_center] > 0).to(fatmoid_X), atol=1e-3
            )

            # testing differentiability
            X.requires_grad = True
            log_fatmoid(X, tau=tau).sum().backward()
            self.assertFalse(X.grad.isinf().any())
            self.assertFalse(X.grad.isnan().any())
            self.assertTrue((X.grad > 0).all())

            # testing constraint indicator
            constraints = [sum_constraint]
            b = 3
            q = 4
            m = 5
            samples = torch.randn(b, q, m, **tkwargs)
            eta = 1e-3
            fat = True
            log_feas_vals = compute_smoothed_feasibility_indicator(
                constraints=constraints,
                samples=samples,
                eta=eta,
                log=True,
                fat=fat,
            )
            self.assertTrue(log_feas_vals.shape == torch.Size([b, q]))
            expected_feas_vals = sum_constraint(samples) < 0
            hard_feas_vals = log_feas_vals.exp() > 1 / 2
            self.assertAllClose(hard_feas_vals, expected_feas_vals)

            # with deterministic inputs:
            samples = torch.ones(1, 1, m, **tkwargs)  # sum is greater than 0
            log_feas_vals = compute_smoothed_feasibility_indicator(
                constraints=constraints,
                samples=samples,
                eta=eta,
                log=True,
                fat=fat,
            )
            self.assertTrue((log_feas_vals.exp() > 1 / 2).item())

            # with deterministic inputs:
            samples = -torch.ones(1, 1, m, **tkwargs)  # sum is smaller than 0
            log_feas_vals = compute_smoothed_feasibility_indicator(
                constraints=constraints,
                samples=samples,
                eta=eta,
                log=True,
                fat=fat,
            )
            self.assertFalse((log_feas_vals.exp() > 1 / 2).item())

            # testing sigmoid wrapper function
            X = torch.randn(3, 4, 5, **tkwargs)
            sigmoid_X = torch.sigmoid(X)
            self.assertAllClose(sigmoid(X), sigmoid_X)
            self.assertAllClose(sigmoid(X, log=True), logexpit(X))
            self.assertAllClose(sigmoid(X, log=True).exp(), sigmoid_X)
            fatmoid_X = fatmoid(X)
            self.assertAllClose(sigmoid(X, fat=True), fatmoid_X)
            self.assertAllClose(sigmoid(X, log=True, fat=True).exp(), fatmoid_X)

        with self.assertRaisesRegex(UnsupportedError, "Only dtypes"):
            log_softplus(torch.randn(2, dtype=torch.float16))
