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
from botorch.utils import safe_math
from botorch.utils.constants import get_constants_like
from botorch.utils.testing import BotorchTestCase
from torch import finfo, Tensor

INF = float("inf")


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
