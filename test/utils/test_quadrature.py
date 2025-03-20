#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import itertools

import torch
from botorch.utils.quadrature import (
    clenshaw_curtis_quadrature,
    higher_dimensional_quadrature,
)
from botorch.utils.testing import BotorchTestCase


class TestQuadrature(BotorchTestCase):
    def test_clenshaw_curtis_quadrature(self):
        deg_list = [5, 8, 11]
        bounds_list = [
            None,
            (0, 1),
            (-1, 1),
            (torch.tensor(torch.e), torch.tensor(2 * torch.pi)),
        ]

        for (deg, bounds), dtype in itertools.product(
            zip(deg_list, bounds_list), (torch.float32, torch.float64)
        ):
            tkwargs = {"dtype": dtype, "device": self.device}
            if bounds is None:
                x, w = clenshaw_curtis_quadrature(deg=deg, **tkwargs)
                a, b = 0, 1
            else:
                a, b = bounds
                if isinstance(a, torch.Tensor):
                    a = a.to(**tkwargs)
                if isinstance(b, torch.Tensor):
                    b = b.to(**tkwargs)
                x, w = clenshaw_curtis_quadrature(deg=deg, a=a, b=b, **tkwargs)
            self.assertEqual(x[0].item(), a)
            self.assertEqual(x[-1].item(), b)
            self.assertEqual(len(x), deg)
            self.assertAllClose(w.sum(), torch.tensor(b - a, **tkwargs), atol=1e-6)

            # integrates polynomials of degree up to deg exactly
            for i in range(0, deg):
                self.assertAllClose(
                    x.pow(i).dot(w),
                    torch.tensor((b ** (i + 1) - a ** (i + 1)) / (i + 1), **tkwargs),
                    atol=1e-6,
                )

        a, b = 0, 1
        x, w = clenshaw_curtis_quadrature(deg=deg, **tkwargs)
        xd, wd = higher_dimensional_quadrature((x, x), (w, w))
        # testing integral of multi-dimensional additive function
        for i in range(0, deg):
            self.assertAllClose(
                xd.pow(i).sum(dim=-1).dot(wd),
                2 * torch.tensor((b ** (i + 1) - a ** (i + 1)) / (i + 1), **tkwargs),
                atol=1e-6,
            )
