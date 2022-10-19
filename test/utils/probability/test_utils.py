#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch
from botorch.utils.probability import utils
from botorch.utils.testing import BotorchTestCase
from numpy.polynomial.legendre import leggauss as numpy_leggauss


class TestProbabilityUtils(BotorchTestCase):
    def test_case_dispatcher(self):
        with torch.random.fork_rng():
            torch.random.manual_seed(0)
            values = torch.rand([32])

        # Test default case
        output = utils.case_dispatcher(
            out=torch.full_like(values, float("nan")),
            default=lambda mask: 0,
        )
        self.assertTrue(output.eq(0).all())

        # Test randomized value assignments
        levels = 0.25, 0.5, 0.75
        cases = [  # switching cases
            (lambda level=level: values < level, lambda mask, i=i: i)
            for i, level in enumerate(levels)
        ]

        cases.append(  # dummy case whose predicate is always False
            (lambda: torch.full(values.shape, False), lambda mask: float("nan"))
        )

        output = utils.case_dispatcher(
            out=torch.full_like(values, float("nan")),
            cases=cases,
            default=lambda mask: len(levels),
        )

        self.assertTrue(output.isfinite().all())
        active = torch.full(values.shape, True)
        for i, level in enumerate(levels):
            mask = active & (values < level)
            self.assertTrue(output[mask].eq(i).all())
            active[mask] = False
        self.assertTrue(~active.any() or output[active].eq(len(levels)).all())

        # testing mask.all() branch
        edge_cases = [
            (lambda: torch.full(values.shape, True), lambda mask: float("nan"))
        ]
        output = utils.case_dispatcher(
            out=torch.full_like(values, float("nan")),
            cases=edge_cases,
            default=lambda mask: len(levels),
        )

        # testing if not active.any() branch
        pred = torch.full(values.shape, True)
        pred[0] = False
        edge_cases = [
            (lambda: pred, lambda mask: False),
            (lambda: torch.full(values.shape, True), lambda mask: False),
        ]
        output = utils.case_dispatcher(
            out=torch.full_like(values, float("nan")),
            cases=edge_cases,
            default=lambda mask: len(levels),
        )

    def test_build_positional_indices(self):
        with torch.random.fork_rng():
            torch.random.manual_seed(0)
            values = torch.rand(3, 2, 5)

        for dim in (values.ndim, -values.ndim - 1):
            with self.assertRaisesRegex(ValueError, r"dim=(-?\d+) invalid for shape"):
                utils.build_positional_indices(shape=values.shape, dim=dim)

        start = utils.build_positional_indices(shape=values.shape, dim=-2)
        self.assertEqual(start.shape, values.shape[:-1])
        self.assertTrue(start.remainder(values.shape[-1]).eq(0).all())

        max_values, max_indices = values.max(dim=-1)
        self.assertTrue(values.view(-1)[start + max_indices].equal(max_values))

    def test_leggaus(self):
        for a, b in zip(utils.leggauss(20, dtype=torch.float64), numpy_leggauss(20)):
            self.assertEqual(a.dtype, torch.float64)
            self.assertTrue((a.numpy() == b).all())

    def test_swap_along_dim_(self):
        with torch.random.fork_rng():
            torch.random.manual_seed(0)
            values = torch.rand(3, 2, 5)

        start = utils.build_positional_indices(shape=values.shape, dim=-2)
        min_values, i = values.min(dim=-1)
        max_values, j = values.max(dim=-1)
        out = utils.swap_along_dim_(values.clone(), i=i, j=j, dim=-1)

        # Verify that positions of minimum and maximum values were swapped
        for vec, min_val, min_idx, max_val, max_idx in zip(
            out.view(-1, values.shape[-1]),
            min_values.ravel(),
            i.ravel(),
            max_values.ravel(),
            j.ravel(),
        ):
            self.assertEqual(vec[min_idx], max_val)
            self.assertEqual(vec[max_idx], min_val)

        start = utils.build_positional_indices(shape=values.shape, dim=-2)
        i_lidx = (start + i).ravel()
        j_lidx = (start + j).ravel()

        # Test passing in a pre-allocated copy buffer
        temp = values.view(-1).clone()[i_lidx]
        buff = torch.empty_like(temp)
        out2 = utils.swap_along_dim_(values.clone(), i=i, j=j, dim=-1, buffer=buff)
        self.assertTrue(out.equal(out2))
        self.assertTrue(temp.equal(buff))

        # Test homogeneous swaps
        temp = utils.swap_along_dim_(values.clone(), i=0, j=2, dim=-1)
        self.assertTrue(values[..., 0].equal(temp[..., 2]))
        self.assertTrue(values[..., 2].equal(temp[..., 0]))

        # Test exception handling
        with self.assertRaisesRegex(ValueError, "Batch shapes of `i`"):
            utils.swap_along_dim_(values, i=i.unsqueeze(-1), j=j, dim=-1)

        with self.assertRaisesRegex(ValueError, "Batch shapes of `j`"):
            utils.swap_along_dim_(values, i=i, j=j.unsqueeze(-1), dim=-1)

        with self.assertRaisesRegex(ValueError, "at most 1-dimensional"):
            utils.swap_along_dim_(values.view(-1), i=i, j=j_lidx, dim=0)

        with self.assertRaisesRegex(ValueError, "at most 1-dimensional"):
            utils.swap_along_dim_(values.view(-1), i=i_lidx, j=j, dim=0)
