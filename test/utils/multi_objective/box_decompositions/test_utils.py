#! /usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch
from botorch.exceptions.errors import BotorchTensorDimensionError, UnsupportedError
from botorch.utils.multi_objective.box_decompositions.utils import (
    _expand_ref_point,
    _pad_batch_pareto_frontier,
)
from botorch.utils.testing import BotorchTestCase


class TestExpandRefPoint(BotorchTestCase):
    def test_expand_ref_point(self):
        ref_point = torch.tensor([1.0, 2.0], device=self.device)
        for dtype in (torch.float, torch.double):
            ref_point = ref_point.to(dtype=dtype)
            # test non-batch
            self.assertTrue(
                torch.equal(
                    _expand_ref_point(ref_point, batch_shape=torch.Size([])),
                    ref_point,
                )
            )
            self.assertTrue(
                torch.equal(
                    _expand_ref_point(ref_point, batch_shape=torch.Size([3])),
                    ref_point.unsqueeze(0).expand(3, -1),
                )
            )
            # test ref point with wrong shape batch_shape
            with self.assertRaises(BotorchTensorDimensionError):
                _expand_ref_point(ref_point.unsqueeze(0), batch_shape=torch.Size([]))
            with self.assertRaises(BotorchTensorDimensionError):
                _expand_ref_point(ref_point.unsqueeze(0).expand(3, -1), torch.Size([2]))


class TestPadBatchParetoFrontier(BotorchTestCase):
    def test_pad_batch_pareto_frontier(self):
        for dtype in (torch.float, torch.double):
            Y1 = torch.tensor(
                [
                    [1.0, 5.0],
                    [10.0, 3.0],
                    [4.0, 5.0],
                    [4.0, 5.0],
                    [5.0, 5.0],
                    [8.5, 3.5],
                    [8.5, 3.5],
                    [8.5, 3.0],
                    [9.0, 1.0],
                    [8.0, 1.0],
                ],
                dtype=dtype,
                device=self.device,
            )

            Y2 = torch.tensor(
                [
                    [1.0, 9.0],
                    [10.0, 3.0],
                    [4.0, 5.0],
                    [4.0, 5.0],
                    [5.0, 5.0],
                    [8.5, 3.5],
                    [8.5, 3.5],
                    [8.5, 3.0],
                    [9.0, 5.0],
                    [9.0, 4.0],
                ],
                dtype=dtype,
                device=self.device,
            )
            Y = torch.stack([Y1, Y2], dim=0)
            ref_point = torch.full((2, 2), 2.0, dtype=dtype, device=self.device)
            padded_pareto = _pad_batch_pareto_frontier(
                Y=Y, ref_point=ref_point, is_pareto=False
            )
            expected_nondom_Y1 = torch.tensor(
                [[10.0, 3.0], [5.0, 5.0], [8.5, 3.5]],
                dtype=dtype,
                device=self.device,
            )
            expected_padded_nondom_Y2 = torch.tensor(
                [
                    [10.0, 3.0],
                    [9.0, 5.0],
                    [9.0, 5.0],
                ],
                dtype=dtype,
                device=self.device,
            )
            expected_padded_pareto = torch.stack(
                [expected_nondom_Y1, expected_padded_nondom_Y2], dim=0
            )
            self.assertTrue(torch.equal(padded_pareto, expected_padded_pareto))

            # test feasibility mask
            feas = (Y >= 9.0).any(dim=-1)
            expected_nondom_Y1 = torch.tensor(
                [[10.0, 3.0], [10.0, 3.0]],
                dtype=dtype,
                device=self.device,
            )
            expected_padded_nondom_Y2 = torch.tensor(
                [[10.0, 3.0], [9.0, 5.0]],
                dtype=dtype,
                device=self.device,
            )
            expected_padded_pareto = torch.stack(
                [expected_nondom_Y1, expected_padded_nondom_Y2], dim=0
            )
            padded_pareto = _pad_batch_pareto_frontier(
                Y=Y, ref_point=ref_point, feasibility_mask=feas, is_pareto=False
            )
            self.assertTrue(torch.equal(padded_pareto, expected_padded_pareto))

            # test is_pareto=True
            # one row of Y2 should be dropped because it is not better than the
            # reference point
            Y1 = torch.tensor(
                [[10.0, 3.0], [5.0, 5.0], [8.5, 3.5]],
                dtype=dtype,
                device=self.device,
            )
            Y2 = torch.tensor(
                [
                    [1.0, 9.0],
                    [10.0, 3.0],
                    [9.0, 5.0],
                ],
                dtype=dtype,
                device=self.device,
            )
            Y = torch.stack([Y1, Y2], dim=0)
            expected_padded_pareto = torch.stack(
                [
                    Y1,
                    torch.cat([Y2[1:], Y2[-1:]], dim=0),
                ],
                dim=0,
            )
            padded_pareto = _pad_batch_pareto_frontier(
                Y=Y, ref_point=ref_point, is_pareto=True
            )
            self.assertTrue(torch.equal(padded_pareto, expected_padded_pareto))

        # test multiple batch dims
        with self.assertRaises(UnsupportedError):
            _pad_batch_pareto_frontier(
                Y=Y.unsqueeze(0), ref_point=ref_point, is_pareto=False
            )
