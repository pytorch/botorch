#! /usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.testing import BotorchTestCase


class TestPareto(BotorchTestCase):
    def test_is_non_dominated(self) -> None:
        tkwargs = {"device": self.device}
        Y = torch.tensor(
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
            ]
        )
        expected_nondom_Y = torch.tensor([[10.0, 3.0], [5.0, 5.0], [8.5, 3.5]])
        Yb = Y.clone()
        Yb[1] = 0
        expected_nondom_Yb = torch.tensor([[5.0, 5.0], [8.5, 3.5], [9.0, 1.0]])
        Y3 = torch.tensor(
            [
                [4.0, 2.0, 3.0],
                [2.0, 4.0, 1.0],
                [3.0, 5.0, 1.0],
                [2.0, 4.0, 2.0],
                [2.0, 4.0, 2.0],
                [1.0, 3.0, 4.0],
                [1.0, 2.0, 4.0],
                [1.0, 2.0, 6.0],
            ]
        )
        Y3b = Y3.clone()
        Y3b[0] = 0
        expected_nondom_Y3 = torch.tensor(
            [
                [4.0, 2.0, 3.0],
                [3.0, 5.0, 1.0],
                [2.0, 4.0, 2.0],
                [1.0, 3.0, 4.0],
                [1.0, 2.0, 6.0],
            ]
        )
        expected_nondom_Y3b = expected_nondom_Y3[1:]
        for dtype in (torch.float, torch.double):
            tkwargs["dtype"] = dtype
            Y = Y.to(**tkwargs)
            expected_nondom_Y = expected_nondom_Y.to(**tkwargs)
            Yb = Yb.to(**tkwargs)
            expected_nondom_Yb = expected_nondom_Yb.to(**tkwargs)
            Y3 = Y3.to(**tkwargs)
            expected_nondom_Y3 = expected_nondom_Y3.to(**tkwargs)
            Y3b = Y3b.to(**tkwargs)
            expected_nondom_Y3b = expected_nondom_Y3b.to(**tkwargs)

            # test 2d
            nondom_Y = Y[is_non_dominated(Y)]
            self.assertTrue(torch.equal(expected_nondom_Y, nondom_Y))
            # test deduplicate=False
            expected_nondom_Y_no_dedup = torch.cat(
                [expected_nondom_Y, expected_nondom_Y[-1:]], dim=0
            )
            nondom_Y = Y[is_non_dominated(Y, deduplicate=False)]
            self.assertTrue(torch.equal(expected_nondom_Y_no_dedup, nondom_Y))

            # test batch
            batch_Y = torch.stack([Y, Yb], dim=0)
            nondom_mask = is_non_dominated(batch_Y)
            self.assertTrue(torch.equal(batch_Y[0][nondom_mask[0]], expected_nondom_Y))
            self.assertTrue(torch.equal(batch_Y[1][nondom_mask[1]], expected_nondom_Yb))
            # test deduplicate=False
            expected_nondom_Yb_no_dedup = torch.cat(
                [expected_nondom_Yb[:-1], expected_nondom_Yb[-2:]], dim=0
            )
            nondom_mask = is_non_dominated(batch_Y, deduplicate=False)
            self.assertTrue(
                torch.equal(batch_Y[0][nondom_mask[0]], expected_nondom_Y_no_dedup)
            )
            self.assertTrue(
                torch.equal(batch_Y[1][nondom_mask[1]], expected_nondom_Yb_no_dedup)
            )

            # test 3d
            nondom_Y3 = Y3[is_non_dominated(Y3)]
            self.assertTrue(torch.equal(expected_nondom_Y3, nondom_Y3))
            # test deduplicate=False
            expected_nondom_Y3_no_dedup = torch.cat(
                [expected_nondom_Y3[:3], expected_nondom_Y3[2:]], dim=0
            )
            nondom_Y3 = Y3[is_non_dominated(Y3, deduplicate=False)]
            self.assertTrue(torch.equal(expected_nondom_Y3_no_dedup, nondom_Y3))
            # test batch
            batch_Y3 = torch.stack([Y3, Y3b], dim=0)
            nondom_mask3 = is_non_dominated(batch_Y3)
            self.assertTrue(
                torch.equal(batch_Y3[0][nondom_mask3[0]], expected_nondom_Y3)
            )
            self.assertTrue(
                torch.equal(batch_Y3[1][nondom_mask3[1]], expected_nondom_Y3b)
            )
            # test deduplicate=False
            nondom_mask3 = is_non_dominated(batch_Y3, deduplicate=False)
            self.assertTrue(
                torch.equal(batch_Y3[0][nondom_mask3[0]], expected_nondom_Y3_no_dedup)
            )
            expected_nondom_Y3b_no_dedup = torch.cat(
                [expected_nondom_Y3b[:2], expected_nondom_Y3b[1:]], dim=0
            )
            self.assertTrue(
                torch.equal(batch_Y3[1][nondom_mask3[1]], expected_nondom_Y3b_no_dedup)
            )
