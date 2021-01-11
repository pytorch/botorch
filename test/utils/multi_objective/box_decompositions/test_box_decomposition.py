#! /usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from itertools import product
from unittest import mock

import torch
from botorch.exceptions.errors import BotorchError
from botorch.utils.multi_objective.box_decompositions.box_decomposition import (
    BoxDecomposition,
)
from botorch.utils.testing import BotorchTestCase


class DummyBoxDecomposition(BoxDecomposition):
    def partition_space_2d(self):
        pass

    def _partition_space(self):
        pass

    def compute_hypervolume(self):
        pass

    def get_hypercell_bounds(self):
        pass


class TestBoxDecomposition(BotorchTestCase):
    def test_box_decomposition(self):
        with self.assertRaises(TypeError):
            BoxDecomposition()
        ref_point_raw = torch.zeros(3, device=self.device)
        Y_raw = torch.tensor(
            [
                [1.0, 2.0, 1.0],
                [1.0, 1.0, 1.0],
                [2.0, 0.5, 1.0],
            ],
            device=self.device,
        )
        pareto_Y_raw = torch.tensor(
            [
                [1.0, 2.0, 1.0],
                [2.0, 0.5, 1.0],
            ],
            device=self.device,
        )
        for dtype, m, sort in product(
            (torch.float, torch.double), (2, 3), (True, False)
        ):
            with mock.patch.object(
                DummyBoxDecomposition,
                "partition_space_2d" if m == 2 else "partition_space",
            ) as mock_partition_space:

                ref_point = ref_point_raw[:m].to(dtype=dtype)
                Y = Y_raw[:, :m].to(dtype=dtype)
                pareto_Y = pareto_Y_raw[:, :m].to(dtype=dtype)
                bd = DummyBoxDecomposition(ref_point=ref_point, sort=sort)

                # test pareto_Y before it is initialized
                with self.assertRaises(BotorchError):
                    bd.pareto_Y
                bd = DummyBoxDecomposition(ref_point=ref_point, sort=sort, Y=Y)

                mock_partition_space.assert_called_once()
                # test attributes
                expected_pareto_Y = (
                    pareto_Y[torch.argsort(-pareto_Y[:, 0])] if sort else pareto_Y
                )
                self.assertTrue(torch.equal(bd.pareto_Y, expected_pareto_Y))
                self.assertTrue(torch.equal(bd.Y, Y))
                self.assertTrue(torch.equal(bd._neg_Y, -Y))
                self.assertTrue(torch.equal(bd._neg_pareto_Y, -expected_pareto_Y))
                self.assertTrue(torch.equal(bd.ref_point, ref_point))
                self.assertTrue(torch.equal(bd._neg_ref_point, -ref_point))
                self.assertEqual(bd.num_outcomes, m)

                # test empty Y
                bd = DummyBoxDecomposition(ref_point=ref_point, sort=sort, Y=Y[:0])
                self.assertTrue(torch.equal(bd.pareto_Y, expected_pareto_Y[:0]))

                # test batch mode
                if m == 2:
                    batch_Y = torch.stack([Y, Y + 1], dim=0)
                    bd = DummyBoxDecomposition(
                        ref_point=ref_point, sort=sort, Y=batch_Y
                    )
                    batch_expected_pareto_Y = torch.stack(
                        [expected_pareto_Y, expected_pareto_Y + 1], dim=0
                    )
                    self.assertTrue(torch.equal(bd.pareto_Y, batch_expected_pareto_Y))
                    self.assertTrue(torch.equal(bd.Y, batch_Y))
                    self.assertTrue(torch.equal(bd.ref_point, ref_point))
                    # test batch ref point
                    batch_ref_point = torch.stack([ref_point, ref_point + 1], dim=0)
                    bd = DummyBoxDecomposition(
                        ref_point=batch_ref_point, sort=sort, Y=batch_Y
                    )
                    self.assertTrue(torch.equal(bd.ref_point, batch_ref_point))
                    # test multiple batch dims
                    with self.assertRaises(NotImplementedError):
                        DummyBoxDecomposition(
                            ref_point=ref_point,
                            sort=sort,
                            Y=batch_Y.unsqueeze(0),
                        )
                    # test empty Y
                    bd = DummyBoxDecomposition(
                        ref_point=ref_point, sort=sort, Y=batch_Y[:, :0]
                    )
                    self.assertTrue(
                        torch.equal(bd.pareto_Y, batch_expected_pareto_Y[:, :0])
                    )

                    # test padded pareto frontiers with different numbers of
                    # points
                    batch_Y[1, 1] = batch_Y[1, 0] - 1
                    batch_Y[1, 2] = batch_Y[1, 0] - 2
                    bd = DummyBoxDecomposition(
                        ref_point=ref_point, sort=sort, Y=batch_Y
                    )
                    batch_expected_pareto_Y = torch.stack(
                        [
                            expected_pareto_Y,
                            batch_Y[1, :1].expand(expected_pareto_Y.shape),
                        ],
                        dim=0,
                    )
                    self.assertTrue(torch.equal(bd.pareto_Y, batch_expected_pareto_Y))
                    self.assertTrue(torch.equal(bd.Y, batch_Y))

                else:
                    with self.assertRaises(NotImplementedError):
                        DummyBoxDecomposition(
                            ref_point=ref_point, sort=sort, Y=Y.unsqueeze(0)
                        )
