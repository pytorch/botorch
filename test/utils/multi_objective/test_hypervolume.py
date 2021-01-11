#! /usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch
from botorch.exceptions.errors import BotorchTensorDimensionError
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.testing import BotorchTestCase


EPS = 1e-4

pareto_Y_5d = [
    [
        -0.42890000759972685,
        -0.1446377658556118,
        -0.10335085850913295,
        -0.49502106785623134,
        -0.7344368200145969,
    ],
    [
        -0.5124511265981003,
        -0.5332028064973291,
        -0.36775794432917486,
        -0.5261970836251835,
        -0.20238412378158688,
    ],
    [
        -0.5960106882406603,
        -0.32491865590163566,
        -0.5815435820797972,
        -0.08375675085018466,
        -0.44044408882261904,
    ],
    [
        -0.6135323874039154,
        -0.5658986040644925,
        -0.39684098121151284,
        -0.3798488823307603,
        -0.03960860698719982,
    ],
    [
        -0.3957157311550265,
        -0.4045394517331393,
        -0.07282417302694655,
        -0.5699496614967537,
        -0.5912790502720109,
    ],
    [
        -0.06392539039575441,
        -0.17204800894814581,
        -0.6620860391018546,
        -0.7241037454151875,
        -0.06024010111083461,
    ],
]


class TestHypervolume(BotorchTestCase):
    def test_hypervolume(self):
        tkwargs = {"device": self.device}
        for dtype in (torch.float, torch.double):
            tkwargs["dtype"] = dtype

            ref_point = torch.tensor([0.0, 0.0], **tkwargs)
            hv = Hypervolume(ref_point)

            # test ref point
            self.assertTrue(torch.equal(ref_point, hv.ref_point))
            self.assertTrue(torch.equal(-ref_point, hv._ref_point))

            # test dimension errors
            with self.assertRaises(BotorchTensorDimensionError):
                hv.compute(pareto_Y=torch.empty(2, **tkwargs))
            with self.assertRaises(BotorchTensorDimensionError):
                hv.compute(pareto_Y=torch.empty(1, 1, 2, **tkwargs))
            with self.assertRaises(BotorchTensorDimensionError):
                hv.compute(pareto_Y=torch.empty(1, 3, **tkwargs))

            # test no pareto points
            pareto_Y = (ref_point - 1).view(1, 2)
            volume = hv.compute(pareto_Y)
            self.assertEqual(volume, 0.0)

            # test 1-d
            hv = Hypervolume(ref_point[:1])
            volume = hv.compute(pareto_Y=torch.ones(1, 1, **tkwargs))
            self.assertEqual(volume, 1.0)

            # test m=2
            hv = Hypervolume(ref_point)
            pareto_Y = torch.tensor(
                [[8.5, 3.0], [8.5, 3.5], [5.0, 5.0], [9.0, 1.0], [4.0, 5.0]], **tkwargs
            )
            volume = hv.compute(pareto_Y=pareto_Y)
            self.assertTrue(abs(volume - 37.75) < EPS)

            # test nonzero reference point
            ref_point = torch.tensor([1.0, 0.5], **tkwargs)
            hv = Hypervolume(ref_point)
            volume = hv.compute(pareto_Y=pareto_Y)
            self.assertTrue(abs(volume - 28.75) < EPS)

            # test m=3
            # ref_point = torch.tensor([-1.1, -1.1, -1.1], **tkwargs)
            ref_point = torch.tensor([-2.1, -2.5, -2.3], **tkwargs)
            hv = Hypervolume(ref_point)
            pareto_Y = torch.tensor(
                [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], **tkwargs
            )
            volume = hv.compute(pareto_Y=pareto_Y)
            self.assertTrue(abs(volume - 11.075) < EPS)
            # self.assertTrue(abs(volume - 0.45980908291719647) < EPS)

            # test m=4
            ref_point = torch.tensor([-2.1, -2.5, -2.3, -2.0], **tkwargs)
            hv = Hypervolume(ref_point)
            pareto_Y = torch.tensor(
                [
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, 0.0, 0.0, -1.0],
                ],
                **tkwargs
            )
            volume = hv.compute(pareto_Y=pareto_Y)
            self.assertTrue(abs(volume - 23.15) < EPS)

            # test m=5
            # this pareto front is from DTLZ2 and covers several edge cases
            ref_point = torch.full(torch.Size([5]), -1.1, **tkwargs)
            hv = Hypervolume(ref_point)
            pareto_Y = torch.tensor(pareto_Y_5d, **tkwargs)
            volume = hv.compute(pareto_Y)
            self.assertTrue(abs(volume - 0.42127855991587) < EPS)
