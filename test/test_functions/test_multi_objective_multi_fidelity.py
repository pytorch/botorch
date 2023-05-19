# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from botorch.test_functions.multi_objective_multi_fidelity import (
    MOMFBraninCurrin,
    MOMFPark,
)
from botorch.utils.testing import (
    BaseTestProblemTestCaseMixIn,
    BotorchTestCase,
    MultiObjectiveTestProblemTestCaseMixin,
)


class TestMOMFBraninCurrin(
    BotorchTestCase,
    BaseTestProblemTestCaseMixIn,
    MultiObjectiveTestProblemTestCaseMixin,
):
    functions = [MOMFBraninCurrin()]
    bounds = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]

    def test_init(self):
        for f in self.functions:
            self.assertEqual(f.num_objectives, 2)
            self.assertEqual(f.dim, 3)
            self.assertTrue(
                torch.equal(f.bounds, torch.tensor(self.bounds).to(f.bounds))
            )


class TestMOMFPark(
    BotorchTestCase,
    BaseTestProblemTestCaseMixIn,
    MultiObjectiveTestProblemTestCaseMixin,
):
    functions = [MOMFPark()]
    bounds = [[0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0]]

    def test_init(self):
        for f in self.functions:
            self.assertEqual(f.num_objectives, 2)
            self.assertEqual(f.dim, 5)
            self.assertTrue(
                torch.equal(f.bounds, torch.tensor(self.bounds).to(f.bounds))
            )
