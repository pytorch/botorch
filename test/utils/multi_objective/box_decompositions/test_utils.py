#! /usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch
from botorch.exceptions.errors import BotorchTensorDimensionError
from botorch.utils.multi_objective.box_decompositions.utils import _expand_ref_point
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
