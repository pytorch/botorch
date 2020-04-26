#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch
from botorch.generation.utils import _flip_sub_unique
from botorch.utils.testing import BotorchTestCase


class TestGenerationUtils(BotorchTestCase):
    def test_flip_sub_unique(self):
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            x = torch.tensor([0.69, 0.75, 0.69, 0.21, 0.86, 0.21], **tkwargs)
            y = _flip_sub_unique(x=x, k=1)
            y_exp = torch.tensor([0.21], **tkwargs)
            self.assertTrue(torch.allclose(y, y_exp))
            y = _flip_sub_unique(x=x, k=3)
            y_exp = torch.tensor([0.21, 0.86, 0.69], **tkwargs)
            self.assertTrue(torch.allclose(y, y_exp))
            y = _flip_sub_unique(x=x, k=10)
            y_exp = torch.tensor([0.21, 0.86, 0.69, 0.75], **tkwargs)
            self.assertTrue(torch.allclose(y, y_exp))
        # long dtype
        tkwargs["dtype"] = torch.long
        x = torch.tensor([1, 6, 4, 3, 6, 3], **tkwargs)
        y = _flip_sub_unique(x=x, k=1)
        y_exp = torch.tensor([3], **tkwargs)
        self.assertTrue(torch.allclose(y, y_exp))
        y = _flip_sub_unique(x=x, k=3)
        y_exp = torch.tensor([3, 6, 4], **tkwargs)
        self.assertTrue(torch.allclose(y, y_exp))
        y = _flip_sub_unique(x=x, k=4)
        y_exp = torch.tensor([3, 6, 4, 1], **tkwargs)
        self.assertTrue(torch.allclose(y, y_exp))
        y = _flip_sub_unique(x=x, k=10)
        self.assertTrue(torch.allclose(y, y_exp))
