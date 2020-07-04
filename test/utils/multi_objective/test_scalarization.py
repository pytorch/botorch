#! /usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch
from botorch.exceptions.errors import BotorchTensorDimensionError
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.utils.testing import BotorchTestCase
from botorch.utils.transforms import normalize


class TestGetChebyshevScalarization(BotorchTestCase):
    def test_get_chebyshev_scalarization(self):
        tkwargs = {"device": self.device}
        Y_train = torch.rand(4, 2, **tkwargs)
        Y_bounds = torch.stack(
            [
                Y_train.min(dim=-2, keepdim=True).values,
                Y_train.max(dim=-2, keepdim=True).values,
            ],
            dim=0,
        )
        for dtype in (torch.float, torch.double):
            for batch_shape in (torch.Size([]), torch.Size([3])):
                tkwargs["dtype"] = dtype
                Y_test = torch.rand(batch_shape + torch.Size([5, 2]), **tkwargs)
                Y_train = Y_train.to(**tkwargs)
                Y_bounds = Y_bounds.to(**tkwargs)
                normalized_Y_test = normalize(Y_test, Y_bounds)
                # test wrong shape
                with self.assertRaises(BotorchTensorDimensionError):
                    get_chebyshev_scalarization(
                        weights=torch.zeros(3, **tkwargs), Y=Y_train
                    )
                weights = torch.ones(2, **tkwargs)
                # test batch Y
                with self.assertRaises(NotImplementedError):
                    get_chebyshev_scalarization(weights=weights, Y=Y_train.unsqueeze(0))
                # basic test
                objective_transform = get_chebyshev_scalarization(
                    weights=weights, Y=Y_train
                )
                Y_transformed = objective_transform(Y_test)
                expected_Y_transformed = normalized_Y_test.min(
                    dim=-1
                ).values + 0.05 * normalized_Y_test.sum(dim=-1)
                self.assertTrue(torch.equal(Y_transformed, expected_Y_transformed))
                # test different alpha
                objective_transform = get_chebyshev_scalarization(
                    weights=weights, Y=Y_train, alpha=1.0
                )
                Y_transformed = objective_transform(Y_test)
                expected_Y_transformed = normalized_Y_test.min(
                    dim=-1
                ).values + normalized_Y_test.sum(dim=-1)
                self.assertTrue(torch.equal(Y_transformed, expected_Y_transformed))
                # Test different weights
                weights = torch.tensor([0.3, 0.7], **tkwargs)
                objective_transform = get_chebyshev_scalarization(
                    weights=weights, Y=Y_train
                )
                Y_transformed = objective_transform(Y_test)
                expected_Y_transformed = (weights * normalized_Y_test).min(
                    dim=-1
                ).values + 0.05 * (weights * normalized_Y_test).sum(dim=-1)
                self.assertTrue(torch.equal(Y_transformed, expected_Y_transformed))
