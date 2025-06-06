#! /usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch
from botorch.exceptions.errors import BotorchTensorDimensionError, UnsupportedError
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.utils.testing import BotorchTestCase
from botorch.utils.transforms import normalize


class TestGetChebyshevScalarization(BotorchTestCase):
    def test_get_chebyshev_scalarization(self):
        torch.manual_seed(1234)
        tkwargs = {"device": self.device}
        Y_train = torch.rand(4, 2, **tkwargs)
        neg_Y_train = -Y_train
        neg_Y_bounds = torch.stack(
            [
                neg_Y_train.min(dim=-2, keepdim=True).values,
                neg_Y_train.max(dim=-2, keepdim=True).values,
            ],
            dim=0,
        )
        for dtype in (torch.float, torch.double):
            for batch_shape in (torch.Size([]), torch.Size([3])):
                tkwargs["dtype"] = dtype
                Y_test = torch.rand(batch_shape + torch.Size([5, 2]), **tkwargs)
                neg_Y_test = -Y_test
                Y_train = Y_train.to(**tkwargs)
                neg_Y_bounds = neg_Y_bounds.to(**tkwargs)
                normalized_neg_Y_test = normalize(neg_Y_test, neg_Y_bounds)
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
                expected_Y_transformed = -(
                    normalized_neg_Y_test.max(dim=-1).values
                    + 0.05 * normalized_neg_Y_test.sum(dim=-1)
                )
                self.assertTrue(torch.equal(Y_transformed, expected_Y_transformed))
                # check that using negative objectives and negative weights
                # yields an equivalent scalarized outcome
                objective_transform2 = get_chebyshev_scalarization(
                    weights=-weights, Y=-Y_train
                )
                Y_transformed2 = objective_transform2(-Y_test)
                self.assertAllClose(Y_transformed, Y_transformed2)
                # test different alpha
                objective_transform = get_chebyshev_scalarization(
                    weights=weights, Y=Y_train, alpha=1.0
                )
                Y_transformed = objective_transform(Y_test)
                expected_Y_transformed = -(
                    normalized_neg_Y_test.max(dim=-1).values
                    + normalized_neg_Y_test.sum(dim=-1)
                )
                self.assertTrue(torch.equal(Y_transformed, expected_Y_transformed))
                # Test different weights
                weights = torch.tensor([0.3, 0.7], **tkwargs)
                objective_transform = get_chebyshev_scalarization(
                    weights=weights, Y=Y_train
                )
                Y_transformed = objective_transform(Y_test)
                expected_Y_transformed = -(
                    (weights * normalized_neg_Y_test).max(dim=-1).values
                    + 0.05 * (weights * normalized_neg_Y_test).sum(dim=-1)
                )
                self.assertTrue(torch.equal(Y_transformed, expected_Y_transformed))
                # test that when minimizing an objective (i.e. with a negative weight),
                # normalized Y values are shifted from [0,1] to [-1,0]
                weights = torch.tensor([0.3, -0.7], **tkwargs)
                objective_transform = get_chebyshev_scalarization(
                    weights=weights, Y=Y_train
                )
                Y_transformed = objective_transform(Y_test)
                normalized_neg_Y_test[..., -1] = normalized_neg_Y_test[..., -1] - 1
                expected_Y_transformed = -(
                    (weights * normalized_neg_Y_test).max(dim=-1).values
                    + 0.05 * (weights * normalized_neg_Y_test).sum(dim=-1)
                )
                self.assertTrue(torch.equal(Y_transformed, expected_Y_transformed))
                # test that with no observations there is no normalization
                weights = torch.tensor([0.3, 0.7], **tkwargs)
                objective_transform = get_chebyshev_scalarization(
                    weights=weights, Y=Y_train[:0]
                )
                Y_transformed = objective_transform(Y_test)
                expected_Y_transformed = -(
                    (weights * neg_Y_test).max(dim=-1).values
                    + 0.05 * (weights * neg_Y_test).sum(dim=-1)
                )
                self.assertTrue(torch.equal(Y_transformed, expected_Y_transformed))
                # test that error is raised with negative weights and empty Y
                with self.assertRaises(UnsupportedError):
                    get_chebyshev_scalarization(weights=-weights, Y=Y_train[:0])
                # test that with one observation, we normalize by subtracting
                # neg_Y_train
                single_Y_train = Y_train[:1]
                objective_transform = get_chebyshev_scalarization(
                    weights=weights, Y=single_Y_train
                )
                Y_transformed = objective_transform(Y_test)
                normalized_neg_Y_test = neg_Y_test + single_Y_train
                expected_Y_transformed = -(
                    (weights * normalized_neg_Y_test).max(dim=-1).values
                    + 0.05 * (weights * normalized_neg_Y_test).sum(dim=-1)
                )
                self.assertAllClose(Y_transformed, expected_Y_transformed)

                # Test that it works when Y is constant in each dimension.
                objective_transform = get_chebyshev_scalarization(
                    weights=weights, Y=torch.zeros(2, 2, **tkwargs), alpha=0.0
                )
                Y_transformed = objective_transform(Y_test)
                self.assertFalse(Y_transformed.isnan().any())
                self.assertFalse(Y_transformed.isinf().any())
                expected_Y = -(-weights * Y_test).max(dim=-1).values
                self.assertAllClose(Y_transformed, expected_Y)
