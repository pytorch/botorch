#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from botorch.utils import get_outcome_constraint_transforms
from botorch.utils.constraints import (
    get_monotonicity_constraints,
    LogTransformedInterval,
)
from botorch.utils.testing import BotorchTestCase


class TestConstraintUtils(BotorchTestCase):
    def setUp(self):
        super().setUp()
        self.A = torch.tensor([[-1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
        self.b = torch.tensor([[-0.5], [1.0]])
        self.Ys = torch.tensor([[0.75, 1.0, 0.5], [0.25, 1.5, 1.0]]).unsqueeze(0)
        self.results = torch.tensor([[-0.25, 0.5], [0.25, 1.5]]).view(1, 2, 2)

    def test_get_outcome_constraint_transforms(self):
        # test None
        self.assertIsNone(get_outcome_constraint_transforms(None))

        # test basic evaluation
        for dtype in (torch.float, torch.double):
            tkwargs = {"dtype": dtype, "device": self.device}
            A = self.A.to(**tkwargs)
            b = self.b.to(**tkwargs)
            Ys = self.Ys.to(**tkwargs)
            results = self.results.to(**tkwargs)
            ocs = get_outcome_constraint_transforms((A, b))
            self.assertEqual(len(ocs), 2)
            for i in (0, 1):
                for j in (0, 1):
                    self.assertTrue(torch.equal(ocs[j](Ys[:, i]), results[:, i, j]))

            # test broadcasted evaluation
            k, t = 3, 4
            mc_samples, b, q = 6, 4, 5
            A_ = torch.randn(k, t, **tkwargs)
            b_ = torch.randn(k, 1, **tkwargs)
            Y = torch.randn(mc_samples, b, q, t, **tkwargs)
            ocs = get_outcome_constraint_transforms((A_, b_))
            self.assertEqual(len(ocs), k)
            self.assertEqual(ocs[0](Y).shape, torch.Size([mc_samples, b, q]))

    def test_get_monotonicity_constraints(self):
        for dtype in (torch.float, torch.double):
            tkwargs = {"dtype": dtype, "device": self.device}
            for d in (3, 17):
                with self.subTest(dtype=dtype, d=d):
                    A, b = get_monotonicity_constraints(d, **tkwargs)
                    self.assertEqual(A.shape, (d - 1, d))
                    self.assertEqual(A.dtype, dtype)
                    self.assertEqual(A.device.type, self.device.type)

                    self.assertEqual(b.shape, (d - 1, 1))
                    self.assertEqual(b.dtype, dtype)
                    self.assertEqual(b.device.type, self.device.type)

                    unique_vals = torch.tensor([-1, 0, 1], **tkwargs)
                    self.assertAllClose(A.unique(), unique_vals)
                    self.assertAllClose(b, torch.zeros_like(b))
                    self.assertTrue(
                        torch.equal(A.sum(dim=-1), torch.zeros(d - 1, **tkwargs))
                    )

                    n_test = 3
                    X_test = torch.randn(d, n_test, **tkwargs)

                    X_diff_true = -X_test.diff(dim=0)  # x[i] - x[i+1] < 0
                    X_diff = A @ X_test
                    self.assertAllClose(X_diff, X_diff_true)

                    is_monotonic_true = (X_diff_true < 0).all(dim=0)
                    is_monotonic = (X_diff < b).all(dim=0)
                    self.assertAllClose(is_monotonic, is_monotonic_true)

                    Ad, bd = get_monotonicity_constraints(d, descending=True, **tkwargs)
                    self.assertAllClose(Ad, -A)
                    self.assertAllClose(bd, b)

    def test_log_transformed_interval(self):
        constraint = LogTransformedInterval(
            lower_bound=0.1, upper_bound=0.2, initial_value=0.15
        )
        x = torch.tensor(0.1, device=self.device)
        self.assertAllClose(constraint.transform(x), x.exp())
        self.assertAllClose(constraint.inverse_transform(constraint.transform(x)), x)
        with self.assertRaisesRegex(
            RuntimeError, "Cannot make an Interval directly with non-finite bounds"
        ):
            constraint = LogTransformedInterval(
                lower_bound=-torch.inf, upper_bound=torch.inf, initial_value=0.15
            )
