#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import warnings

import torch
from botorch import settings
from botorch.acquisition.cost_aware import (
    CostAwareUtility,
    GenericCostAwareUtility,
    InverseCostWeightedUtility,
)
from botorch.exceptions.warnings import CostAwareWarning
from botorch.sampling import IIDNormalSampler
from botorch.utils.testing import BotorchTestCase, MockModel, MockPosterior


class TestCostAwareUtilities(BotorchTestCase):
    def test_abstract_raises(self):
        with self.assertRaises(TypeError):
            CostAwareUtility()

    def test_GenericCostAwareUtility(self):
        def cost(X, deltas, **kwargs):
            return deltas.mean(dim=-1) / X[..., 1].sum(dim=-1)

        for dtype in (torch.float, torch.double):
            u = GenericCostAwareUtility(cost)
            X = torch.rand(3, 2, device=self.device, dtype=dtype)
            deltas = torch.rand(5, 3, device=self.device, dtype=dtype)
            self.assertIsInstance(u, GenericCostAwareUtility)
            self.assertTrue(torch.equal(u(X, deltas), cost(X, deltas)))
            X = torch.rand(4, 3, 2, device=self.device, dtype=dtype)
            deltas = torch.rand(5, 4, 3, device=self.device, dtype=dtype)
            self.assertIsInstance(u, GenericCostAwareUtility)
            self.assertTrue(torch.equal(u(X, deltas), cost(X, deltas)))

    def test_InverseCostWeightedUtility(self):
        for batch_shape in ([], [2]):
            for dtype in (torch.float, torch.double):
                # the event shape is `batch_shape x q x t`
                mean = 1 + torch.rand(
                    *batch_shape, 2, 1, device=self.device, dtype=dtype
                )
                mm = MockModel(MockPosterior(mean=mean))

                X = torch.randn(*batch_shape, 3, 2, device=self.device, dtype=dtype)
                deltas = torch.rand(4, *batch_shape, device=self.device, dtype=dtype)

                # test that sampler is required if use_mean=False
                icwu = InverseCostWeightedUtility(mm, use_mean=False)
                with self.assertRaises(RuntimeError):
                    icwu(X, deltas)

                # check warning for negative cost
                mm = MockModel(MockPosterior(mean=mean.clamp_max(-1e-6)))
                icwu = InverseCostWeightedUtility(mm)
                with warnings.catch_warnings(record=True) as ws, settings.debug(True):
                    icwu(X, deltas)
                    self.assertTrue(
                        any(issubclass(w.category, CostAwareWarning) for w in ws)
                    )

                # basic test
                mm = MockModel(MockPosterior(mean=mean))
                icwu = InverseCostWeightedUtility(mm)
                ratios = icwu(X, deltas)
                self.assertTrue(
                    torch.equal(ratios, deltas / mean.squeeze(-1).sum(dim=-1))
                )

                # sampling test
                samples = 1 + torch.rand(  # event shape is q x m
                    *batch_shape, 3, 1, device=self.device, dtype=dtype
                )
                mm = MockModel(MockPosterior(samples=samples))
                icwu = InverseCostWeightedUtility(mm, use_mean=False)
                ratios = icwu(
                    X, deltas, sampler=IIDNormalSampler(sample_shape=torch.Size([4]))
                )
                self.assertTrue(
                    torch.equal(ratios, deltas / samples.squeeze(-1).sum(dim=-1))
                )

                # test min cost
                mm = MockModel(MockPosterior(mean=mean))
                icwu = InverseCostWeightedUtility(mm, min_cost=1.5)
                ratios = icwu(X, deltas)
                self.assertTrue(
                    torch.equal(
                        ratios, deltas / mean.clamp_min(1.5).squeeze(-1).sum(dim=-1)
                    )
                )
