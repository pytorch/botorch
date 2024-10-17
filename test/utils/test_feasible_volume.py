#! /usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
from botorch.utils.feasible_volume import (
    estimate_feasible_volume,
    get_feasible_samples,
    get_outcome_feasibility_probability,
)
from botorch.utils.testing import BotorchTestCase, MockModel, MockPosterior


class TestFeasibleVolumeEstimates(BotorchTestCase):
    def test_feasible_samples(self):
        # -X[0]+X[1]>=1
        inequality_constraints = [(torch.tensor([0, 1]), torch.tensor([-1.0, 1.0]), 1)]
        box_samples = torch.tensor([[1.1, 2.0], [0.9, 2.1], [1.5, 2], [1.8, 2.2]])

        feasible_samples, p_linear = get_feasible_samples(
            samples=box_samples, inequality_constraints=inequality_constraints
        )

        feasible = box_samples[:, 1] - box_samples[:, 0] >= 1

        self.assertTrue(
            torch.all(torch.eq(feasible_samples, box_samples[feasible])).item()
        )
        self.assertEqual(p_linear, feasible.sum(0).float().item() / feasible.size(0))

    def test_outcome_feasibility_probability(self):
        for dtype in (torch.float, torch.double):
            samples = torch.zeros(1, 1, 1, device=self.device, dtype=dtype)
            mm = MockModel(MockPosterior(samples=samples))
            X = torch.zeros(1, 1, device=self.device, dtype=torch.double)

            for outcome_constraints in [
                [lambda y: y[..., 0] - 0.5],
                [lambda y: y[..., 0] + 1.0],
            ]:
                p_outcome = get_outcome_feasibility_probability(
                    model=mm,
                    X=X,
                    outcome_constraints=outcome_constraints,
                    nsample_outcome=2,
                )
                feasible = outcome_constraints[0](samples) <= 0
                self.assertEqual(p_outcome, feasible)

    def test_estimate_feasible_volume(self):
        for dtype in (torch.float, torch.double):
            for samples in (
                torch.zeros(1, 2, 1, device=self.device, dtype=dtype),
                torch.ones(1, 1, 1, device=self.device, dtype=dtype),
            ):
                mm = MockModel(MockPosterior(samples=samples))
                bounds = torch.ones((2, 1))
                outcome_constraints = [lambda y: y[..., 0] - 0.5]

                p_linear, p_outcome = estimate_feasible_volume(
                    bounds=bounds,
                    model=mm,
                    outcome_constraints=outcome_constraints,
                    nsample_feature=2,
                    nsample_outcome=1,
                    dtype=dtype,
                )

                self.assertEqual(p_linear, 1.0)
                self.assertEqual(p_outcome, 1.0 - samples[0, 0].item())

                p_linear, p_outcome = estimate_feasible_volume(
                    bounds=bounds,
                    model=mm,
                    outcome_constraints=None,
                    nsample_feature=2,
                    nsample_outcome=1,
                    dtype=dtype,
                )
                self.assertEqual(p_linear, 1.0)
                self.assertEqual(p_outcome, 1.0)
