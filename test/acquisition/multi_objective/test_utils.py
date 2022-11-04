#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from contextlib import ExitStack
from itertools import product
from unittest import mock

import torch
from botorch import settings
from botorch.acquisition.multi_objective.objective import (
    MCMultiOutputObjective,
    UnstandardizeMCMultiOutputObjective,
)
from botorch.acquisition.multi_objective.utils import (
    compute_sample_box_decomposition,
    get_default_partitioning_alpha,
    prune_inferior_points_multi_objective,
)
from botorch.exceptions.errors import UnsupportedError
from botorch.exceptions.warnings import SamplingWarning
from botorch.utils.testing import BotorchTestCase, MockModel, MockPosterior
from torch import Tensor


class TestUtils(BotorchTestCase):
    def test_get_default_partitioning_alpha(self):
        for m in range(2, 7):
            expected_val = 0.0 if m < 5 else 10 ** (-8 + m)
            self.assertEqual(
                expected_val, get_default_partitioning_alpha(num_objectives=m)
            )
        # In `BotorchTestCase.setUp` warnings are filtered, so here we
        # remove the filter to ensure a warning is issued as expected.
        warnings.resetwarnings()
        with warnings.catch_warnings(record=True) as ws:
            self.assertEqual(0.1, get_default_partitioning_alpha(num_objectives=7))
        self.assertEqual(len(ws), 1)


class DummyMCMultiOutputObjective(MCMultiOutputObjective):
    def forward(self, samples: Tensor) -> Tensor:
        return samples


class TestMultiObjectiveUtils(BotorchTestCase):
    def setUp(self):
        super().setUp()
        self.model = mock.MagicMock()
        self.objective = DummyMCMultiOutputObjective()
        self.X_observed = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
        self.X_pending = torch.tensor([[1.0, 3.0, 4.0]])
        self.mc_samples = 250
        self.qmc = True
        self.ref_point = [0.0, 0.0]
        self.Y = torch.tensor([[1.0, 2.0]])
        self.seed = 1

    def test_prune_inferior_points_multi_objective(self):
        tkwargs = {"device": self.device}
        for dtype in (torch.float, torch.double):
            tkwargs["dtype"] = dtype
            X = torch.rand(3, 2, **tkwargs)
            ref_point = torch.tensor([0.25, 0.25], **tkwargs)
            # the event shape is `q x m` = 3 x 2
            samples = torch.tensor([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0]], **tkwargs)
            mm = MockModel(MockPosterior(samples=samples))
            # test that a batched X raises errors
            with self.assertRaises(UnsupportedError):
                prune_inferior_points_multi_objective(
                    model=mm, X=X.expand(2, 3, 2), ref_point=ref_point
                )
            # test that a batched model raises errors (event shape is `q x m` = 3 x m)
            mm2 = MockModel(MockPosterior(samples=samples.expand(2, 3, 2)))
            with self.assertRaises(UnsupportedError):
                prune_inferior_points_multi_objective(
                    model=mm2, X=X, ref_point=ref_point
                )
            # test that invalid max_frac is checked properly
            with self.assertRaises(ValueError):
                prune_inferior_points_multi_objective(
                    model=mm, X=X, max_frac=1.1, ref_point=ref_point
                )
            # test basic behaviour
            X_pruned = prune_inferior_points_multi_objective(
                model=mm, X=X, ref_point=ref_point
            )
            self.assertTrue(torch.equal(X_pruned, X[[-1]]))
            # test unstd objective
            unstd_obj = UnstandardizeMCMultiOutputObjective(
                Y_mean=samples.mean(dim=0), Y_std=samples.std(dim=0), outcomes=[0, 1]
            )
            X_pruned = prune_inferior_points_multi_objective(
                model=mm, X=X, ref_point=ref_point, objective=unstd_obj
            )
            self.assertTrue(torch.equal(X_pruned, X[[-1]]))
            # test constraints
            samples_constrained = torch.tensor(
                [[1.0, 2.0, -1.0], [2.0, 1.0, -1.0], [3.0, 4.0, 1.0]], **tkwargs
            )
            mm_constrained = MockModel(MockPosterior(samples=samples_constrained))
            X_pruned = prune_inferior_points_multi_objective(
                model=mm_constrained,
                X=X,
                ref_point=ref_point,
                objective=unstd_obj,
                constraints=[lambda Y: Y[..., -1]],
            )
            self.assertTrue(torch.equal(X_pruned, X[:2]))

            # test non-repeated samples (requires mocking out MockPosterior's rsample)
            samples = torch.tensor(
                [[[3.0], [0.0], [0.0]], [[0.0], [2.0], [0.0]], [[0.0], [0.0], [1.0]]],
                device=self.device,
                dtype=dtype,
            )
            with mock.patch.object(MockPosterior, "rsample", return_value=samples):
                mm = MockModel(MockPosterior(samples=samples))
                X_pruned = prune_inferior_points_multi_objective(
                    model=mm, X=X, ref_point=ref_point
                )
            self.assertTrue(torch.equal(X_pruned, X))
            # test max_frac limiting
            with mock.patch.object(MockPosterior, "rsample", return_value=samples):
                mm = MockModel(MockPosterior(samples=samples))
                X_pruned = prune_inferior_points_multi_objective(
                    model=mm, X=X, ref_point=ref_point, max_frac=2 / 3
                )
            if self.device.type == "cuda":
                # sorting has different order on cuda
                self.assertTrue(
                    torch.equal(X_pruned, X[[2, 1]]) or torch.equal(X_pruned, X[[1, 2]])
                )
            else:
                self.assertTrue(torch.equal(X_pruned, X[:2]))
            # test that zero-probability is in fact pruned
            samples[2, 0, 0] = 10
            with mock.patch.object(MockPosterior, "rsample", return_value=samples):
                mm = MockModel(MockPosterior(samples=samples))
                X_pruned = prune_inferior_points_multi_objective(
                    model=mm, X=X, ref_point=ref_point
                )
            self.assertTrue(torch.equal(X_pruned, X[:2]))
            # test high-dim sampling
            with ExitStack() as es:
                mock_event_shape = es.enter_context(
                    mock.patch(
                        "botorch.utils.testing.MockPosterior.event_shape",
                        new_callable=mock.PropertyMock,
                    )
                )
                mock_event_shape.return_value = torch.Size(
                    [1, 1, torch.quasirandom.SobolEngine.MAXDIM + 1]
                )
                es.enter_context(
                    mock.patch.object(MockPosterior, "rsample", return_value=samples)
                )
                mm = MockModel(MockPosterior(samples=samples))
                with warnings.catch_warnings(record=True) as ws, settings.debug(True):
                    prune_inferior_points_multi_objective(
                        model=mm, X=X, ref_point=ref_point
                    )
                    self.assertTrue(issubclass(ws[-1].category, SamplingWarning))

            # test marginalize_dim and constraints
            samples = torch.tensor([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0]], **tkwargs)
            samples = samples.unsqueeze(-3).expand(
                *samples.shape[:-2],
                2,
                *samples.shape[-2:],
            )
            mm = MockModel(MockPosterior(samples=samples))
            X_pruned = prune_inferior_points_multi_objective(
                model=mm,
                X=X,
                ref_point=ref_point,
                objective=unstd_obj,
                constraints=[lambda Y: Y[..., -1] - 3.0],
                marginalize_dim=-3,
            )
            self.assertTrue(torch.equal(X_pruned, X[:2]))

    def test_compute_sample_box_decomposition(self):
        tkwargs = {"device": self.device}
        for dtype, maximize in product((torch.float, torch.double), (True, False)):
            tkwargs["dtype"] = dtype

            # test error when inputting incorrect Pareto front
            X = torch.rand(4, 3, 2, 1, **tkwargs)
            with self.assertRaises(UnsupportedError):
                compute_sample_box_decomposition(pareto_fronts=X, maximize=maximize)

            # test single and multi-objective setting
            for num_objectives in (1, 5):
                X = torch.rand(4, 3, num_objectives, **tkwargs)
                bd1 = compute_sample_box_decomposition(
                    pareto_fronts=X, maximize=maximize
                )

                # assess shape
                self.assertTrue(bd1.ndim == 4)
                self.assertTrue(bd1.shape[-1] == num_objectives)
                self.assertTrue(bd1.shape[-3] == 2)
                if num_objectives == 1:
                    self.assertTrue(bd1.shape[-2] == 1)

                # assess whether upper bound is greater than lower bound
                self.assertTrue(torch.all(bd1[:, 1, ...] - bd1[:, 0, ...] >= 0))

                # test constrained setting
                num_constraints = 7
                bd2 = compute_sample_box_decomposition(
                    pareto_fronts=X,
                    maximize=maximize,
                    num_constraints=num_constraints,
                )

                # assess shape
                self.assertTrue(bd2.ndim == 4)
                self.assertTrue(bd2.shape[-1] == num_objectives + num_constraints)
                self.assertTrue(bd2.shape[-2] == bd1.shape[-2] + 1)
                self.assertTrue(bd2.shape[-3] == 2)

                # assess whether upper bound is greater than lower bound
                self.assertTrue(torch.all(bd2[:, 1, ...] - bd2[:, 0, ...] >= 0))

                # the constraint padding should not change the box-decomposition
                # if the box-decomposition procedure is not random
                self.assertTrue(torch.equal(bd1, bd2[..., 0:-1, 0:num_objectives]))

                # test with a specified optimum
                opt_X = 2.0 if maximize else -3.0

                X[:, 0, :] = opt_X
                bd3 = compute_sample_box_decomposition(
                    pareto_fronts=X, maximize=maximize
                )

                # check optimum
                if maximize:
                    self.assertTrue(torch.all(bd3[:, 1, ...] == opt_X))
                else:
                    self.assertTrue(torch.all(bd3[:, 0, ...] == opt_X))
