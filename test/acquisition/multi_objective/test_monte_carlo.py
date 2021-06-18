#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from copy import deepcopy
from itertools import product
from unittest import mock

import torch
from botorch import settings
from botorch.acquisition.multi_objective.monte_carlo import (
    MultiObjectiveMCAcquisitionFunction,
    qExpectedHypervolumeImprovement,
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.objective import (
    IdentityMCMultiOutputObjective,
    MCMultiOutputObjective,
)
from botorch.acquisition.objective import IdentityMCObjective
from botorch.exceptions.errors import BotorchError, UnsupportedError
from botorch.exceptions.warnings import BotorchWarning
from botorch.sampling.samplers import IIDNormalSampler, SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
    NondominatedPartitioning,
)
from botorch.utils.testing import BotorchTestCase, MockModel, MockPosterior


class DummyMultiObjectiveMCAcquisitionFunction(MultiObjectiveMCAcquisitionFunction):
    def forward(self, X):
        pass


class DummyMCMultiOutputObjective(MCMultiOutputObjective):
    def forward(self, samples):
        pass


class TestMultiObjectiveMCAcquisitionFunction(BotorchTestCase):
    def test_abstract_raises(self):
        with self.assertRaises(TypeError):
            MultiObjectiveMCAcquisitionFunction()

    def test_init(self):
        mm = MockModel(MockPosterior(mean=torch.rand(2, 1)))
        # test default init
        acqf = DummyMultiObjectiveMCAcquisitionFunction(model=mm)
        self.assertIsInstance(acqf.objective, IdentityMCMultiOutputObjective)
        self.assertIsInstance(acqf.sampler, SobolQMCNormalSampler)
        self.assertEqual(acqf.sampler._sample_shape, torch.Size([128]))
        self.assertTrue(acqf.sampler.collapse_batch_dims, True)
        self.assertFalse(acqf.sampler.resample)
        self.assertIsNone(acqf.X_pending)
        # test custom init
        sampler = SobolQMCNormalSampler(
            num_samples=64, collapse_batch_dims=False, resample=True
        )
        objective = DummyMCMultiOutputObjective()
        X_pending = torch.rand(2, 1)
        acqf = DummyMultiObjectiveMCAcquisitionFunction(
            model=mm, sampler=sampler, objective=objective, X_pending=X_pending
        )
        self.assertEqual(acqf.objective, objective)
        self.assertEqual(acqf.sampler, sampler)
        self.assertTrue(torch.equal(acqf.X_pending, X_pending))
        # test unsupported objective
        with self.assertRaises(UnsupportedError):
            acqf = DummyMultiObjectiveMCAcquisitionFunction(
                model=mm, objective=IdentityMCObjective()
            )


class TestQExpectedHypervolumeImprovement(BotorchTestCase):
    def test_q_expected_hypervolume_improvement(self):
        tkwargs = {"device": self.device}
        for dtype in (torch.float, torch.double):
            tkwargs["dtype"] = dtype
            ref_point = [0.0, 0.0]
            t_ref_point = torch.tensor(ref_point, **tkwargs)
            pareto_Y = torch.tensor(
                [[4.0, 5.0], [5.0, 5.0], [8.5, 3.5], [8.5, 3.0], [9.0, 1.0]], **tkwargs
            )
            partitioning = NondominatedPartitioning(ref_point=t_ref_point)
            # the event shape is `b x q x m` = 1 x 1 x 2
            samples = torch.zeros(1, 1, 2, **tkwargs)
            mm = MockModel(MockPosterior(samples=samples))
            # test error if there is not pareto_Y initialized in partitioning
            with self.assertRaises(BotorchError):
                qExpectedHypervolumeImprovement(
                    model=mm, ref_point=ref_point, partitioning=partitioning
                )
            partitioning.update(Y=pareto_Y)
            # test error if ref point has wrong shape
            with self.assertRaises(ValueError):
                qExpectedHypervolumeImprovement(
                    model=mm, ref_point=ref_point[:1], partitioning=partitioning
                )

            X = torch.zeros(1, 1, **tkwargs)
            # basic test
            sampler = IIDNormalSampler(num_samples=1)
            acqf = qExpectedHypervolumeImprovement(
                model=mm,
                ref_point=ref_point,
                partitioning=partitioning,
                sampler=sampler,
            )
            res = acqf(X)
            self.assertEqual(res.item(), 0.0)
            # check ref point
            self.assertTrue(
                torch.equal(acqf.ref_point, torch.tensor(ref_point, **tkwargs))
            )
            # check cached indices
            self.assertTrue(hasattr(acqf, "q_subset_indices"))
            self.assertIn("q_choose_1", acqf.q_subset_indices)
            self.assertTrue(
                torch.equal(
                    acqf.q_subset_indices["q_choose_1"],
                    torch.tensor([[0]], device=self.device),
                )
            )

            # test q=2
            X2 = torch.zeros(2, 1, **tkwargs)
            samples2 = torch.zeros(1, 2, 2, **tkwargs)
            mm2 = MockModel(MockPosterior(samples=samples2))
            acqf.model = mm2
            res = acqf(X2)
            self.assertEqual(res.item(), 0.0)
            # check cached indices
            self.assertTrue(hasattr(acqf, "q_subset_indices"))
            self.assertIn("q_choose_1", acqf.q_subset_indices)
            self.assertTrue(
                torch.equal(
                    acqf.q_subset_indices["q_choose_1"],
                    torch.tensor([[0], [1]], device=self.device),
                )
            )
            self.assertIn("q_choose_2", acqf.q_subset_indices)
            self.assertTrue(
                torch.equal(
                    acqf.q_subset_indices["q_choose_2"],
                    torch.tensor([[0, 1]], device=self.device),
                )
            )
            self.assertNotIn("q_choose_3", acqf.q_subset_indices)
            # now back to 1 and sure all caches were cleared
            acqf.model = mm
            res = acqf(X)
            self.assertNotIn("q_choose_2", acqf.q_subset_indices)
            self.assertIn("q_choose_1", acqf.q_subset_indices)
            self.assertTrue(
                torch.equal(
                    acqf.q_subset_indices["q_choose_1"],
                    torch.tensor([[0]], device=self.device),
                )
            )

            X = torch.zeros(1, 1, **tkwargs)
            samples = torch.zeros(1, 1, 2, **tkwargs)
            mm = MockModel(MockPosterior(samples=samples))
            # basic test, no resample
            sampler = IIDNormalSampler(num_samples=2, seed=12345)
            acqf = qExpectedHypervolumeImprovement(
                model=mm,
                ref_point=ref_point,
                partitioning=partitioning,
                sampler=sampler,
            )
            res = acqf(X)
            self.assertEqual(res.item(), 0.0)
            self.assertEqual(acqf.sampler.base_samples.shape, torch.Size([2, 1, 1, 2]))
            bs = acqf.sampler.base_samples.clone()
            res = acqf(X)
            self.assertTrue(torch.equal(acqf.sampler.base_samples, bs))

            # basic test, qmc, no resample
            sampler = SobolQMCNormalSampler(num_samples=2)
            acqf = qExpectedHypervolumeImprovement(
                model=mm,
                ref_point=ref_point,
                partitioning=partitioning,
                sampler=sampler,
            )
            res = acqf(X)
            self.assertEqual(res.item(), 0.0)
            self.assertEqual(acqf.sampler.base_samples.shape, torch.Size([2, 1, 1, 2]))
            bs = acqf.sampler.base_samples.clone()
            acqf(X)
            self.assertTrue(torch.equal(acqf.sampler.base_samples, bs))

            # basic test, qmc, resample
            sampler = SobolQMCNormalSampler(num_samples=2, resample=True)
            acqf = qExpectedHypervolumeImprovement(
                model=mm,
                ref_point=ref_point,
                partitioning=partitioning,
                sampler=sampler,
            )
            res = acqf(X)
            self.assertEqual(res.item(), 0.0)
            self.assertEqual(acqf.sampler.base_samples.shape, torch.Size([2, 1, 1, 2]))
            bs = acqf.sampler.base_samples.clone()
            acqf(X)
            self.assertFalse(torch.equal(acqf.sampler.base_samples, bs))

            # basic test for X_pending and warning
            acqf.set_X_pending()
            self.assertIsNone(acqf.X_pending)
            acqf.set_X_pending(None)
            self.assertIsNone(acqf.X_pending)
            acqf.set_X_pending(X)
            self.assertEqual(acqf.X_pending, X)
            res = acqf(X)
            X2 = torch.zeros(1, 1, 1, requires_grad=True, **tkwargs)
            with warnings.catch_warnings(record=True) as ws, settings.debug(True):
                acqf.set_X_pending(X2)
                self.assertEqual(acqf.X_pending, X2)
                self.assertEqual(len(ws), 1)
                self.assertTrue(issubclass(ws[-1].category, BotorchWarning))

            # test objective
            acqf = qExpectedHypervolumeImprovement(
                model=mm,
                ref_point=ref_point,
                partitioning=partitioning,
                sampler=sampler,
                objective=IdentityMCMultiOutputObjective(),
            )
            res = acqf(X)
            self.assertEqual(res.item(), 0.0)

            # Test that the hypervolume improvement is correct for given sample
            # test q = 1
            X = torch.zeros(1, 1, **tkwargs)
            # basic test
            samples = torch.tensor([[[6.5, 4.5]]], **tkwargs)
            mm = MockModel(MockPosterior(samples=samples))
            sampler = IIDNormalSampler(num_samples=1)
            acqf = qExpectedHypervolumeImprovement(
                model=mm,
                ref_point=ref_point,
                partitioning=partitioning,
                sampler=sampler,
            )
            res = acqf(X)
            self.assertEqual(res.item(), 1.5)
            # test q = 1, does not contribute
            samples = torch.tensor([0.0, 1.0], **tkwargs).view(1, 1, 2)
            sampler = IIDNormalSampler(1)
            mm = MockModel(MockPosterior(samples=samples))
            acqf.model = mm
            res = acqf(X)
            self.assertEqual(res.item(), 0.0)

            # test q = 2, both points contribute
            X = torch.zeros(2, 1, **tkwargs)
            samples = torch.tensor([[6.5, 4.5], [7.0, 4.0]], **tkwargs).unsqueeze(0)
            mm = MockModel(MockPosterior(samples=samples))
            acqf.model = mm
            res = acqf(X)
            self.assertEqual(res.item(), 1.75)

            # test q = 2, only 1 point contributes
            samples = torch.tensor([[6.5, 4.5], [6.0, 4.0]], **tkwargs).unsqueeze(0)
            mm = MockModel(MockPosterior(samples=samples))
            acqf.model = mm
            res = acqf(X)
            self.assertEqual(res.item(), 1.5)

            # test q = 2, neither contributes
            samples = torch.tensor([[2.0, 2.0], [0.0, 0.1]], **tkwargs).unsqueeze(0)
            mm = MockModel(MockPosterior(samples=samples))
            acqf.model = mm
            res = acqf(X)
            self.assertEqual(res.item(), 0.0)

            # test q = 2, test point better than current best second objective
            samples = torch.tensor([[6.5, 4.5], [6.0, 6.0]], **tkwargs).unsqueeze(0)
            mm = MockModel(MockPosterior(samples=samples))
            acqf.model = mm
            res = acqf(X)
            self.assertEqual(res.item(), 8.0)

            # test q = 2, test point better than current-best first objective
            samples = torch.tensor([[6.5, 4.5], [9.0, 2.0]], **tkwargs).unsqueeze(0)
            mm = MockModel(MockPosterior(samples=samples))
            acqf = qExpectedHypervolumeImprovement(
                model=mm,
                ref_point=ref_point,
                partitioning=partitioning,
                sampler=sampler,
            )
            res = acqf(X)
            self.assertEqual(res.item(), 2.0)
            # test q = 3, all contribute
            X = torch.zeros(3, 1, **tkwargs)
            samples = torch.tensor(
                [[6.5, 4.5], [9.0, 2.0], [7.0, 4.0]], **tkwargs
            ).unsqueeze(0)
            mm = MockModel(MockPosterior(samples=samples))
            acqf = qExpectedHypervolumeImprovement(
                model=mm,
                ref_point=ref_point,
                partitioning=partitioning,
                sampler=sampler,
            )
            res = acqf(X)
            self.assertEqual(res.item(), 2.25)
            # test q = 3, not all contribute
            samples = torch.tensor(
                [[6.5, 4.5], [9.0, 2.0], [7.0, 5.0]], **tkwargs
            ).unsqueeze(0)
            mm = MockModel(MockPosterior(samples=samples))
            acqf = qExpectedHypervolumeImprovement(
                model=mm,
                ref_point=ref_point,
                partitioning=partitioning,
                sampler=sampler,
            )
            res = acqf(X)
            self.assertEqual(res.item(), 3.5)
            # test q = 3, none contribute
            samples = torch.tensor(
                [[0.0, 4.5], [1.0, 2.0], [3.0, 0.0]], **tkwargs
            ).unsqueeze(0)
            mm = MockModel(MockPosterior(samples=samples))
            acqf = qExpectedHypervolumeImprovement(
                model=mm,
                ref_point=ref_point,
                partitioning=partitioning,
                sampler=sampler,
            )
            res = acqf(X)
            self.assertEqual(res.item(), 0.0)

            # test m = 3, q=1
            pareto_Y = torch.tensor(
                [[4.0, 2.0, 3.0], [3.0, 5.0, 1.0], [2.0, 4.0, 2.0], [1.0, 3.0, 4.0]],
                **tkwargs,
            )
            ref_point = [-1.0] * 3
            t_ref_point = torch.tensor(ref_point, **tkwargs)
            partitioning = NondominatedPartitioning(ref_point=t_ref_point, Y=pareto_Y)
            samples = torch.tensor([[1.0, 2.0, 6.0]], **tkwargs).unsqueeze(0)
            mm = MockModel(MockPosterior(samples=samples))

            acqf = qExpectedHypervolumeImprovement(
                model=mm,
                ref_point=ref_point,
                partitioning=partitioning,
                sampler=sampler,
            )
            X = torch.zeros(1, 2, **tkwargs)
            res = acqf(X)
            self.assertEqual(res.item(), 12.0)

            # change reference point
            ref_point = [0.0] * 3
            t_ref_point = torch.tensor(ref_point, **tkwargs)
            partitioning = NondominatedPartitioning(ref_point=t_ref_point, Y=pareto_Y)
            acqf = qExpectedHypervolumeImprovement(
                model=mm,
                ref_point=ref_point,
                partitioning=partitioning,
                sampler=sampler,
            )
            res = acqf(X)
            self.assertEqual(res.item(), 4.0)

            # test m = 3, no contribution
            ref_point = [1.0] * 3
            t_ref_point = torch.tensor(ref_point, **tkwargs)
            partitioning = NondominatedPartitioning(ref_point=t_ref_point, Y=pareto_Y)
            acqf = qExpectedHypervolumeImprovement(
                model=mm,
                ref_point=ref_point,
                partitioning=partitioning,
                sampler=sampler,
            )
            res = acqf(X)
            self.assertEqual(res.item(), 0.0)

            # test m = 3, q = 2
            pareto_Y = torch.tensor(
                [[4.0, 2.0, 3.0], [3.0, 5.0, 1.0], [2.0, 4.0, 2.0]], **tkwargs
            )
            samples = torch.tensor(
                [[1.0, 2.0, 6.0], [1.0, 3.0, 4.0]], **tkwargs
            ).unsqueeze(0)
            mm = MockModel(MockPosterior(samples=samples))
            ref_point = [-1.0] * 3
            t_ref_point = torch.tensor(ref_point, **tkwargs)
            partitioning = NondominatedPartitioning(ref_point=t_ref_point, Y=pareto_Y)
            acqf = qExpectedHypervolumeImprovement(
                model=mm,
                ref_point=ref_point,
                partitioning=partitioning,
                sampler=sampler,
            )
            X = torch.zeros(2, 2, **tkwargs)
            res = acqf(X)
            self.assertEqual(res.item(), 22.0)

            # test batched model
            pareto_Y = torch.tensor(
                [[4.0, 2.0, 3.0], [3.0, 5.0, 1.0], [2.0, 4.0, 2.0]], **tkwargs
            )
            samples = torch.tensor(
                [[1.0, 2.0, 6.0], [1.0, 3.0, 4.0]], **tkwargs
            ).unsqueeze(0)
            samples = torch.stack([samples, samples + 1], dim=1)
            mm = MockModel(MockPosterior(samples=samples))
            ref_point = [-1.0] * 3
            t_ref_point = torch.tensor(ref_point, **tkwargs)
            partitioning = NondominatedPartitioning(ref_point=t_ref_point, Y=pareto_Y)
            acqf = qExpectedHypervolumeImprovement(
                model=mm,
                ref_point=ref_point,
                partitioning=partitioning,
                sampler=sampler,
            )
            X = torch.zeros(2, 2, **tkwargs)
            res = acqf(X)
            self.assertTrue(
                torch.equal(
                    res,
                    # batch_shape x model_batch_shape
                    torch.tensor([[22.0, 60.0]], **tkwargs),
                )
            )
            # test batched model with batched partitioning with multiple batch dims
            pareto_Y = torch.tensor(
                [[4.0, 5.0], [5.0, 5.0], [8.5, 3.5], [8.5, 3.0], [9.0, 1.0]], **tkwargs
            )
            pareto_Y = torch.stack(
                [
                    pareto_Y,
                    pareto_Y + 0.5,
                ],
                dim=0,
            )
            samples = torch.tensor([[6.5, 4.5], [7.0, 4.0]], **tkwargs).unsqueeze(0)
            samples = torch.stack([samples, samples + 1], dim=1)
            mm = MockModel(MockPosterior(samples=samples))
            ref_point = [-1.0] * 2
            t_ref_point = torch.tensor(ref_point, **tkwargs)
            partitioning = FastNondominatedPartitioning(
                ref_point=t_ref_point, Y=pareto_Y
            )
            cell_bounds = partitioning.get_hypercell_bounds().unsqueeze(1)
            with mock.patch.object(
                partitioning, "get_hypercell_bounds", return_value=cell_bounds
            ):
                acqf = qExpectedHypervolumeImprovement(
                    model=mm,
                    ref_point=ref_point,
                    partitioning=partitioning,
                    sampler=sampler,
                )
                # test multiple batch dims
                self.assertEqual(acqf.cell_lower_bounds.shape, torch.Size([1, 2, 4, 2]))
                self.assertEqual(acqf.cell_upper_bounds.shape, torch.Size([1, 2, 4, 2]))
            X = torch.zeros(2, 2, **tkwargs)
            res = acqf(X)
            self.assertTrue(
                torch.equal(
                    res,
                    # batch_shape x model_batch_shape
                    torch.tensor(
                        [[1.75, 3.5]], dtype=samples.dtype, device=samples.device
                    ),
                )
            )

    def test_constrained_q_expected_hypervolume_improvement(self):
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            ref_point = [0.0, 0.0]
            t_ref_point = torch.tensor(ref_point, **tkwargs)
            pareto_Y = torch.tensor(
                [[4.0, 5.0], [5.0, 5.0], [8.5, 3.5], [8.5, 3.0], [9.0, 1.0]], **tkwargs
            )
            partitioning = NondominatedPartitioning(ref_point=t_ref_point)
            partitioning.update(Y=pareto_Y)

            # test q=1
            # the event shape is `b x q x m` = 1 x 1 x 2
            samples = torch.tensor([[[6.5, 4.5]]], **tkwargs)
            mm = MockModel(MockPosterior(samples=samples))
            sampler = IIDNormalSampler(num_samples=1)
            X = torch.zeros(1, 1, **tkwargs)
            # test zero slack
            for eta in (1e-1, 1e-2):
                acqf = qExpectedHypervolumeImprovement(
                    model=mm,
                    ref_point=ref_point,
                    partitioning=partitioning,
                    sampler=sampler,
                    constraints=[lambda Z: torch.zeros_like(Z[..., -1])],
                    eta=eta,
                )
                res = acqf(X)
                self.assertAlmostEqual(res.item(), 0.5 * 1.5, places=4)
            # test feasible
            acqf = qExpectedHypervolumeImprovement(
                model=mm,
                ref_point=ref_point,
                partitioning=partitioning,
                sampler=sampler,
                constraints=[lambda Z: -100.0 * torch.ones_like(Z[..., -1])],
                eta=1e-3,
            )
            res = acqf(X)
            self.assertAlmostEqual(res.item(), 1.5, places=4)
            # test infeasible
            acqf = qExpectedHypervolumeImprovement(
                model=mm,
                ref_point=ref_point,
                partitioning=partitioning,
                sampler=sampler,
                constraints=[lambda Z: 100.0 * torch.ones_like(Z[..., -1])],
                eta=1e-3,
            )
            res = acqf(X)
            self.assertAlmostEqual(res.item(), 0.0, places=4)

            # TODO: Test non-trivial constraint values, multiple constraints, and q > 1


class TestQNoisyExpectedHypervolumeImprovement(BotorchTestCase):
    def setUp(self):
        self.ref_point = [0.0, 0.0, 0.0]
        self.Y_raw = torch.tensor(
            [
                [2.0, 0.5, 1.0],
                [1.0, 2.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            device=self.device,
        )
        self.pareto_Y_raw = torch.tensor(
            [
                [2.0, 0.5, 1.0],
                [1.0, 2.0, 1.0],
            ],
            device=self.device,
        )

    def test_q_noisy_expected_hypervolume_improvement(self):
        tkwargs = {"device": self.device}
        for dtype, m in product(
            (torch.float, torch.double),
            (2, 3),
        ):
            tkwargs["dtype"] = dtype
            ref_point = self.ref_point[:m]
            Y = self.Y_raw[:, :m].to(**tkwargs)
            pareto_Y = self.pareto_Y_raw[:, :m].to(**tkwargs)
            X_baseline = torch.rand(Y.shape[0], 1, **tkwargs)
            # the event shape is `b x q + r x m` = 1 x 1 x 2
            baseline_samples = Y
            samples = torch.cat(
                [baseline_samples.unsqueeze(0), torch.zeros(1, 1, m, **tkwargs)],
                dim=1,
            )
            mm = MockModel(MockPosterior(samples=baseline_samples))
            X = torch.zeros(1, 1, **tkwargs)
            # basic test
            sampler = IIDNormalSampler(num_samples=1)
            acqf = qNoisyExpectedHypervolumeImprovement(
                model=mm, ref_point=ref_point, X_baseline=X_baseline, sampler=sampler
            )
            # set the MockPosterior to use samples over baseline points and new
            # candidates
            acqf.model._posterior._samples = samples
            res = acqf(X)
            self.assertEqual(res.item(), 0.0)
            # check ref point
            self.assertTrue(
                torch.equal(acqf.ref_point, torch.tensor(ref_point, **tkwargs))
            )
            # check cached indices
            self.assertTrue(hasattr(acqf, "q_subset_indices"))
            self.assertIn("q_choose_1", acqf.q_subset_indices)
            self.assertTrue(
                torch.equal(
                    acqf.q_subset_indices["q_choose_1"],
                    torch.tensor([[0]], device=self.device),
                )
            )

            # test q=2
            X2 = torch.zeros(2, 1, **tkwargs)
            samples2 = torch.cat(
                [baseline_samples.unsqueeze(0), torch.zeros(1, 2, m, **tkwargs)],
                dim=1,
            )
            mm2 = MockModel(MockPosterior(samples=baseline_samples))
            sampler = IIDNormalSampler(num_samples=1)
            acqf = qNoisyExpectedHypervolumeImprovement(
                model=mm2, ref_point=ref_point, X_baseline=X_baseline, sampler=sampler
            )
            # set the MockPosterior to use samples over baseline points and new
            # candidates
            acqf.model._posterior._samples = samples2
            res = acqf(X2)
            self.assertEqual(res.item(), 0.0)
            # check cached indices
            self.assertTrue(hasattr(acqf, "q_subset_indices"))
            self.assertIn("q_choose_1", acqf.q_subset_indices)
            self.assertTrue(
                torch.equal(
                    acqf.q_subset_indices["q_choose_1"],
                    torch.tensor([[0], [1]], device=self.device),
                )
            )
            self.assertIn("q_choose_2", acqf.q_subset_indices)
            self.assertTrue(
                torch.equal(
                    acqf.q_subset_indices["q_choose_2"],
                    torch.tensor([[0, 1]], device=self.device),
                )
            )
            self.assertNotIn("q_choose_3", acqf.q_subset_indices)
            # now back to 1 and sure all caches were cleared
            acqf.model = mm
            res = acqf(X)
            self.assertNotIn("q_choose_2", acqf.q_subset_indices)
            self.assertIn("q_choose_1", acqf.q_subset_indices)
            self.assertTrue(
                torch.equal(
                    acqf.q_subset_indices["q_choose_1"],
                    torch.tensor([[0]], device=self.device),
                )
            )

            # test error is raised if X_baseline is batched
            sampler = IIDNormalSampler(num_samples=1)
            with self.assertRaises(UnsupportedError):
                qNoisyExpectedHypervolumeImprovement(
                    model=mm2,
                    ref_point=ref_point,
                    X_baseline=X_baseline.unsqueeze(0),
                    sampler=sampler,
                )

            # test error is raised if collapse_batch_dims=False
            sampler_uncollapsed = IIDNormalSampler(1, collapse_batch_dims=False)
            with self.assertRaises(UnsupportedError):
                qNoisyExpectedHypervolumeImprovement(
                    model=mm2,
                    ref_point=ref_point,
                    X_baseline=X_baseline,
                    sampler=sampler_uncollapsed,
                )

            # test sampler with base_samples already initialized
            sampler.base_samples = torch.rand(1, 5, 3, **tkwargs)
            mm2 = MockModel(MockPosterior(samples=baseline_samples))
            with warnings.catch_warnings(record=True) as ws, settings.debug(True):
                acqf = qNoisyExpectedHypervolumeImprovement(
                    model=mm2,
                    ref_point=ref_point,
                    X_baseline=X_baseline,
                    sampler=sampler,
                )
                self.assertEqual(
                    acqf.sampler.base_samples.shape,
                    torch.Size([1, X_baseline.shape[0], m]),
                )
                self.assertEqual(len(ws), 1)
                self.assertTrue(issubclass(ws[-1].category, BotorchWarning))

            # test objective
            # set the MockPosterior to use samples over baseline points
            mm._posterior._samples = baseline_samples
            sampler = IIDNormalSampler(num_samples=1)
            acqf = qNoisyExpectedHypervolumeImprovement(
                model=mm,
                ref_point=ref_point,
                X_baseline=X_baseline,
                sampler=sampler,
                objective=IdentityMCMultiOutputObjective(),
            )
            # sample_shape x n x m
            original_base_samples = sampler.base_samples.detach().clone()
            # set the MockPosterior to use samples over baseline points and new
            # candidates
            mm._posterior._samples = samples
            res = acqf(X)
            self.assertEqual(res.item(), 0.0)
            # test that original base samples were retained
            self.assertTrue(
                torch.equal(
                    # sample_shape x batch_shape x n x m
                    sampler.base_samples[0, 0, : original_base_samples.shape[1], :],
                    original_base_samples[0],
                )
            )

            # test that base_samples for X_baseline are fixed
            # set the MockPosterior to use samples over baseline points
            mm._posterior._samples = baseline_samples
            sampler = IIDNormalSampler(num_samples=1)
            acqf = qNoisyExpectedHypervolumeImprovement(
                model=mm, ref_point=ref_point, X_baseline=X_baseline, sampler=sampler
            )
            orig_base_sampler = deepcopy(acqf.base_sampler)
            # set the MockPosterior to use samples over baseline points and new
            # candidates
            mm._posterior._samples = samples
            with torch.no_grad():
                acqf(X)
            self.assertTrue(
                torch.equal(
                    orig_base_sampler.base_samples, acqf.base_sampler.base_samples
                )
            )
            self.assertTrue(
                torch.allclose(
                    acqf.base_sampler.base_samples,
                    acqf.sampler.base_samples[..., : X_baseline.shape[0], :],
                )
            )
            mm._posterior._samples = baseline_samples
            # test empty pareto set
            ref_point2 = [15.0, 14.0, 16.0][:m]
            sampler = IIDNormalSampler(num_samples=1)
            acqf = qNoisyExpectedHypervolumeImprovement(
                model=mm,
                ref_point=ref_point2,
                X_baseline=X_baseline,
                sampler=sampler,
                objective=IdentityMCMultiOutputObjective(),
            )
            self.assertTrue((acqf.cell_lower_bounds[..., 0] == 15).all())
            self.assertTrue((acqf.cell_lower_bounds[..., 1] == 14).all())
            if m == 3:
                self.assertTrue((acqf.cell_lower_bounds[..., 2] == 16).all())
            self.assertTrue(torch.isinf(acqf.cell_upper_bounds).all())
            for b in (acqf.cell_lower_bounds, acqf.cell_upper_bounds):
                self.assertEqual(list(b.shape), [1, 1, m])
                self.assertEqual(list(b.shape), [1, 1, m])

            # test no baseline points
            ref_point2 = [15.0, 14.0, 16.0][:m]
            sampler = IIDNormalSampler(num_samples=1)
            acqf = qNoisyExpectedHypervolumeImprovement(
                model=mm,
                ref_point=ref_point2,
                X_baseline=X_baseline,
                sampler=sampler,
                objective=IdentityMCMultiOutputObjective(),
                prune_baseline=True,
            )
            self.assertTrue((acqf.cell_lower_bounds[..., 0] == 15).all())
            self.assertTrue((acqf.cell_lower_bounds[..., 1] == 14).all())
            if m == 3:
                self.assertTrue((acqf.cell_lower_bounds[..., 2] == 16).all())
            self.assertTrue(torch.isinf(acqf.cell_upper_bounds).all())
            for b in (acqf.cell_lower_bounds, acqf.cell_upper_bounds):
                self.assertEqual(list(b.shape), [1, 1, m])
                self.assertEqual(list(b.shape), [1, 1, m])

            # test X_pending with CBD
            for incremental_nehvi in (False, True):
                mm._posterior._samples = baseline_samples
                sampler = IIDNormalSampler(num_samples=1)
                acqf = qNoisyExpectedHypervolumeImprovement(
                    model=mm,
                    ref_point=ref_point,
                    X_baseline=X_baseline,
                    sampler=sampler,
                    objective=IdentityMCMultiOutputObjective(),
                    incremental_nehvi=incremental_nehvi,
                )
                original_base_samples = sampler.base_samples.detach().clone()
                self.assertTrue(torch.equal(acqf.partitioning.pareto_Y[0], pareto_Y))
                self.assertIsNone(acqf.X_pending)
                new_Y = torch.tensor(
                    [[0.5, 3.0, 0.5][:m]], dtype=dtype, device=self.device
                )
                mm._posterior._samples = torch.cat(
                    [
                        baseline_samples,
                        new_Y,
                    ]
                ).unsqueeze(0)
                bd = DominatedPartitioning(
                    ref_point=torch.tensor(ref_point).to(**tkwargs), Y=pareto_Y
                )
                initial_hv = bd.compute_hypervolume()
                # test _initial_hvs
                if not incremental_nehvi:
                    self.assertTrue(hasattr(acqf, "_initial_hvs"))
                    self.assertTrue(torch.equal(acqf._initial_hvs, initial_hv.view(-1)))
                # test forward
                X_test = torch.rand(1, 1, dtype=dtype, device=self.device)
                with torch.no_grad():
                    val = acqf(X_test)
                bd.update(mm._posterior._samples[0, -1:])
                expected_val = bd.compute_hypervolume() - initial_hv
                self.assertTrue(torch.equal(val, expected_val.view(-1)))
                # test that original base_samples were retained
                self.assertTrue(
                    torch.equal(
                        # sample_shape x batch_shape x n x m
                        sampler.base_samples[0, 0, : original_base_samples.shape[1], :],
                        original_base_samples[0],
                    )
                )
                # test X_pending
                mm._posterior._samples = baseline_samples
                sampler = IIDNormalSampler(num_samples=1)
                acqf = qNoisyExpectedHypervolumeImprovement(
                    model=mm,
                    ref_point=ref_point,
                    X_baseline=X_baseline,
                    sampler=sampler,
                    objective=IdentityMCMultiOutputObjective(),
                    incremental_nehvi=incremental_nehvi,
                )
                # sample_shape x n x m
                original_base_samples = sampler.base_samples.detach().clone()
                mm._posterior._samples = torch.cat(
                    [
                        baseline_samples,
                        new_Y,
                    ],
                    dim=0,
                )
                X_pending = torch.rand(1, 1, dtype=dtype, device=self.device)
                acqf.set_X_pending(X_pending)
                if not incremental_nehvi:
                    self.assertTrue(torch.equal(expected_val, acqf._prev_nehvi))
                self.assertIsNone(acqf.X_pending)
                # check that X_baseline has been updated
                self.assertTrue(torch.equal(acqf.X_baseline[:-1], acqf._X_baseline))
                self.assertTrue(torch.equal(acqf.X_baseline[-1:], X_pending))
                # check that partitioning has been updated
                acqf_pareto_Y = acqf.partitioning.pareto_Y[0]
                self.assertTrue(torch.equal(acqf_pareto_Y[:-1], pareto_Y))
                self.assertTrue(torch.equal(acqf_pareto_Y[-1:], new_Y))
                # test that base samples were retained
                self.assertTrue(
                    torch.equal(
                        # sample_shape x n x m
                        sampler.base_samples[0, : original_base_samples.shape[1], :],
                        original_base_samples[0],
                    )
                )
                self.assertTrue(
                    torch.equal(
                        acqf.sampler.base_samples,
                        acqf.base_sampler.base_samples,
                    )
                )

                # test incremental nehvi in forward
                new_Y2 = torch.cat(
                    [
                        new_Y,
                        torch.tensor(
                            [[0.25, 9.5, 1.5][:m]], dtype=dtype, device=self.device
                        ),
                    ],
                    dim=0,
                )
                mm._posterior._samples = torch.cat(
                    [
                        baseline_samples,
                        new_Y2,
                    ]
                ).unsqueeze(0)
                X_test = torch.rand(1, 1, dtype=dtype, device=self.device)
                with torch.no_grad():
                    val = acqf(X_test)
                if incremental_nehvi:
                    # set initial hv to include X_pending
                    initial_hv = bd.compute_hypervolume()
                bd.update(mm._posterior._samples[0, -1:])
                expected_val = bd.compute_hypervolume() - initial_hv
                self.assertTrue(torch.equal(val, expected_val.view(-1)))

            # add another point
            X_pending2 = torch.cat(
                [X_pending, torch.rand(1, 1, dtype=dtype, device=self.device)], dim=0
            )
            mm._posterior._samples = mm._posterior._samples.squeeze(0)
            acqf.set_X_pending(X_pending2)
            self.assertIsNone(acqf.X_pending)
            # check that X_baseline has been updated
            self.assertTrue(torch.equal(acqf.X_baseline[:-2], acqf._X_baseline))
            self.assertTrue(torch.equal(acqf.X_baseline[-2:], X_pending2))
            # check that partitioning has been updated
            acqf_pareto_Y = acqf.partitioning.pareto_Y[0]
            self.assertTrue(torch.equal(acqf_pareto_Y[:-2], pareto_Y))
            self.assertTrue(torch.equal(acqf_pareto_Y[-2:], new_Y2))

            # test set X_pending with grad
            with warnings.catch_warnings(record=True) as ws, settings.debug(True):
                acqf.set_X_pending(
                    torch.cat([X_pending2, X_pending2], dim=0).requires_grad_(True)
                )
                self.assertIsNone(acqf.X_pending)
                self.assertEqual(len(ws), 1)
                self.assertTrue(issubclass(ws[-1].category, BotorchWarning))

            # test max iep
            mm._posterior._samples = baseline_samples
            sampler = IIDNormalSampler(num_samples=1)
            acqf = qNoisyExpectedHypervolumeImprovement(
                model=mm,
                ref_point=ref_point,
                X_baseline=X_baseline,
                sampler=sampler,
                objective=IdentityMCMultiOutputObjective(),
                incremental_nehvi=False,
                max_iep=1,
            )
            mm._posterior._samples = torch.cat(
                [
                    baseline_samples,
                    new_Y,
                ]
            )
            acqf.set_X_pending(X_pending)
            self.assertTrue(torch.equal(acqf.X_pending, X_pending))
            acqf_pareto_Y = acqf.partitioning.pareto_Y[0]
            self.assertTrue(torch.equal(acqf_pareto_Y, pareto_Y))
            mm._posterior._samples = torch.cat(
                [
                    baseline_samples,
                    new_Y2,
                ]
            )
            # check that after second pending point is added, X_pending is set to None
            # and the pending points are included in the box decompositions
            acqf.set_X_pending(X_pending2)
            self.assertIsNone(acqf.X_pending)
            acqf_pareto_Y = acqf.partitioning.pareto_Y[0]
            self.assertTrue(torch.equal(acqf_pareto_Y[:-2], pareto_Y))
            self.assertTrue(torch.equal(acqf_pareto_Y[-2:], new_Y2))

            # test qNEHVI without CBD
            mm._posterior._samples = baseline_samples
            sampler = IIDNormalSampler(num_samples=1)
            acqf = qNoisyExpectedHypervolumeImprovement(
                model=mm,
                ref_point=ref_point,
                X_baseline=X_baseline,
                sampler=sampler,
                objective=IdentityMCMultiOutputObjective(),
                cache_pending=False,
            )
            mm._posterior._samples = torch.cat(
                [
                    baseline_samples,
                    new_Y,
                ]
            )
            X_pending10 = X_pending.expand(10, 1)
            acqf.set_X_pending(X_pending10)
            self.assertTrue(torch.equal(acqf.X_pending, X_pending10))
            acqf_pareto_Y = acqf.partitioning.pareto_Y[0]
            self.assertTrue(torch.equal(acqf_pareto_Y, pareto_Y))
            acqf.set_X_pending(X_pending)
            mm._posterior._samples = torch.cat(
                [
                    baseline_samples,
                    new_Y2,
                ]
            )
            with torch.no_grad():
                val = acqf(X_test)
            bd = DominatedPartitioning(
                ref_point=torch.tensor(ref_point).to(**tkwargs), Y=pareto_Y
            )
            initial_hv = bd.compute_hypervolume()
            bd.update(mm._posterior._samples)
            expected_val = bd.compute_hypervolume() - initial_hv
            self.assertTrue(torch.equal(expected_val, val))
            # test alpha > 0
            mm._posterior._samples = baseline_samples
            sampler = IIDNormalSampler(num_samples=1)
            acqf = qNoisyExpectedHypervolumeImprovement(
                model=mm,
                ref_point=ref_point,
                X_baseline=X_baseline,
                sampler=sampler,
                objective=IdentityMCMultiOutputObjective(),
                cache_pending=False,
                alpha=1e-3,
            )
            if len(ref_point) == 2:
                partitioning = acqf.partitioning
            else:
                partitioning = acqf.partitioning.box_decompositions[0]
            self.assertIsInstance(partitioning, NondominatedPartitioning)
            self.assertEqual(partitioning.alpha, 1e-3)
            # test set_X_pending when X_pending = None
            acqf.set_X_pending(X_pending10)
            self.assertTrue(torch.equal(acqf.X_pending, X_pending10))
            acqf.set_X_pending(None)
            self.assertIsNone(acqf.X_pending)
            # test X_pending is not None on __init__
            mm._posterior._samples = baseline_samples
            sampler = IIDNormalSampler(num_samples=1)
            acqf = qNoisyExpectedHypervolumeImprovement(
                model=mm,
                ref_point=ref_point,
                X_baseline=X_baseline,
                sampler=sampler,
                objective=IdentityMCMultiOutputObjective(),
                alpha=1e-3,
                X_pending=X_pending2,
            )
            self.assertTrue(torch.equal(X_baseline, acqf._X_baseline))
            self.assertTrue(torch.equal(acqf.X_baseline[:-2], acqf._X_baseline))
            self.assertTrue(torch.equal(acqf.X_baseline[-2:], X_pending2))

    def test_constrained_q_noisy_expected_hypervolume_improvement(self):
        # TODO: improve tests with constraints
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            ref_point = [0.0, 0.0]
            pareto_Y = torch.tensor(
                [[4.0, 5.0], [5.0, 5.0], [8.5, 3.5], [8.5, 3.0], [9.0, 1.0]], **tkwargs
            )
            X_baseline = torch.zeros(pareto_Y.shape[0], 1, **tkwargs)
            baseline_samples = pareto_Y

            # test q=1
            # the event shape is `b x q x m` = 1 x 1 x 2
            samples = torch.cat(
                [
                    baseline_samples.unsqueeze(0),
                    torch.tensor([[[6.5, 4.5]]], **tkwargs),
                ],
                dim=1,
            )
            mm = MockModel(MockPosterior(samples=baseline_samples))
            X = torch.zeros(1, 1, **tkwargs)
            # test zero slack
            for eta in (1e-1, 1e-2):
                # set the MockPosterior to use samples over baseline points
                mm._posterior._samples = baseline_samples
                sampler = IIDNormalSampler(num_samples=1)
                acqf = qNoisyExpectedHypervolumeImprovement(
                    model=mm,
                    ref_point=ref_point,
                    X_baseline=X_baseline,
                    sampler=sampler,
                    constraints=[lambda Z: torch.zeros_like(Z[..., -1])],
                    eta=eta,
                )
                # set the MockPosterior to use samples over baseline points and new
                # candidates
                mm._posterior._samples = samples
                res = acqf(X)
                self.assertAlmostEqual(res.item(), 0.5 * 1.5, places=4)
            # set X_pending
            X_pending = torch.rand(1, 1, **tkwargs)
            acqf.set_X_pending(X_pending)
            samples = torch.cat(
                [
                    samples,
                    torch.tensor([[[10.0, 0.5]]], **tkwargs),
                ],
                dim=1,
            )
            mm._posterior._samples = samples
            res = acqf(X)
            self.assertAlmostEqual(res.item(), 0.5 * 0.5, places=4)

            # test incremental nehvi=False
            mm._posterior._samples = baseline_samples
            acqf = qNoisyExpectedHypervolumeImprovement(
                model=mm,
                ref_point=ref_point,
                X_baseline=X_baseline,
                sampler=sampler,
                constraints=[lambda Z: torch.zeros_like(Z[..., -1])],
                eta=1e-3,
                incremental_nehvi=False,
            )
            samples = torch.cat(
                [
                    baseline_samples.unsqueeze(0),
                    torch.tensor([[[6.5, 4.5]]], **tkwargs),
                ],
                dim=1,
            )
            mm._posterior._samples = samples
            res = acqf(X)
            self.assertAlmostEqual(res.item(), 0.5 * 1.5, places=4)
            acqf.set_X_pending(X_pending)
            samples = torch.cat(
                [
                    samples,
                    torch.tensor([[[10.0, 0.5]]], **tkwargs),
                ],
                dim=1,
            )
            mm._posterior._samples = samples
            res = acqf(X)
            # test that HVI is not incremental
            # Note that the cached pending point uses strict constraint evaluation
            # so the HVI from the cached pending point is 1.5.
            # The new X contributes an HVI of 0.5, but with a constraint slack of 0,
            # the sigmoid soft-evaluation yields a constrained HVI of 0.25
            self.assertAlmostEqual(res.item(), 1.75, places=4)

            # test feasible
            # set the MockPosterior to use samples over baseline points
            mm._posterior._samples = baseline_samples
            sampler = IIDNormalSampler(num_samples=1)
            acqf = qNoisyExpectedHypervolumeImprovement(
                model=mm,
                ref_point=ref_point,
                X_baseline=X_baseline,
                sampler=sampler,
                constraints=[lambda Z: -100.0 * torch.ones_like(Z[..., -1])],
                eta=1e-3,
            )
            samples = torch.cat(
                [
                    baseline_samples.unsqueeze(0),
                    torch.tensor([[[6.5, 4.5]]], **tkwargs),
                ],
                dim=1,
            )
            mm._posterior._samples = samples
            res = acqf(X)
            self.assertAlmostEqual(res.item(), 1.5, places=4)
            # test infeasible
            # set the MockPosterior to use samples over baseline points
            mm._posterior._samples = baseline_samples
            sampler = IIDNormalSampler(num_samples=1)
            acqf = qNoisyExpectedHypervolumeImprovement(
                model=mm,
                ref_point=ref_point,
                X_baseline=X_baseline,
                sampler=sampler,
                constraints=[lambda Z: 100.0 * torch.ones_like(Z[..., -1])],
                eta=1e-3,
            )
            # set the MockPosterior to use samples over baseline points and new
            # candidates
            mm._posterior._samples = samples
            res = acqf(X)
            self.assertAlmostEqual(res.item(), 0.0, places=4)

        # test >2 objectives
        ref_point = [0.0, 0.0, 0.0]
        baseline_samples = torch.tensor(
            [
                [4.0, 5.0, 1.0],
                [5.0, 5.0, 1.0],
                [8.5, 3.5, 1.0],
                [8.5, 3.0, 1.0],
                [9.0, 1.0, 1.0],
            ],
            **tkwargs,
        )
        mm._posterior._samples = baseline_samples
        sampler = IIDNormalSampler(num_samples=1)
        acqf = qNoisyExpectedHypervolumeImprovement(
            model=mm,
            ref_point=ref_point,
            X_baseline=X_baseline,
            sampler=sampler,
            constraints=[lambda Z: -100.0 * torch.ones_like(Z[..., -1])],
            eta=1e-3,
        )
        # set the MockPosterior to use samples over baseline points and new
        # candidates
        samples = torch.cat(
            [
                baseline_samples.unsqueeze(0),
                torch.tensor([[[6.5, 4.5, 1.0]]], **tkwargs),
            ],
            dim=1,
        )
        mm._posterior._samples = samples
        res = acqf(X)
        self.assertAlmostEqual(res.item(), 1.5, places=4)

    def test_prune_baseline(self):
        # test prune_baseline
        no = "botorch.utils.testing.MockModel.num_outputs"
        prune = (
            "botorch.acquisition.multi_objective.monte_carlo."
            "prune_inferior_points_multi_objective"
        )
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            ref_point = [0.0, 0.0]
            pareto_Y = torch.tensor(
                [[4.0, 5.0], [5.0, 5.0], [8.5, 3.5], [8.5, 3.0], [9.0, 1.0]], **tkwargs
            )
            X_baseline = torch.zeros(pareto_Y.shape[0], 1, **tkwargs)
            baseline_samples = pareto_Y
            X_pruned = torch.rand(1, 1, device=self.device, dtype=dtype)
            sampler = IIDNormalSampler(1)
            with mock.patch(no, new_callable=mock.PropertyMock) as mock_num_outputs:
                mock_num_outputs.return_value = 2
                mm = MockModel(MockPosterior(samples=baseline_samples))
                with mock.patch(prune, return_value=X_pruned) as mock_prune:
                    acqf = qNoisyExpectedHypervolumeImprovement(
                        model=mm,
                        ref_point=ref_point,
                        X_baseline=X_baseline,
                        sampler=sampler,
                        prune_baseline=True,
                    )
                mock_prune.assert_called_once()
                self.assertTrue(torch.equal(acqf.X_baseline, X_pruned))
