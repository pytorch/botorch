#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings

import torch
from botorch import settings
from botorch.acquisition.multi_objective.monte_carlo import (
    MultiObjectiveMCAcquisitionFunction,
    qExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.objective import (
    IdentityMCMultiOutputObjective,
    MCMultiOutputObjective,
)
from botorch.acquisition.objective import IdentityMCObjective
from botorch.exceptions.errors import BotorchError, UnsupportedError
from botorch.exceptions.warnings import BotorchWarning
from botorch.sampling.samplers import IIDNormalSampler, SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
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
        self.assertEqual(acqf.sampler._sample_shape, torch.Size([512]))
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
