#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import ExitStack
from unittest import mock

import torch
from botorch.acquisition.analytic import PosteriorMean, ScalarizedPosteriorMean
from botorch.acquisition.cost_aware import GenericCostAwareUtility
from botorch.acquisition.knowledge_gradient import (
    _get_value_function,
    _split_fantasy_points,
    qKnowledgeGradient,
    qMultiFidelityKnowledgeGradient,
    ProjectedAcquisitionFunction,
)
from botorch.acquisition.monte_carlo import qExpectedImprovement, qSimpleRegret
from botorch.acquisition.objective import GenericMCObjective, ScalarizedObjective
from botorch.acquisition.utils import project_to_sample_points
from botorch.exceptions.errors import UnsupportedError
from botorch.models import SingleTaskGP
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.sampling.samplers import IIDNormalSampler, SobolQMCNormalSampler
from botorch.utils.testing import BotorchTestCase, MockModel, MockPosterior
from gpytorch.distributions import MultitaskMultivariateNormal

NO = "botorch.utils.testing.MockModel.num_outputs"


def mock_util(X, deltas):
    return 0.5 * deltas.sum(dim=0)


class TestQKnowledgeGradient(BotorchTestCase):
    def test_initialize_q_knowledge_gradient(self):
        for dtype in (torch.float, torch.double):
            mean = torch.zeros(1, 1, device=self.device, dtype=dtype)
            mm = MockModel(MockPosterior(mean=mean))
            # test error when neither specifying neither sampler nor num_fantasies
            with self.assertRaises(ValueError):
                qKnowledgeGradient(model=mm, num_fantasies=None)
            # test error when sampler and num_fantasies arg are inconsistent
            sampler = IIDNormalSampler(num_samples=16)
            with self.assertRaises(ValueError):
                qKnowledgeGradient(model=mm, num_fantasies=32, sampler=sampler)
            # test default construction
            qKG = qKnowledgeGradient(model=mm, num_fantasies=32)
            self.assertEqual(qKG.num_fantasies, 32)
            self.assertIsInstance(qKG.sampler, SobolQMCNormalSampler)
            self.assertEqual(qKG.sampler.sample_shape, torch.Size([32]))
            self.assertIsNone(qKG.objective)
            self.assertIsNone(qKG.inner_sampler)
            self.assertIsNone(qKG.X_pending)
            self.assertIsNone(qKG.current_value)
            self.assertEqual(qKG.get_augmented_q_batch_size(q=3), 32 + 3)
            # test custom construction
            obj = GenericMCObjective(lambda Y, X: Y.mean(dim=-1))
            sampler = IIDNormalSampler(num_samples=16)
            X_pending = torch.zeros(2, 2, device=self.device, dtype=dtype)
            qKG = qKnowledgeGradient(
                model=mm,
                num_fantasies=16,
                sampler=sampler,
                objective=obj,
                X_pending=X_pending,
            )
            self.assertEqual(qKG.num_fantasies, 16)
            self.assertEqual(qKG.sampler, sampler)
            self.assertEqual(qKG.sampler.sample_shape, torch.Size([16]))
            self.assertEqual(qKG.objective, obj)
            self.assertIsInstance(qKG.inner_sampler, SobolQMCNormalSampler)
            self.assertEqual(qKG.inner_sampler.sample_shape, torch.Size([128]))
            self.assertTrue(torch.equal(qKG.X_pending, X_pending))
            self.assertIsNone(qKG.current_value)
            self.assertEqual(qKG.get_augmented_q_batch_size(q=3), 16 + 3)
            # test assignment of num_fantasies from sampler if not provided
            qKG = qKnowledgeGradient(model=mm, num_fantasies=None, sampler=sampler)
            self.assertEqual(qKG.sampler.sample_shape, torch.Size([16]))
            # test custom construction with inner sampler and current value
            inner_sampler = SobolQMCNormalSampler(num_samples=256)
            current_value = torch.zeros(1, device=self.device, dtype=dtype)
            qKG = qKnowledgeGradient(
                model=mm,
                num_fantasies=8,
                objective=obj,
                inner_sampler=inner_sampler,
                current_value=current_value,
            )
            self.assertEqual(qKG.num_fantasies, 8)
            self.assertEqual(qKG.sampler.sample_shape, torch.Size([8]))
            self.assertEqual(qKG.objective, obj)
            self.assertIsInstance(qKG.inner_sampler, SobolQMCNormalSampler)
            self.assertEqual(qKG.inner_sampler, inner_sampler)
            self.assertIsNone(qKG.X_pending)
            self.assertTrue(torch.equal(qKG.current_value, current_value))
            self.assertEqual(qKG.get_augmented_q_batch_size(q=3), 8 + 3)
            # test construction with non-MC objective (ScalarizedObjective)
            qKG_s = qKnowledgeGradient(
                model=mm,
                num_fantasies=16,
                sampler=sampler,
                objective=ScalarizedObjective(weights=torch.rand(2)),
            )
            self.assertIsNone(qKG_s.inner_sampler)
            self.assertIsInstance(qKG_s.objective, ScalarizedObjective)
            # test error if no objective and multi-output model
            mean2 = torch.zeros(1, 2, device=self.device, dtype=dtype)
            mm2 = MockModel(MockPosterior(mean=mean2))
            with self.assertRaises(UnsupportedError):
                qKnowledgeGradient(model=mm2)

    def test_evaluate_q_knowledge_gradient(self):
        for dtype in (torch.float, torch.double):
            # basic test
            n_f = 4
            mean = torch.rand(n_f, 1, 1, device=self.device, dtype=dtype)
            variance = torch.rand(n_f, 1, 1, device=self.device, dtype=dtype)
            mfm = MockModel(MockPosterior(mean=mean, variance=variance))
            with mock.patch.object(MockModel, "fantasize", return_value=mfm) as patch_f:
                with mock.patch(NO, new_callable=mock.PropertyMock) as mock_num_outputs:
                    mock_num_outputs.return_value = 1
                    mm = MockModel(None)
                    qKG = qKnowledgeGradient(model=mm, num_fantasies=n_f)
                    X = torch.rand(n_f + 1, 1, device=self.device, dtype=dtype)
                    val = qKG(X)
                    patch_f.assert_called_once()
                    cargs, ckwargs = patch_f.call_args
                    self.assertEqual(ckwargs["X"].shape, torch.Size([1, 1, 1]))
            self.assertTrue(torch.allclose(val, mean.mean(), atol=1e-4))
            self.assertTrue(torch.equal(qKG.extract_candidates(X), X[..., :-n_f, :]))
            # batched evaluation
            b = 2
            mean = torch.rand(n_f, b, 1, device=self.device, dtype=dtype)
            variance = torch.rand(n_f, b, 1, device=self.device, dtype=dtype)
            mfm = MockModel(MockPosterior(mean=mean, variance=variance))
            X = torch.rand(b, n_f + 1, 1, device=self.device, dtype=dtype)
            with mock.patch.object(MockModel, "fantasize", return_value=mfm) as patch_f:
                with mock.patch(NO, new_callable=mock.PropertyMock) as mock_num_outputs:
                    mock_num_outputs.return_value = 1
                    mm = MockModel(None)
                    qKG = qKnowledgeGradient(model=mm, num_fantasies=n_f)
                    val = qKG(X)
                    patch_f.assert_called_once()
                    cargs, ckwargs = patch_f.call_args
                    self.assertEqual(ckwargs["X"].shape, torch.Size([b, 1, 1]))
            self.assertTrue(
                torch.allclose(val, mean.mean(dim=0).squeeze(-1), atol=1e-4)
            )
            self.assertTrue(torch.equal(qKG.extract_candidates(X), X[..., :-n_f, :]))
            # pending points and current value
            X_pending = torch.rand(2, 1, device=self.device, dtype=dtype)
            mean = torch.rand(n_f, 1, 1, device=self.device, dtype=dtype)
            variance = torch.rand(n_f, 1, 1, device=self.device, dtype=dtype)
            mfm = MockModel(MockPosterior(mean=mean, variance=variance))
            current_value = torch.rand(1, device=self.device, dtype=dtype)
            X = torch.rand(n_f + 1, 1, device=self.device, dtype=dtype)
            with mock.patch.object(MockModel, "fantasize", return_value=mfm) as patch_f:
                with mock.patch(NO, new_callable=mock.PropertyMock) as mock_num_outputs:
                    mock_num_outputs.return_value = 1
                    mm = MockModel(None)
                    qKG = qKnowledgeGradient(
                        model=mm,
                        num_fantasies=n_f,
                        X_pending=X_pending,
                        current_value=current_value,
                    )
                    val = qKG(X)
                    patch_f.assert_called_once()
                    cargs, ckwargs = patch_f.call_args
                    self.assertEqual(ckwargs["X"].shape, torch.Size([1, 3, 1]))
            self.assertTrue(torch.allclose(val, mean.mean() - current_value, atol=1e-4))
            self.assertTrue(torch.equal(qKG.extract_candidates(X), X[..., :-n_f, :]))
            # test objective (inner MC sampling)
            objective = GenericMCObjective(objective=lambda Y, X: Y.norm(dim=-1))
            samples = torch.randn(3, 1, 1, device=self.device, dtype=dtype)
            mfm = MockModel(MockPosterior(samples=samples))
            X = torch.rand(n_f + 1, 1, device=self.device, dtype=dtype)
            with mock.patch.object(MockModel, "fantasize", return_value=mfm) as patch_f:
                with mock.patch(NO, new_callable=mock.PropertyMock) as mock_num_outputs:
                    mock_num_outputs.return_value = 1
                    mm = MockModel(None)
                    qKG = qKnowledgeGradient(
                        model=mm, num_fantasies=n_f, objective=objective
                    )
                    val = qKG(X)
                    patch_f.assert_called_once()
                    cargs, ckwargs = patch_f.call_args
                    self.assertEqual(ckwargs["X"].shape, torch.Size([1, 1, 1]))
            self.assertTrue(torch.allclose(val, objective(samples).mean(), atol=1e-4))
            self.assertTrue(torch.equal(qKG.extract_candidates(X), X[..., :-n_f, :]))
            # test non-MC objective (ScalarizedObjective)
            weights = torch.rand(2, device=self.device, dtype=dtype)
            objective = ScalarizedObjective(weights=weights)
            mean = torch.tensor([1.0, 0.5], device=self.device, dtype=dtype).expand(
                n_f, 1, 2
            )
            cov = torch.tensor(
                [[1.0, 0.1], [0.1, 0.5]], device=self.device, dtype=dtype
            ).expand(n_f, 2, 2)
            posterior = GPyTorchPosterior(MultitaskMultivariateNormal(mean, cov))
            mfm = MockModel(posterior)
            with mock.patch.object(MockModel, "fantasize", return_value=mfm) as patch_f:
                with mock.patch(NO, new_callable=mock.PropertyMock) as mock_num_outputs:
                    mock_num_outputs.return_value = 2
                    mm = MockModel(None)
                    qKG = qKnowledgeGradient(
                        model=mm, num_fantasies=n_f, objective=objective
                    )
                    val = qKG(X)
                    patch_f.assert_called_once()
                    cargs, ckwargs = patch_f.call_args
                    self.assertEqual(ckwargs["X"].shape, torch.Size([1, 1, 1]))
                    val_expected = (mean * weights).sum(-1).mean(0)
                    self.assertTrue(torch.allclose(val, val_expected))

    def test_evaluate_kg(self):
        # a thorough test using real model and dtype double
        d = 2
        dtype = torch.double
        bounds = torch.tensor([[0], [1]], device=self.device, dtype=dtype).repeat(1, d)
        train_X = torch.rand(3, d, device=self.device, dtype=dtype)
        train_Y = torch.rand(3, 1, device=self.device, dtype=dtype)
        model = SingleTaskGP(train_X, train_Y)
        qKG = qKnowledgeGradient(
            model=model,
            num_fantasies=2,
            objective=None,
            X_pending=torch.rand(2, d, device=self.device, dtype=dtype),
            current_value=torch.rand(1, device=self.device, dtype=dtype),
        )
        X = torch.rand(4, 3, d, device=self.device, dtype=dtype)
        options = {"num_inner_restarts": 2, "raw_inner_samples": 3}
        val = qKG.evaluate(
            X, bounds=bounds, num_restarts=2, raw_samples=3, options=options
        )
        # verify output shape
        self.assertEqual(val.size(), torch.Size([4]))
        # verify dtype
        self.assertEqual(val.dtype, dtype)

        # test i) no dimension is squeezed out, ii) dtype float, iii) MC objective,
        # and iv) t_batch_mode_transform
        dtype = torch.float
        bounds = torch.tensor([[0], [1]], device=self.device, dtype=dtype)
        train_X = torch.rand(1, 1, device=self.device, dtype=dtype)
        train_Y = torch.rand(1, 1, device=self.device, dtype=dtype)
        model = SingleTaskGP(train_X, train_Y)
        qKG = qKnowledgeGradient(
            model=model,
            num_fantasies=1,
            objective=GenericMCObjective(objective=lambda Y, X: Y.norm(dim=-1)),
        )
        X = torch.rand(1, 1, device=self.device, dtype=dtype)
        options = {"num_inner_restarts": 1, "raw_inner_samples": 1}
        val = qKG.evaluate(
            X, bounds=bounds, num_restarts=1, raw_samples=1, options=options
        )
        # verify output shape
        self.assertEqual(val.size(), torch.Size([1]))
        # verify dtype
        self.assertEqual(val.dtype, dtype)


class TestQMultiFidelityKnowledgeGradient(BotorchTestCase):
    def test_initialize_qMFKG(self):
        for dtype in (torch.float, torch.double):
            mean = torch.zeros(1, 1, device=self.device, dtype=dtype)
            mm = MockModel(MockPosterior(mean=mean))
            # test error when not specifying current_value
            with self.assertRaises(UnsupportedError):
                qMultiFidelityKnowledgeGradient(
                    model=mm, num_fantasies=None, cost_aware_utility=mock.Mock()
                )
            # test default construction
            mock_cau = mock.Mock()
            current_value = torch.zeros(1, device=self.device, dtype=dtype)
            qMFKG = qMultiFidelityKnowledgeGradient(
                model=mm,
                num_fantasies=32,
                current_value=current_value,
                cost_aware_utility=mock_cau,
            )
            self.assertEqual(qMFKG.num_fantasies, 32)
            self.assertIsInstance(qMFKG.sampler, SobolQMCNormalSampler)
            self.assertEqual(qMFKG.sampler.sample_shape, torch.Size([32]))
            self.assertIsNone(qMFKG.objective)
            self.assertIsNone(qMFKG.inner_sampler)
            self.assertIsNone(qMFKG.X_pending)
            self.assertEqual(qMFKG.get_augmented_q_batch_size(q=3), 32 + 3)
            self.assertEqual(qMFKG.cost_aware_utility, mock_cau)
            self.assertTrue(torch.equal(qMFKG.current_value, current_value))
            self.assertIsNone(qMFKG._cost_sampler)
            X = torch.rand(2, 3, device=self.device, dtype=dtype)
            self.assertTrue(torch.equal(qMFKG.project(X), X))
            self.assertTrue(torch.equal(qMFKG.expand(X), X))
            self.assertIsNone(qMFKG.valfunc_cls)
            self.assertIsNone(qMFKG.valfunc_argfac)
            # make sure cost sampling logic works
            self.assertIsInstance(qMFKG.cost_sampler, SobolQMCNormalSampler)
            self.assertEqual(qMFKG.cost_sampler.sample_shape, torch.Size([32]))

    def test_evaluate_qMFKG(self):
        for dtype in (torch.float, torch.double):
            # basic test
            n_f = 4
            current_value = torch.rand(1, device=self.device, dtype=dtype)
            cau = GenericCostAwareUtility(mock_util)
            mean = torch.rand(n_f, 1, 1, device=self.device, dtype=dtype)
            variance = torch.rand(n_f, 1, 1, device=self.device, dtype=dtype)
            mfm = MockModel(MockPosterior(mean=mean, variance=variance))
            with mock.patch.object(MockModel, "fantasize", return_value=mfm) as patch_f:
                with mock.patch(NO, new_callable=mock.PropertyMock) as mock_num_outputs:
                    mock_num_outputs.return_value = 1
                    mm = MockModel(None)
                    qMFKG = qMultiFidelityKnowledgeGradient(
                        model=mm,
                        num_fantasies=n_f,
                        current_value=current_value,
                        cost_aware_utility=cau,
                    )
                    X = torch.rand(n_f + 1, 1, device=self.device, dtype=dtype)
                    val = qMFKG(X)
                    patch_f.assert_called_once()
                    cargs, ckwargs = patch_f.call_args
                    self.assertEqual(ckwargs["X"].shape, torch.Size([1, 1, 1]))
            val_exp = mock_util(X, mean.squeeze(-1) - current_value).mean(dim=0)
            self.assertTrue(torch.allclose(val, val_exp, atol=1e-4))
            self.assertTrue(torch.equal(qMFKG.extract_candidates(X), X[..., :-n_f, :]))
            # batched evaluation
            b = 2
            current_value = torch.rand(b, device=self.device, dtype=dtype)
            cau = GenericCostAwareUtility(mock_util)
            mean = torch.rand(n_f, b, 1, device=self.device, dtype=dtype)
            variance = torch.rand(n_f, b, 1, device=self.device, dtype=dtype)
            mfm = MockModel(MockPosterior(mean=mean, variance=variance))
            X = torch.rand(b, n_f + 1, 1, device=self.device, dtype=dtype)
            with mock.patch.object(MockModel, "fantasize", return_value=mfm) as patch_f:
                with mock.patch(NO, new_callable=mock.PropertyMock) as mock_num_outputs:
                    mock_num_outputs.return_value = 1
                    mm = MockModel(None)
                    qMFKG = qMultiFidelityKnowledgeGradient(
                        model=mm,
                        num_fantasies=n_f,
                        current_value=current_value,
                        cost_aware_utility=cau,
                    )
                    val = qMFKG(X)
                    patch_f.assert_called_once()
                    cargs, ckwargs = patch_f.call_args
                    self.assertEqual(ckwargs["X"].shape, torch.Size([b, 1, 1]))
            val_exp = mock_util(X, mean.squeeze(-1) - current_value).mean(dim=0)
            self.assertTrue(torch.allclose(val, val_exp, atol=1e-4))
            self.assertTrue(torch.equal(qMFKG.extract_candidates(X), X[..., :-n_f, :]))
            # pending points and current value
            mean = torch.rand(n_f, 1, 1, device=self.device, dtype=dtype)
            variance = torch.rand(n_f, 1, 1, device=self.device, dtype=dtype)
            X_pending = torch.rand(2, 1, device=self.device, dtype=dtype)
            mfm = MockModel(MockPosterior(mean=mean, variance=variance))
            current_value = torch.rand(1, device=self.device, dtype=dtype)
            X = torch.rand(n_f + 1, 1, device=self.device, dtype=dtype)
            with mock.patch.object(MockModel, "fantasize", return_value=mfm) as patch_f:
                with mock.patch(NO, new_callable=mock.PropertyMock) as mock_num_outputs:
                    mock_num_outputs.return_value = 1
                    mm = MockModel(None)
                    qMFKG = qMultiFidelityKnowledgeGradient(
                        model=mm,
                        num_fantasies=n_f,
                        X_pending=X_pending,
                        current_value=current_value,
                        cost_aware_utility=cau,
                    )
                    val = qMFKG(X)
                    patch_f.assert_called_once()
                    cargs, ckwargs = patch_f.call_args
                    self.assertEqual(ckwargs["X"].shape, torch.Size([1, 3, 1]))
            val_exp = mock_util(X, mean.squeeze(-1) - current_value).mean(dim=0)
            self.assertTrue(torch.allclose(val, val_exp, atol=1e-4))
            self.assertTrue(torch.equal(qMFKG.extract_candidates(X), X[..., :-n_f, :]))
            # test objective (inner MC sampling)
            objective = GenericMCObjective(objective=lambda Y, X: Y.norm(dim=-1))
            samples = torch.randn(3, 1, 1, device=self.device, dtype=dtype)
            mfm = MockModel(MockPosterior(samples=samples))
            X = torch.rand(n_f + 1, 1, device=self.device, dtype=dtype)
            with mock.patch.object(MockModel, "fantasize", return_value=mfm) as patch_f:
                with mock.patch(NO, new_callable=mock.PropertyMock) as mock_num_outputs:
                    mock_num_outputs.return_value = 1
                    mm = MockModel(None)
                    qMFKG = qMultiFidelityKnowledgeGradient(
                        model=mm,
                        num_fantasies=n_f,
                        objective=objective,
                        current_value=current_value,
                        cost_aware_utility=cau,
                    )
                    val = qMFKG(X)
                    patch_f.assert_called_once()
                    cargs, ckwargs = patch_f.call_args
                    self.assertEqual(ckwargs["X"].shape, torch.Size([1, 1, 1]))
            val_exp = mock_util(X, objective(samples) - current_value).mean(dim=0)
            self.assertTrue(torch.allclose(val, val_exp, atol=1e-4))
            self.assertTrue(torch.equal(qMFKG.extract_candidates(X), X[..., :-n_f, :]))
            # test valfunc_cls and valfunc_argfac
            d, p, d_prime = 4, 3, 2
            samples = torch.ones(3, 1, 1, device=self.device, dtype=dtype)
            mean = torch.tensor(
                [[0.25], [0.5], [0.75]], device=self.device, dtype=dtype
            )
            weights = torch.tensor([0.5, 1.0, 1.0], device=self.device, dtype=dtype)
            mfm = MockModel(MockPosterior(mean=mean, samples=samples))
            X = torch.rand(n_f * d + d, d, device=self.device, dtype=dtype)
            sample_points = torch.rand(p, d_prime, device=self.device, dtype=dtype)
            with mock.patch.object(MockModel, "fantasize", return_value=mfm) as patch_f:
                with mock.patch(NO, new_callable=mock.PropertyMock) as mock_num_outputs:
                    mock_num_outputs.return_value = 1
                    mm = MockModel(None)
                    qMFKG = qMultiFidelityKnowledgeGradient(
                        model=mm,
                        num_fantasies=n_f,
                        project=lambda X: project_to_sample_points(X, sample_points),
                        valfunc_cls=ScalarizedPosteriorMean,
                        valfunc_argfac=lambda model: {"weights": weights},
                    )
                    val = qMFKG(X)
                    patch_f.assert_called_once()
                    cargs, ckwargs = patch_f.call_args
                    self.assertEqual(ckwargs["X"].shape, torch.Size([1, 16, 4]))
                    val_exp = torch.tensor([1.375], dtype=dtype)
                    self.assertTrue(torch.allclose(val, val_exp, atol=1e-4))

                    patch_f.reset_mock()
                    qMFKG = qMultiFidelityKnowledgeGradient(
                        model=mm,
                        num_fantasies=n_f,
                        project=lambda X: project_to_sample_points(X, sample_points),
                        valfunc_cls=qExpectedImprovement,
                        valfunc_argfac=lambda model: {"best_f": 0.0},
                    )
                    val = qMFKG(X)
                    patch_f.assert_called_once()
                    cargs, ckwargs = patch_f.call_args
                    self.assertEqual(ckwargs["X"].shape, torch.Size([1, 16, 4]))
                    val_exp = torch.tensor([1.0], dtype=dtype)
                    self.assertTrue(torch.allclose(val, val_exp, atol=1e-4))

    def test_fixed_evaluation_qMFKG(self):
        # mock test qMFKG.evaluate() with expand, project & cost aware utility
        for dtype in (torch.float, torch.double):
            mean = torch.zeros(1, 1, 1, device=self.device, dtype=dtype)
            mm = MockModel(MockPosterior(mean=mean))
            cau = GenericCostAwareUtility(mock_util)
            n_f = 4
            mean = torch.rand(n_f, 2, 1, 1, device=self.device, dtype=dtype)
            variance = torch.rand(n_f, 2, 1, 1, device=self.device, dtype=dtype)
            mfm = MockModel(MockPosterior(mean=mean, variance=variance))
            with ExitStack() as es:
                patch_f = es.enter_context(
                    mock.patch.object(MockModel, "fantasize", return_value=mfm)
                )
                mock_num_outputs = es.enter_context(
                    mock.patch(NO, new_callable=mock.PropertyMock)
                )
                es.enter_context(
                    mock.patch(
                        "botorch.optim.optimize.optimize_acqf",
                        return_value=(
                            torch.ones(1, 1, 1, device=self.device, dtype=dtype),
                            torch.ones(1, device=self.device, dtype=dtype),
                        ),
                    ),
                )
                es.enter_context(
                    mock.patch(
                        "botorch.generation.gen.gen_candidates_scipy",
                        return_value=(
                            torch.ones(1, 1, 1, device=self.device, dtype=dtype),
                            torch.ones(1, device=self.device, dtype=dtype),
                        ),
                    ),
                )

                mock_num_outputs.return_value = 1
                qMFKG = qMultiFidelityKnowledgeGradient(
                    model=mm,
                    num_fantasies=n_f,
                    X_pending=torch.rand(1, 1, 1, device=self.device, dtype=dtype),
                    current_value=torch.zeros(1, device=self.device, dtype=dtype),
                    cost_aware_utility=cau,
                    project=lambda X: torch.zeros_like(X),
                    expand=lambda X: torch.ones_like(X),
                )
                val = qMFKG.evaluate(
                    X=torch.zeros(1, 1, 1, device=self.device, dtype=dtype),
                    bounds=torch.tensor([[0.0], [1.0]]),
                    num_restarts=1,
                    raw_samples=1,
                )
                patch_f.asset_called_once()
                cargs, ckwargs = patch_f.call_args
                self.assertTrue(
                    torch.equal(
                        ckwargs["X"],
                        torch.ones(1, 2, 1, device=self.device, dtype=dtype),
                    )
                )
            self.assertEqual(
                val, cau(None, torch.ones(1, device=self.device, dtype=dtype))
            )


class TestKGUtils(BotorchTestCase):
    def test_get_value_function(self):
        with mock.patch(NO, new_callable=mock.PropertyMock) as mock_num_outputs:
            mock_num_outputs.return_value = 1
            mm = MockModel(None)
            # test PosteriorMean
            vf = _get_value_function(mm)
            self.assertIsInstance(vf, PosteriorMean)
            self.assertIsNone(vf.objective)
            # test SimpleRegret
            obj = GenericMCObjective(lambda Y, X: Y.sum(dim=-1))
            sampler = IIDNormalSampler(num_samples=2)
            vf = _get_value_function(model=mm, objective=obj, sampler=sampler)
            self.assertIsInstance(vf, qSimpleRegret)
            self.assertEqual(vf.objective, obj)
            self.assertEqual(vf.sampler, sampler)
            # test with project
            mock_project = mock.Mock(
                return_value=torch.ones(1, 1, 1, device=self.device)
            )
            vf = _get_value_function(
                model=mm,
                objective=obj,
                sampler=sampler,
                project=mock_project,
            )
            self.assertIsInstance(vf, ProjectedAcquisitionFunction)
            self.assertEqual(vf.objective, obj)
            self.assertEqual(vf.sampler, sampler)
            self.assertEqual(vf.project, mock_project)
            test_X = torch.rand(1, 1, 1, device=self.device)
            with mock.patch.object(
                vf, "base_value_function", __class__=torch.nn.Module, return_value=None
            ) as patch_bvf:
                vf(test_X)
                mock_project.assert_called_once_with(test_X)
                patch_bvf.assert_called_once_with(
                    torch.ones(1, 1, 1, device=self.device)
                )

    def test_split_fantasy_points(self):
        for dtype in (torch.float, torch.double):
            X = torch.randn(5, 3, device=self.device, dtype=dtype)
            # test error when passing inconsistent n_f
            with self.assertRaises(ValueError):
                _split_fantasy_points(X, n_f=6)
            # basic test
            X_actual, X_fantasies = _split_fantasy_points(X=X, n_f=2)
            self.assertEqual(X_actual.shape, torch.Size([3, 3]))
            self.assertEqual(X_fantasies.shape, torch.Size([2, 1, 3]))
            self.assertTrue(torch.equal(X_actual, X[:3, :]))
            self.assertTrue(torch.equal(X_fantasies, X[3:, :].unsqueeze(-2)))
            # batched test
            X = torch.randn(2, 5, 3, device=self.device, dtype=dtype)
            X_actual, X_fantasies = _split_fantasy_points(X=X, n_f=2)
            self.assertEqual(X_actual.shape, torch.Size([2, 3, 3]))
            self.assertEqual(X_fantasies.shape, torch.Size([2, 2, 1, 3]))
            self.assertTrue(torch.equal(X_actual, X[..., :3, :]))
            X_fantasies_exp = X[..., 3:, :].unsqueeze(-2).permute(1, 0, 2, 3)
            self.assertTrue(torch.equal(X_fantasies, X_fantasies_exp))
