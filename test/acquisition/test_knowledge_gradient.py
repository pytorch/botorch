#! /usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from unittest import mock

import torch
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.acquisition.objective import GenericMCObjective
from botorch.sampling.samplers import IIDNormalSampler, SobolQMCNormalSampler
from botorch.utils.testing import BotorchTestCase, MockModel, MockPosterior


class TestQKnowledgeGradient(BotorchTestCase):
    def test_initialize_q_knowledge_gradient(self):
        for dtype in (torch.float, torch.double):
            mm = MockModel(MockPosterior())
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
            obj = GenericMCObjective(lambda Y: Y.mean(dim=-1))
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

    def test_evaluate_q_knowledge_gradient(self):
        for dtype in (torch.float, torch.double):
            # basic test
            n_f = 4
            mean = torch.rand(n_f, 1, device=self.device, dtype=dtype)
            variance = torch.rand(n_f, 1, device=self.device, dtype=dtype)
            mfm = MockModel(MockPosterior(mean=mean, variance=variance))
            with mock.patch.object(MockModel, "fantasize", return_value=mfm) as patch_f:
                mm = MockModel(None)
                qKG = qKnowledgeGradient(model=mm, num_fantasies=n_f)
                X = torch.rand(n_f + 1, 1, device=self.device, dtype=dtype)
                val = qKG(X)
                patch_f.assert_called_once()
                cargs, ckwargs = patch_f.call_args
                self.assertEqual(ckwargs["X"].shape, torch.Size([1, 1]))
            self.assertTrue(torch.allclose(val, mean.mean(), atol=1e-4))
            self.assertTrue(torch.equal(qKG.extract_candidates(X), X[..., :-n_f, :]))
            # batched evaluation
            b = 2
            mean = torch.rand(n_f, b, 1, device=self.device, dtype=dtype)
            variance = torch.rand(n_f, b, 1, device=self.device, dtype=dtype)
            mfm = MockModel(MockPosterior(mean=mean, variance=variance))
            X = torch.rand(b, n_f + 1, 1, device=self.device, dtype=dtype)
            with mock.patch.object(MockModel, "fantasize", return_value=mfm) as patch_f:
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
            mean = torch.rand(n_f, 1, device=self.device, dtype=dtype)
            variance = torch.rand(n_f, 1, device=self.device, dtype=dtype)
            X_pending = torch.rand(2, 1, device=self.device, dtype=dtype)
            mfm = MockModel(MockPosterior(mean=mean, variance=variance))
            current_value = torch.rand(1, device=self.device, dtype=dtype)
            X = torch.rand(n_f + 1, 1, device=self.device, dtype=dtype)
            with mock.patch.object(MockModel, "fantasize", return_value=mfm) as patch_f:
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
                self.assertEqual(ckwargs["X"].shape, torch.Size([3, 1]))
            self.assertTrue(torch.allclose(val, mean.mean() - current_value, atol=1e-4))
            self.assertTrue(torch.equal(qKG.extract_candidates(X), X[..., :-n_f, :]))
            # test objective (inner MC sampling)
            objective = GenericMCObjective(objective=lambda Y: Y.norm(dim=-1))
            samples = torch.randn(3, 1, 1, device=self.device, dtype=dtype)
            mfm = MockModel(MockPosterior(samples=samples))
            X = torch.rand(n_f + 1, 1, device=self.device, dtype=dtype)
            with mock.patch.object(MockModel, "fantasize", return_value=mfm) as patch_f:
                mm = MockModel(None)
                qKG = qKnowledgeGradient(
                    model=mm, num_fantasies=n_f, objective=objective
                )
                val = qKG(X)
                patch_f.assert_called_once()
                cargs, ckwargs = patch_f.call_args
                self.assertEqual(ckwargs["X"].shape, torch.Size([1, 1]))
            self.assertTrue(torch.allclose(val, objective(samples).mean(), atol=1e-4))
            self.assertTrue(torch.equal(qKG.extract_candidates(X), X[..., :-n_f, :]))
