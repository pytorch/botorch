#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
from botorch.models import SingleTaskGP
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.testing import BotorchTestCase, MockModel, MockPosterior
from botorch_community.acquisition.hentropy_search import (
    qHEntropySearch,
    qLossFunctionMinMax,
    qLossFunctionTopK,
)

NO = "botorch.utils.testing.MockModel.num_outputs"


class TestQHEntropySearch(BotorchTestCase):
    def test_initialize_q_hentropy_search(self):
        for dtype in (torch.float, torch.double):
            mean = torch.zeros(1, 1, device=self.device, dtype=dtype)
            mm = MockModel(MockPosterior(mean=mean))
            # test error when neither specifying neither sampler nor num_fantasies
            with self.assertRaisesRegex(
                ValueError, "Must specify `num_points` if no `sampler` is provided."
            ):
                qHEntropySearch(
                    model=mm,
                    loss_function_class=qLossFunctionMinMax,
                    loss_function_hyperparameters={},
                    n_fantasy_at_design_pts=None,
                    n_fantasy_at_action_pts=None,
                )
            # test error when sampler and num_fantasies arg are inconsistent
            design_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([16]))
            action_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([16]))
            with self.assertRaises(ValueError):
                qHEntropySearch(
                    model=mm,
                    loss_function_class=qLossFunctionMinMax,
                    loss_function_hyperparameters={},
                    n_fantasy_at_design_pts=32,
                    n_fantasy_at_action_pts=32,
                    design_sampler=design_sampler,
                    action_sampler=action_sampler,
                )

            # test default construction
            hes = qHEntropySearch(
                model=mm,
                loss_function_class=qLossFunctionMinMax,
                loss_function_hyperparameters={},
            )
            self.assertEqual(hes.n_fantasy_at_design_pts, 64)
            self.assertEqual(hes.n_fantasy_at_action_pts, 64)
            self.assertIsInstance(hes.design_sampler, SobolQMCNormalSampler)
            self.assertIsInstance(hes.action_sampler, SobolQMCNormalSampler)
            self.assertEqual(hes.design_sampler.sample_shape, torch.Size([64]))
            self.assertEqual(hes.action_sampler.sample_shape, torch.Size([64]))
            self.assertEqual(hes.get_augmented_q_batch_size(q=3), 64 + 3)

            # test custom construction
            design_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([16]))
            action_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([16]))
            hes = qHEntropySearch(
                model=mm,
                loss_function_class=qLossFunctionMinMax,
                loss_function_hyperparameters={},
                n_fantasy_at_design_pts=16,
                n_fantasy_at_action_pts=16,
                design_sampler=design_sampler,
                action_sampler=action_sampler,
            )
            self.assertEqual(hes.n_fantasy_at_design_pts, 16)
            self.assertEqual(hes.n_fantasy_at_action_pts, 16)
            self.assertIsInstance(hes.design_sampler, SobolQMCNormalSampler)
            self.assertIsInstance(hes.action_sampler, SobolQMCNormalSampler)
            self.assertEqual(hes.design_sampler.sample_shape, torch.Size([16]))
            self.assertEqual(hes.action_sampler.sample_shape, torch.Size([16]))
            self.assertEqual(hes.get_augmented_q_batch_size(q=3), 16 + 3)

            # test assignment of num_fantasies from sampler if not provided
            hes = qHEntropySearch(
                model=mm,
                loss_function_class=qLossFunctionMinMax,
                loss_function_hyperparameters={},
                n_fantasy_at_design_pts=None,
                n_fantasy_at_action_pts=None,
                design_sampler=design_sampler,
                action_sampler=action_sampler,
            )
            self.assertEqual(hes.design_sampler.sample_shape, torch.Size([16]))
            self.assertEqual(hes.action_sampler.sample_shape, torch.Size([16]))

    def test_qhentropy_search(self):
        d = 2
        q = 2
        num_data = 3
        num_fantasies = 4
        t_batch_size = 8

        # MinMax loss function
        for dtype in (torch.float, torch.double):
            bounds = torch.tensor([[0], [1]], device=self.device, dtype=dtype)
            bounds = bounds.repeat(1, d)
            train_X = torch.rand(num_data, d, device=self.device, dtype=dtype)
            train_Y = torch.rand(num_data, 1, device=self.device, dtype=dtype)
            model = SingleTaskGP(train_X, train_Y)

            # default evaluation tests
            hes = qHEntropySearch(
                model=model,
                loss_function_class=qLossFunctionMinMax,
                loss_function_hyperparameters={},
                n_fantasy_at_design_pts=num_fantasies,
                n_fantasy_at_action_pts=num_fantasies,
            )
            eval_X = torch.rand(t_batch_size, q, d, device=self.device, dtype=dtype)
            eval_A = torch.rand(
                t_batch_size, num_fantasies, q, d, device=self.device, dtype=dtype
            )
            result = hes(X=eval_X, A=eval_A)
            self.assertEqual(result.shape, torch.Size([t_batch_size]))

            # add dummy base_weights to samplers
            samplers = [
                SobolQMCNormalSampler(sample_shape=torch.Size([num_fantasies]))
                for _ in range(2)
            ]
            for s in samplers:
                s.base_weights = torch.ones(
                    s.sample_shape[0], 1, device=self.device, dtype=dtype
                )

            hes = qHEntropySearch(
                model=model,
                loss_function_class=qLossFunctionMinMax,
                loss_function_hyperparameters={},
                n_fantasy_at_design_pts=num_fantasies,
                n_fantasy_at_action_pts=num_fantasies,
                design_sampler=samplers[0],
                action_sampler=samplers[1],
            )
            q_prime = hes.get_augmented_q_batch_size(q)

            eval_X = torch.rand(
                t_batch_size, q_prime, d, device=self.device, dtype=dtype
            )
            eval_A = torch.rand(
                t_batch_size, num_fantasies, q, d, device=self.device, dtype=dtype
            )
            result = hes(X=eval_X, A=eval_A)
            self.assertEqual(result.shape, torch.Size([t_batch_size]))

            # extract candidates
            cand = hes.extract_candidates(eval_X)
            self.assertEqual(cand.shape, torch.Size([t_batch_size, q, d]))

        # TopK loss function
        for dtype in (torch.float, torch.double):
            bounds = torch.tensor([[0], [1]], device=self.device, dtype=dtype)
            bounds = bounds.repeat(1, d)
            train_X = torch.rand(num_data, d, device=self.device, dtype=dtype)
            train_Y = torch.rand(num_data, 1, device=self.device, dtype=dtype)
            model = SingleTaskGP(train_X, train_Y)

            # default evaluation tests
            hes = qHEntropySearch(
                model=model,
                loss_function_class=qLossFunctionTopK,
                loss_function_hyperparameters={
                    "dist_weight": 1.0,
                    "dist_threshold": 0.5,
                },
                n_fantasy_at_design_pts=num_fantasies,
                n_fantasy_at_action_pts=num_fantasies,
            )
            eval_X = torch.rand(t_batch_size, q, d, device=self.device, dtype=dtype)
            eval_A = torch.rand(
                t_batch_size, num_fantasies, q, d, device=self.device, dtype=dtype
            )
            result = hes(X=eval_X, A=eval_A)
            self.assertEqual(result.shape, torch.Size([t_batch_size]))

            # add dummy base_weights to samplers
            samplers = [
                SobolQMCNormalSampler(sample_shape=torch.Size([num_fantasies]))
                for _ in range(2)
            ]
            for s in samplers:
                s.base_weights = torch.ones(
                    s.sample_shape[0], 1, device=self.device, dtype=dtype
                )

            hes = qHEntropySearch(
                model=model,
                loss_function_class=qLossFunctionTopK,
                loss_function_hyperparameters={
                    "dist_weight": 1.0,
                    "dist_threshold": 0.5,
                },
                n_fantasy_at_design_pts=num_fantasies,
                n_fantasy_at_action_pts=num_fantasies,
                design_sampler=samplers[0],
                action_sampler=samplers[1],
            )
            q_prime = hes.get_augmented_q_batch_size(q)

            eval_X = torch.rand(
                t_batch_size, q_prime, d, device=self.device, dtype=dtype
            )
            eval_A = torch.rand(
                t_batch_size, num_fantasies, q, d, device=self.device, dtype=dtype
            )
            result = hes(X=eval_X, A=eval_A)
            self.assertEqual(result.shape, torch.Size([t_batch_size]))

            # extract candidates
            cand = hes.extract_candidates(eval_X)
            self.assertEqual(cand.shape, torch.Size([t_batch_size, q, d]))
