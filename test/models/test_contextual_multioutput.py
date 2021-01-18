#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
from botorch import fit_gpytorch_model
from botorch.models.contextual_multioutput import LCEMGP, FixedNoiseLCEMGP
from botorch.models.multitask import MultiTaskGP
from botorch.posteriors import GPyTorchPosterior
from botorch.utils.testing import BotorchTestCase
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from gpytorch.lazy import LazyTensor
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from torch import Tensor


class ContextualMultiOutputTest(BotorchTestCase):
    def testLCEMGP(self):
        d = 1
        for dtype in (torch.float, torch.double):
            train_x = torch.rand(10, d, device=self.device, dtype=dtype)
            train_y = torch.cos(train_x)
            # 2 contexts here
            task_indices = torch.tensor(
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                device=self.device,
                dtype=dtype,
            )
            train_x = torch.cat([train_x, task_indices.unsqueeze(-1)], axis=1)

            model = LCEMGP(train_X=train_x, train_Y=train_y, task_feature=d)
            self.assertIsInstance(model, LCEMGP)
            self.assertIsInstance(model, MultiTaskGP)
            self.assertIsNone(model.context_emb_feature)
            self.assertIsInstance(model.context_cat_feature, Tensor)
            self.assertEqual(model.context_cat_feature.shape, torch.Size([2, 1]))
            self.assertEqual(len(model.emb_layers), 1)
            self.assertEqual(model.emb_dims, [(2, 1)])

            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_model(mll, options={"maxiter": 1})

            context_covar = model._eval_context_covar()
            self.assertIsInstance(context_covar, LazyTensor)
            self.assertEqual(context_covar.shape, torch.Size([2, 2]))

            embeddings = model._task_embeddings()
            self.assertIsInstance(embeddings, Tensor)
            self.assertEqual(embeddings.shape, torch.Size([2, 1]))

            test_x = torch.rand(5, d, device=self.device, dtype=dtype)
            task_indices = torch.tensor(
                [0.0, 0.0, 0.0, 0.0, 0.0], device=self.device, dtype=dtype
            )
            test_x = torch.cat([test_x, task_indices.unsqueeze(-1)], axis=1)
            self.assertIsInstance(model(test_x), MultivariateNormal)

            # test posterior
            posterior_f = model.posterior(test_x[:, :d])
            self.assertIsInstance(posterior_f, GPyTorchPosterior)
            self.assertIsInstance(posterior_f.mvn, MultitaskMultivariateNormal)

            # test posterior w/ single output index
            posterior_f = model.posterior(test_x[:, :d], output_indices=[0])
            self.assertIsInstance(posterior_f, GPyTorchPosterior)
            self.assertIsInstance(posterior_f.mvn, MultivariateNormal)

            # test input embs_dim_list (one categorical feature)
            # test input pre-trained emb context_emb_feature
            model2 = LCEMGP(
                train_X=train_x,
                train_Y=train_y,
                task_feature=d,
                embs_dim_list=[2],  # increase dim from 1 to 2
                context_emb_feature=torch.Tensor([[0.2], [0.3]]),
            )
            self.assertIsInstance(model2, LCEMGP)
            self.assertIsInstance(model2, MultiTaskGP)
            self.assertIsNotNone(model2.context_emb_feature)
            self.assertIsInstance(model2.context_cat_feature, Tensor)
            self.assertEqual(model2.context_cat_feature.shape, torch.Size([2, 1]))
            self.assertEqual(len(model2.emb_layers), 1)
            self.assertEqual(model2.emb_dims, [(2, 2)])

            embeddings2 = model2._task_embeddings()
            self.assertIsInstance(embeddings2, Tensor)
            self.assertEqual(embeddings2.shape, torch.Size([2, 3]))

    def testFixedNoiseLCEMGP(self):
        d = 1
        for dtype in (torch.float, torch.double):
            train_x = torch.rand(10, d, device=self.device, dtype=dtype)
            train_y = torch.cos(train_x)
            task_indices = torch.tensor(
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0], device=self.device
            )
            train_x = torch.cat([train_x, task_indices.unsqueeze(-1)], axis=1)
            train_yvar = torch.ones(10, 1, device=self.device, dtype=dtype) * 0.01

            model = FixedNoiseLCEMGP(
                train_X=train_x, train_Y=train_y, train_Yvar=train_yvar, task_feature=d
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_model(mll, options={"maxiter": 1})

            self.assertIsInstance(model, FixedNoiseLCEMGP)

            test_x = torch.rand(5, d, device=self.device, dtype=dtype)
            task_indices = torch.tensor(
                [0.0, 0.0, 0.0, 0.0, 0.0], device=self.device, dtype=dtype
            )
            test_x = torch.cat(
                [test_x, task_indices.unsqueeze(-1)],
                axis=1,
            )
            self.assertIsInstance(model(test_x), MultivariateNormal)
