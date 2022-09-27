#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from botorch.models.kernels.contextual_lcea import LCEAKernel
from botorch.models.kernels.contextual_sac import SACKernel
from botorch.utils.testing import BotorchTestCase
from gpytorch.kernels.matern_kernel import MaternKernel
from torch import Tensor
from torch.nn import ModuleDict


class ContextualKernelTest(BotorchTestCase):
    def test_SACKernel(self):
        decomposition = {"1": [0, 3], "2": [1, 2]}
        kernel = SACKernel(decomposition=decomposition, batch_shape=torch.Size([]))

        self.assertIsInstance(kernel.kernel_dict, ModuleDict)
        self.assertIsInstance(kernel.base_kernel, MaternKernel)
        self.assertDictEqual(kernel.decomposition, decomposition)

        # test diag works well for lazy tensor
        x1 = torch.rand(5, 4)
        x2 = torch.rand(5, 4)
        res = kernel(x1, x2).to_dense()
        res_diag = kernel(x1, x2, diag=True)
        self.assertLess(torch.norm(res_diag - res.diag()), 1e-4)

        # test raise of ValueError
        with self.assertRaises(ValueError):
            SACKernel(decomposition={"1": [0, 3], "2": [1]}, batch_shape=torch.Size([]))

    def testLCEAKernel(self):
        decomposition = {"1": [0, 3], "2": [1, 2]}
        num_contexts = len(decomposition)

        kernel = LCEAKernel(decomposition=decomposition, batch_shape=torch.Size([]))
        # test init
        self.assertListEqual(kernel.context_list, ["1", "2"])
        self.assertDictEqual(kernel.decomposition, decomposition)

        self.assertIsInstance(kernel.base_kernel, MaternKernel)
        self.assertIsInstance(kernel.task_covar_module, MaternKernel)

        # test raise of ValueError
        with self.assertRaises(ValueError):
            LCEAKernel(
                decomposition={"1": [0, 3], "2": [1]}, batch_shape=torch.Size([])
            )

        # test set_outputscale_list
        kernel.initialize(outputscale_list=[0.5, 0.5])
        actual_value = torch.tensor([0.5, 0.5]).view_as(kernel.outputscale_list)
        self.assertLess(torch.norm(kernel.outputscale_list - actual_value), 1e-5)

        self.assertTrue(kernel.train_embedding)
        self.assertEqual(kernel.num_contexts, num_contexts)
        self.assertEqual(kernel.n_embs, 1)
        self.assertIsNone(kernel.context_emb_feature)
        self.assertIsInstance(kernel.context_cat_feature, Tensor)
        self.assertEqual(len(kernel.emb_layers), 1)
        self.assertListEqual(kernel.emb_dims, [(num_contexts, 1)])

        context_covar = kernel._eval_context_covar()
        self.assertIsInstance(context_covar, Tensor)
        self.assertEqual(context_covar.shape, torch.Size([num_contexts, num_contexts]))

        embeddings = kernel._task_embeddings()
        self.assertIsInstance(embeddings, Tensor)
        self.assertEqual(embeddings.shape, torch.Size([num_contexts, 1]))

        self.assertIsInstance(kernel.outputscale_list, Tensor)
        self.assertEqual(kernel.outputscale_list.shape, torch.Size([num_contexts]))

        # test diag works well for lazy tensor
        x1 = torch.rand(5, 4)
        x2 = torch.rand(5, 4)
        res = kernel(x1, x2).to_dense()
        res_diag = kernel(x1, x2, diag=True)
        self.assertLess(torch.norm(res_diag - res.diag()), 1e-4)

        # test batch evaluation
        x1 = torch.rand(3, 5, 4)
        x2 = torch.rand(3, 5, 4)
        res = kernel(x1, x2).to_dense()
        self.assertEqual(res.shape, torch.Size([3, 5, 5]))

        # test input context_weight,
        # test input embs_dim_list (one categorical feature)
        # test input context_cat_feature
        embs_dim_list = [2]
        kernel2 = LCEAKernel(
            decomposition=decomposition,
            context_weight_dict={"1": 0.5, "2": 0.8},
            cat_feature_dict={"1": [0], "2": [1]},
            embs_dim_list=embs_dim_list,  # increase dim from 1 to 2
            batch_shape=torch.Size([]),
        )

        self.assertEqual(kernel2.num_contexts, num_contexts)
        self.assertEqual(kernel2.n_embs, 2)
        self.assertIsNone(kernel2.context_emb_feature)
        self.assertIsInstance(kernel2.context_cat_feature, Tensor)
        self.assertEqual(
            kernel2.context_cat_feature.shape, torch.Size([num_contexts, 1])
        )
        self.assertEqual(len(kernel2.emb_layers), 1)
        self.assertListEqual(kernel2.emb_dims, [(num_contexts, embs_dim_list[0])])

        context_covar2 = kernel2._eval_context_covar()
        self.assertIsInstance(context_covar2, Tensor)
        self.assertEqual(context_covar2.shape, torch.Size([num_contexts, num_contexts]))

        # test input pre-trained embedding
        kernel3 = LCEAKernel(
            decomposition=decomposition,
            embs_feature_dict={"1": [0.2], "2": [0.5]},
            batch_shape=torch.Size([]),
        )
        self.assertEqual(kernel3.num_contexts, num_contexts)
        self.assertEqual(kernel3.n_embs, 2)
        self.assertIsNotNone(kernel3.context_emb_feature)
        self.assertIsInstance(kernel3.context_emb_feature, Tensor)
        self.assertIsInstance(kernel3.context_cat_feature, Tensor)
        self.assertEqual(
            kernel3.context_cat_feature.shape, torch.Size([num_contexts, 1])
        )
        self.assertListEqual(kernel3.emb_dims, [(num_contexts, 1)])
        embeddings3 = kernel3._task_embeddings()
        self.assertEqual(embeddings3.shape, torch.Size([num_contexts, 2]))

        # test only use pre-trained embedding
        kernel4 = LCEAKernel(
            decomposition=decomposition,
            train_embedding=False,
            embs_feature_dict={"1": [0.2], "2": [0.5]},
            batch_shape=torch.Size([]),
        )
        self.assertEqual(kernel4.n_embs, 1)
        self.assertIsNotNone(kernel4.context_emb_feature)
        self.assertIsInstance(kernel4.context_emb_feature, Tensor)
        self.assertIsInstance(kernel4.context_cat_feature, Tensor)
        embeddings4 = kernel4._task_embeddings()
        self.assertEqual(embeddings4.shape, torch.Size([num_contexts, 1]))

        # test batch
        kernel5 = LCEAKernel(decomposition=decomposition, batch_shape=torch.Size([3]))
        self.assertEqual(kernel5.n_embs, 1)  # one dim cat
        self.assertListEqual(kernel5.emb_dims, [(num_contexts, 1)])

        embeddings_batch = kernel5._task_embeddings_batch()
        self.assertIsInstance(embeddings_batch, Tensor)
        self.assertEqual(embeddings_batch.shape, torch.Size([3, num_contexts, 1]))

        context_covar5 = kernel5._eval_context_covar()
        self.assertIsInstance(context_covar5, Tensor)
        self.assertEqual(
            context_covar5.shape, torch.Size([3, num_contexts, num_contexts])
        )

        # test batch with pre-trained features
        kernel6 = LCEAKernel(
            decomposition=decomposition,
            batch_shape=torch.Size([3]),
            embs_feature_dict={"1": [0.2], "2": [0.5]},
        )
        self.assertEqual(kernel6.n_embs, 2)  # one dim cat + one dim pre-train
        self.assertListEqual(kernel6.emb_dims, [(num_contexts, 1)])  # one dim for cat

        embeddings_batch = kernel6._task_embeddings_batch()
        self.assertIsInstance(embeddings_batch, Tensor)
        self.assertEqual(
            embeddings_batch.shape, torch.Size([3, num_contexts, num_contexts])
        )

        context_covar6 = kernel6._eval_context_covar()
        self.assertIsInstance(context_covar6, Tensor)
        self.assertEqual(
            context_covar6.shape, torch.Size([3, num_contexts, num_contexts])
        )
