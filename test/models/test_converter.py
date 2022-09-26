#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
from botorch.exceptions import UnsupportedError
from botorch.models import (
    FixedNoiseGP,
    HeteroskedasticSingleTaskGP,
    ModelListGP,
    SingleTaskGP,
    SingleTaskMultiFidelityGP,
)
from botorch.models.converter import (
    batched_multi_output_to_single_output,
    batched_to_model_list,
    model_list_to_batched,
)
from botorch.models.transforms.input import AppendFeatures, Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.utils.testing import BotorchTestCase
from gpytorch.kernels import RBFKernel
from gpytorch.likelihoods import GaussianLikelihood

from .test_gpytorch import SimpleGPyTorchModel


class TestConverters(BotorchTestCase):
    def test_batched_to_model_list(self):
        for dtype in (torch.float, torch.double):
            # test SingleTaskGP
            train_X = torch.rand(10, 2, device=self.device, dtype=dtype)
            train_Y1 = train_X.sum(dim=-1)
            train_Y2 = train_X[:, 0] - train_X[:, 1]
            train_Y = torch.stack([train_Y1, train_Y2], dim=-1)
            batch_gp = SingleTaskGP(train_X, train_Y)
            list_gp = batched_to_model_list(batch_gp)
            self.assertIsInstance(list_gp, ModelListGP)
            # test FixedNoiseGP
            batch_gp = FixedNoiseGP(train_X, train_Y, torch.rand_like(train_Y))
            list_gp = batched_to_model_list(batch_gp)
            self.assertIsInstance(list_gp, ModelListGP)
            # test SingleTaskMultiFidelityGP
            for lin_trunc in (False, True):
                batch_gp = SingleTaskMultiFidelityGP(
                    train_X, train_Y, iteration_fidelity=1, linear_truncated=lin_trunc
                )
                list_gp = batched_to_model_list(batch_gp)
                self.assertIsInstance(list_gp, ModelListGP)
            # test HeteroskedasticSingleTaskGP
            batch_gp = HeteroskedasticSingleTaskGP(
                train_X, train_Y, torch.rand_like(train_Y)
            )
            with self.assertRaises(NotImplementedError):
                batched_to_model_list(batch_gp)
            # test with transforms
            input_tf = Normalize(
                d=2,
                bounds=torch.tensor(
                    [[0.0, 0.0], [1.0, 1.0]], device=self.device, dtype=dtype
                ),
            )
            octf = Standardize(m=2)
            batch_gp = SingleTaskGP(
                train_X, train_Y, outcome_transform=octf, input_transform=input_tf
            )
            list_gp = batched_to_model_list(batch_gp)
            for i, m in enumerate(list_gp.models):
                self.assertIsInstance(m.input_transform, Normalize)
                self.assertTrue(torch.equal(m.input_transform.bounds, input_tf.bounds))
                self.assertIsInstance(m.outcome_transform, Standardize)
                self.assertEqual(m.outcome_transform._m, 1)
                expected_octf = octf.subset_output(idcs=[i])
                for attr_name in ["means", "stdvs", "_stdvs_sq"]:
                    self.assertTrue(
                        torch.equal(
                            m.outcome_transform.__getattr__(attr_name),
                            expected_octf.__getattr__(attr_name),
                        )
                    )
            # test with AppendFeatures
            input_tf = AppendFeatures(
                feature_set=torch.rand(2, 1, device=self.device, dtype=dtype)
            )
            batch_gp = SingleTaskGP(
                train_X, train_Y, outcome_transform=octf, input_transform=input_tf
            ).eval()
            list_gp = batched_to_model_list(batch_gp)
            self.assertIsInstance(list_gp, ModelListGP)
            self.assertIsInstance(list_gp.models[0].input_transform, AppendFeatures)

    def test_model_list_to_batched(self):
        for dtype in (torch.float, torch.double):
            # basic test
            train_X = torch.rand(10, 2, device=self.device, dtype=dtype)
            train_Y1 = train_X.sum(dim=-1, keepdim=True)
            train_Y2 = (train_X[:, 0] - train_X[:, 1]).unsqueeze(-1)
            gp1 = SingleTaskGP(train_X, train_Y1)
            gp2 = SingleTaskGP(train_X, train_Y2)
            list_gp = ModelListGP(gp1, gp2)
            batch_gp = model_list_to_batched(list_gp)
            self.assertIsInstance(batch_gp, SingleTaskGP)
            # test degenerate (single model)
            batch_gp = model_list_to_batched(ModelListGP(gp1))
            self.assertEqual(batch_gp._num_outputs, 1)
            # test different model classes
            gp2 = FixedNoiseGP(train_X, train_Y1, torch.ones_like(train_Y1))
            with self.assertRaises(UnsupportedError):
                model_list_to_batched(ModelListGP(gp1, gp2))
            # test non-batched models
            gp1_ = SimpleGPyTorchModel(train_X, train_Y1)
            gp2_ = SimpleGPyTorchModel(train_X, train_Y2)
            with self.assertRaises(UnsupportedError):
                model_list_to_batched(ModelListGP(gp1_, gp2_))
            # test list of multi-output models
            train_Y = torch.cat([train_Y1, train_Y2], dim=-1)
            gp2 = SingleTaskGP(train_X, train_Y)
            with self.assertRaises(UnsupportedError):
                model_list_to_batched(ModelListGP(gp1, gp2))
            # test different training inputs
            gp2 = SingleTaskGP(2 * train_X, train_Y2)
            with self.assertRaises(UnsupportedError):
                model_list_to_batched(ModelListGP(gp1, gp2))
            # check scalar agreement
            gp2 = SingleTaskGP(train_X, train_Y2)
            gp2.likelihood.noise_covar.noise_prior.rate.fill_(1.0)
            with self.assertRaises(UnsupportedError):
                model_list_to_batched(ModelListGP(gp1, gp2))
            # check tensor shape agreement
            gp2 = SingleTaskGP(train_X, train_Y2)
            gp2.covar_module.raw_outputscale = torch.nn.Parameter(
                torch.tensor([0.0], device=self.device, dtype=dtype)
            )
            with self.assertRaises(UnsupportedError):
                model_list_to_batched(ModelListGP(gp1, gp2))
            # test HeteroskedasticSingleTaskGP
            gp2 = HeteroskedasticSingleTaskGP(
                train_X, train_Y1, torch.ones_like(train_Y1)
            )
            with self.assertRaises(NotImplementedError):
                model_list_to_batched(ModelListGP(gp2))
            # test custom likelihood
            gp2 = SingleTaskGP(train_X, train_Y2, likelihood=GaussianLikelihood())
            with self.assertRaises(NotImplementedError):
                model_list_to_batched(ModelListGP(gp2))
            # test non-default kernel
            gp1 = SingleTaskGP(train_X, train_Y1, covar_module=RBFKernel())
            gp2 = SingleTaskGP(train_X, train_Y2, covar_module=RBFKernel())
            list_gp = ModelListGP(gp1, gp2)
            batch_gp = model_list_to_batched(list_gp)
            self.assertEqual(type(batch_gp.covar_module), RBFKernel)
            # test error when component GPs have different kernel types
            gp1 = SingleTaskGP(train_X, train_Y1, covar_module=RBFKernel())
            gp2 = SingleTaskGP(train_X, train_Y2)
            list_gp = ModelListGP(gp1, gp2)
            with self.assertRaises(UnsupportedError):
                model_list_to_batched(list_gp)
            # test FixedNoiseGP
            train_X = torch.rand(10, 2, device=self.device, dtype=dtype)
            train_Y1 = train_X.sum(dim=-1, keepdim=True)
            train_Y2 = (train_X[:, 0] - train_X[:, 1]).unsqueeze(-1)
            gp1_ = FixedNoiseGP(train_X, train_Y1, torch.rand_like(train_Y1))
            gp2_ = FixedNoiseGP(train_X, train_Y2, torch.rand_like(train_Y2))
            list_gp = ModelListGP(gp1_, gp2_)
            batch_gp = model_list_to_batched(list_gp)
            # test SingleTaskMultiFidelityGP
            gp1_ = SingleTaskMultiFidelityGP(train_X, train_Y1, iteration_fidelity=1)
            gp2_ = SingleTaskMultiFidelityGP(train_X, train_Y2, iteration_fidelity=1)
            list_gp = ModelListGP(gp1_, gp2_)
            batch_gp = model_list_to_batched(list_gp)
            gp2_ = SingleTaskMultiFidelityGP(train_X, train_Y2, iteration_fidelity=2)
            list_gp = ModelListGP(gp1_, gp2_)
            with self.assertRaises(UnsupportedError):
                model_list_to_batched(list_gp)
            # test input transform
            input_tf = Normalize(
                d=2,
                bounds=torch.tensor(
                    [[0.0, 0.0], [1.0, 1.0]], device=self.device, dtype=dtype
                ),
            )
            gp1_ = SingleTaskGP(train_X, train_Y1, input_transform=input_tf)
            gp2_ = SingleTaskGP(train_X, train_Y2, input_transform=input_tf)
            list_gp = ModelListGP(gp1_, gp2_)
            batch_gp = model_list_to_batched(list_gp)
            self.assertIsInstance(batch_gp.input_transform, Normalize)
            self.assertTrue(
                torch.equal(batch_gp.input_transform.bounds, input_tf.bounds)
            )
            # test with AppendFeatures
            input_tf3 = AppendFeatures(
                feature_set=torch.rand(2, 1, device=self.device, dtype=dtype)
            )
            gp1_ = SingleTaskGP(train_X, train_Y1, input_transform=input_tf3)
            gp2_ = SingleTaskGP(train_X, train_Y2, input_transform=input_tf3)
            list_gp = ModelListGP(gp1_, gp2_).eval()
            batch_gp = model_list_to_batched(list_gp)
            self.assertIsInstance(batch_gp, SingleTaskGP)
            self.assertIsInstance(batch_gp.input_transform, AppendFeatures)
            # test different input transforms
            input_tf2 = Normalize(
                d=2,
                bounds=torch.tensor(
                    [[-1.0, -1.0], [1.0, 1.0]], device=self.device, dtype=dtype
                ),
            )
            gp1_ = SingleTaskGP(train_X, train_Y1, input_transform=input_tf)
            gp2_ = SingleTaskGP(train_X, train_Y2, input_transform=input_tf2)
            list_gp = ModelListGP(gp1_, gp2_)
            with self.assertRaisesRegex(UnsupportedError, "have the same"):
                model_list_to_batched(list_gp)

            # test batched input transform
            input_tf2 = Normalize(
                d=2,
                bounds=torch.tensor(
                    [[-1.0, -1.0], [1.0, 1.0]], device=self.device, dtype=dtype
                ),
                batch_shape=torch.Size([3]),
            )
            gp1_ = SingleTaskGP(train_X, train_Y1, input_transform=input_tf2)
            gp2_ = SingleTaskGP(train_X, train_Y2, input_transform=input_tf2)
            list_gp = ModelListGP(gp1_, gp2_)
            with self.assertRaises(UnsupportedError):
                model_list_to_batched(list_gp)

            # test outcome transform
            octf = Standardize(m=1)
            gp1_ = SingleTaskGP(train_X, train_Y1, outcome_transform=octf)
            gp2_ = SingleTaskGP(train_X, train_Y2, outcome_transform=octf)
            list_gp = ModelListGP(gp1_, gp2_)
            with self.assertRaises(UnsupportedError):
                model_list_to_batched(list_gp)

    def test_roundtrip(self):
        for dtype in (torch.float, torch.double):
            train_X = torch.rand(10, 2, device=self.device, dtype=dtype)
            train_Y1 = train_X.sum(dim=-1)
            train_Y2 = train_X[:, 0] - train_X[:, 1]
            train_Y = torch.stack([train_Y1, train_Y2], dim=-1)
            # SingleTaskGP
            batch_gp = SingleTaskGP(train_X, train_Y)
            list_gp = batched_to_model_list(batch_gp)
            batch_gp_recov = model_list_to_batched(list_gp)
            sd_orig = batch_gp.state_dict()
            sd_recov = batch_gp_recov.state_dict()
            self.assertTrue(set(sd_orig) == set(sd_recov))
            self.assertTrue(all(torch.equal(sd_orig[k], sd_recov[k]) for k in sd_orig))
            # FixedNoiseGP
            batch_gp = FixedNoiseGP(train_X, train_Y, torch.rand_like(train_Y))
            list_gp = batched_to_model_list(batch_gp)
            batch_gp_recov = model_list_to_batched(list_gp)
            sd_orig = batch_gp.state_dict()
            sd_recov = batch_gp_recov.state_dict()
            self.assertTrue(set(sd_orig) == set(sd_recov))
            self.assertTrue(all(torch.equal(sd_orig[k], sd_recov[k]) for k in sd_orig))
            # SingleTaskMultiFidelityGP
            for lin_trunc in (False, True):
                batch_gp = SingleTaskMultiFidelityGP(
                    train_X, train_Y, iteration_fidelity=1, linear_truncated=lin_trunc
                )
                list_gp = batched_to_model_list(batch_gp)
                batch_gp_recov = model_list_to_batched(list_gp)
                sd_orig = batch_gp.state_dict()
                sd_recov = batch_gp_recov.state_dict()
                self.assertTrue(set(sd_orig) == set(sd_recov))
                self.assertTrue(
                    all(torch.equal(sd_orig[k], sd_recov[k]) for k in sd_orig)
                )

    def test_batched_multi_output_to_single_output(self):
        for dtype in (torch.float, torch.double):
            # basic test
            train_X = torch.rand(10, 2, device=self.device, dtype=dtype)
            train_Y = torch.stack(
                [
                    train_X.sum(dim=-1),
                    (train_X[:, 0] - train_X[:, 1]),
                ],
                dim=1,
            )
            batched_mo_model = SingleTaskGP(train_X, train_Y)
            batched_so_model = batched_multi_output_to_single_output(batched_mo_model)
            self.assertIsInstance(batched_so_model, SingleTaskGP)
            self.assertEqual(batched_so_model.num_outputs, 1)
            # test non-batched models
            non_batch_model = SimpleGPyTorchModel(train_X, train_Y[:, :1])
            with self.assertRaises(UnsupportedError):
                batched_multi_output_to_single_output(non_batch_model)
            gp2 = HeteroskedasticSingleTaskGP(
                train_X, train_Y, torch.ones_like(train_Y)
            )
            with self.assertRaises(NotImplementedError):
                batched_multi_output_to_single_output(gp2)
            # test custom likelihood
            gp2 = SingleTaskGP(train_X, train_Y, likelihood=GaussianLikelihood())
            with self.assertRaises(NotImplementedError):
                batched_multi_output_to_single_output(gp2)
            # test FixedNoiseGP
            train_X = torch.rand(10, 2, device=self.device, dtype=dtype)
            batched_mo_model = FixedNoiseGP(train_X, train_Y, torch.rand_like(train_Y))
            batched_so_model = batched_multi_output_to_single_output(batched_mo_model)
            self.assertIsInstance(batched_so_model, FixedNoiseGP)
            self.assertEqual(batched_so_model.num_outputs, 1)
            # test SingleTaskMultiFidelityGP
            batched_mo_model = SingleTaskMultiFidelityGP(
                train_X, train_Y, iteration_fidelity=1
            )
            batched_so_model = batched_multi_output_to_single_output(batched_mo_model)
            self.assertIsInstance(batched_so_model, SingleTaskMultiFidelityGP)
            self.assertEqual(batched_so_model.num_outputs, 1)
            # test input transform
            input_tf = Normalize(
                d=2,
                bounds=torch.tensor(
                    [[0.0, 0.0], [1.0, 1.0]], device=self.device, dtype=dtype
                ),
            )
            batched_mo_model = SingleTaskGP(train_X, train_Y, input_transform=input_tf)
            batch_so_model = batched_multi_output_to_single_output(batched_mo_model)
            self.assertIsInstance(batch_so_model.input_transform, Normalize)
            self.assertTrue(
                torch.equal(batch_so_model.input_transform.bounds, input_tf.bounds)
            )
            # test with AppendFeatures
            input_tf = AppendFeatures(
                feature_set=torch.rand(2, 1, device=self.device, dtype=dtype)
            )
            batched_mo_model = SingleTaskGP(
                train_X, train_Y, input_transform=input_tf
            ).eval()
            batch_so_model = batched_multi_output_to_single_output(batched_mo_model)
            self.assertIsInstance(batch_so_model.input_transform, AppendFeatures)

            # test batched input transform
            input_tf = Normalize(
                d=2,
                bounds=torch.tensor(
                    [[-1.0, -1.0], [1.0, 1.0]], device=self.device, dtype=dtype
                ),
                batch_shape=torch.Size([2]),
            )
            batched_mo_model = SingleTaskGP(train_X, train_Y, input_transform=input_tf)
            batch_so_model = batched_multi_output_to_single_output(batched_mo_model)
            self.assertIsInstance(batch_so_model.input_transform, Normalize)
            self.assertTrue(
                torch.equal(batch_so_model.input_transform.bounds, input_tf.bounds)
            )
            # test outcome transform
            batched_mo_model = SingleTaskGP(
                train_X, train_Y, outcome_transform=Standardize(m=2)
            )
            with self.assertRaises(NotImplementedError):
                batched_multi_output_to_single_output(batched_mo_model)
