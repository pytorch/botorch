#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
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
from botorch.models.converter import batched_to_model_list, model_list_to_batched
from botorch.utils.testing import BotorchTestCase
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
