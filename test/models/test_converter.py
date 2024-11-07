#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
from botorch.exceptions import UnsupportedError
from botorch.models import ModelListGP, SingleTaskGP, SingleTaskMultiFidelityGP
from botorch.models.converter import (
    _batched_kernel,
    batched_multi_output_to_single_output,
    batched_to_model_list,
    DEPRECATION_MESSAGE,
    model_list_to_batched,
)
from botorch.models.transforms.input import AppendFeatures, Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.models.utils.gpytorch_modules import get_matern_kernel_with_gamma_prior
from botorch.utils.test_helpers import SimpleGPyTorchModel
from botorch.utils.testing import BotorchTestCase
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.likelihoods.gaussian_likelihood import FixedNoiseGaussianLikelihood
from gpytorch.priors import LogNormalPrior


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
            self.assertIsInstance(list_gp.models[0].likelihood, GaussianLikelihood)
            # test observed noise
            batch_gp = SingleTaskGP(train_X, train_Y, torch.rand_like(train_Y))
            with self.assertWarnsRegex(DeprecationWarning, DEPRECATION_MESSAGE):
                list_gp = batched_to_model_list(batch_gp)
            self.assertIsInstance(list_gp, ModelListGP)
            self.assertIsInstance(
                list_gp.models[0].likelihood, FixedNoiseGaussianLikelihood
            )
            # test SingleTaskMultiFidelityGP
            for lin_trunc in (False, True):
                batch_gp = SingleTaskMultiFidelityGP(
                    train_X, train_Y, iteration_fidelity=1, linear_truncated=lin_trunc
                )
                list_gp = batched_to_model_list(batch_gp)
                self.assertIsInstance(list_gp, ModelListGP)
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

    def test_batched_kernel(self):
        covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=3))
        batched_covar_module = _batched_kernel(covar_module, 2)
        self.assertEqual(
            batched_covar_module.base_kernel.lengthscale.shape, torch.Size([2, 1, 3])
        )
        self.assertEqual(batched_covar_module.outputscale.shape, torch.Size([2]))

    def test_model_list_to_batched(self):
        for dtype in (torch.float, torch.double):
            # basic test
            train_X = torch.rand(10, 2, device=self.device, dtype=dtype)
            train_Y1 = train_X.sum(dim=-1, keepdim=True)
            train_Y2 = (train_X[:, 0] - train_X[:, 1]).unsqueeze(-1)
            gp1 = SingleTaskGP(train_X, train_Y1, outcome_transform=None)
            gp2 = SingleTaskGP(train_X, train_Y2, outcome_transform=None)
            list_gp = ModelListGP(gp1, gp2)
            batch_gp = model_list_to_batched(list_gp)
            self.assertIsInstance(batch_gp, SingleTaskGP)
            self.assertIsInstance(batch_gp.likelihood, GaussianLikelihood)
            # test degenerate (single model)
            with self.assertWarnsRegex(DeprecationWarning, DEPRECATION_MESSAGE):
                batch_gp = model_list_to_batched(ModelListGP(gp1))
            self.assertEqual(batch_gp._num_outputs, 1)
            # test mixing different likelihoods
            gp2 = SingleTaskGP(
                train_X, train_Y1, torch.ones_like(train_Y1), outcome_transform=None
            )
            with self.assertRaises(UnsupportedError):
                model_list_to_batched(ModelListGP(gp1, gp2))
            # test non-batched models
            gp1_ = SimpleGPyTorchModel(train_X, train_Y1)
            gp2_ = SimpleGPyTorchModel(train_X, train_Y2)
            with self.assertRaises(UnsupportedError):
                model_list_to_batched(ModelListGP(gp1_, gp2_))
            # test list of multi-output models
            train_Y = torch.cat([train_Y1, train_Y2], dim=-1)
            gp2 = SingleTaskGP(train_X, train_Y, outcome_transform=None)
            with self.assertRaises(UnsupportedError):
                model_list_to_batched(ModelListGP(gp1, gp2))
            # test different training inputs
            gp2 = SingleTaskGP(2 * train_X, train_Y2, outcome_transform=None)
            with self.assertRaises(UnsupportedError):
                model_list_to_batched(ModelListGP(gp1, gp2))
            # check scalar agreement
            # modified to check the scalar agreement in a parameter that is accessible
            # since the error is going to slip through for the non-parametrizable
            # priors regardless (like the LogNormal)
            with self.assertRaises(UnsupportedError):
                model_list_to_batched(ModelListGP(gp1, gp2))

            gp2.likelihood.noise_covar.raw_noise_constraint.lower_bound.fill_(1e-3)
            with self.assertRaises(UnsupportedError):
                model_list_to_batched(ModelListGP(gp1, gp2))
            # check tensor shape agreement
            gp2 = SingleTaskGP(train_X, train_Y2, outcome_transform=None)
            gp2.likelihood.noise_covar.raw_noise = torch.nn.Parameter(
                torch.tensor([[0.42]], device=self.device, dtype=dtype)
            )
            with self.assertRaises(UnsupportedError):
                model_list_to_batched(ModelListGP(gp1, gp2))
            # test custom likelihood
            gp2 = SingleTaskGP(
                train_X,
                train_Y2,
                likelihood=GaussianLikelihood(),
                outcome_transform=None,
            )
            with self.assertRaises(NotImplementedError):
                model_list_to_batched(ModelListGP(gp2))
            # test non-default kernel
            gp1 = SingleTaskGP(
                train_X, train_Y1, covar_module=MaternKernel(), outcome_transform=None
            )
            gp2 = SingleTaskGP(
                train_X, train_Y2, covar_module=MaternKernel(), outcome_transform=None
            )
            list_gp = ModelListGP(gp1, gp2)
            batch_gp = model_list_to_batched(list_gp)
            self.assertEqual(type(batch_gp.covar_module), MaternKernel)
            # test error when component GPs have different kernel types
            # added types for both default and non-default kernels for clarity
            gp1 = SingleTaskGP(
                train_X, train_Y1, covar_module=MaternKernel(), outcome_transform=None
            )
            gp2 = SingleTaskGP(
                train_X, train_Y2, covar_module=RBFKernel(), outcome_transform=None
            )
            list_gp = ModelListGP(gp1, gp2)
            with self.assertRaises(UnsupportedError):
                model_list_to_batched(list_gp)
            # test observed noise
            train_X = torch.rand(10, 2, device=self.device, dtype=dtype)
            train_Y1 = train_X.sum(dim=-1, keepdim=True)
            train_Y2 = (train_X[:, 0] - train_X[:, 1]).unsqueeze(-1)
            gp1_ = SingleTaskGP(
                train_X, train_Y1, torch.rand_like(train_Y1), outcome_transform=None
            )
            gp2_ = SingleTaskGP(
                train_X, train_Y2, torch.rand_like(train_Y2), outcome_transform=None
            )
            list_gp = ModelListGP(gp1_, gp2_)
            batch_gp = model_list_to_batched(list_gp)
            self.assertIsInstance(batch_gp.likelihood, FixedNoiseGaussianLikelihood)
            # test SingleTaskMultiFidelityGP
            gp1_ = SingleTaskMultiFidelityGP(
                train_X,
                train_Y1,
                iteration_fidelity=1,
                outcome_transform=None,
            )
            gp2_ = SingleTaskMultiFidelityGP(
                train_X,
                train_Y2,
                iteration_fidelity=1,
                outcome_transform=None,
            )
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
            gp1_ = SingleTaskGP(
                train_X, train_Y1, input_transform=input_tf, outcome_transform=None
            )
            gp2_ = SingleTaskGP(
                train_X, train_Y2, input_transform=input_tf, outcome_transform=None
            )
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
            gp1_ = SingleTaskGP(
                train_X, train_Y1, input_transform=input_tf3, outcome_transform=None
            )
            gp2_ = SingleTaskGP(
                train_X, train_Y2, input_transform=input_tf3, outcome_transform=None
            )
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
            gp1_ = SingleTaskGP(
                train_X, train_Y1, input_transform=input_tf, outcome_transform=None
            )
            gp2_ = SingleTaskGP(
                train_X, train_Y2, input_transform=input_tf2, outcome_transform=None
            )
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
            gp1_ = SingleTaskGP(
                train_X=train_X.unsqueeze(0),
                train_Y=train_Y1.unsqueeze(0),
                input_transform=input_tf2,
                outcome_transform=None,
            )
            gp2_ = SingleTaskGP(
                train_X=train_X.unsqueeze(0),
                train_Y=train_Y2.unsqueeze(0),
                input_transform=input_tf2,
                outcome_transform=None,
            )
            list_gp = ModelListGP(gp1_, gp2_)
            with self.assertRaisesRegex(
                UnsupportedError, "Batched input_transforms are not supported."
            ):
                model_list_to_batched(list_gp)

            # test outcome transform
            octf = Standardize(m=1)
            gp1_ = SingleTaskGP(train_X, train_Y1, outcome_transform=octf)
            gp2_ = SingleTaskGP(train_X, train_Y2, outcome_transform=octf)
            list_gp = ModelListGP(gp1_, gp2_)
            with self.assertRaises(UnsupportedError):
                model_list_to_batched(list_gp)

    def test_model_list_to_batched_with_legacy_prior(self) -> None:
        train_X = torch.rand(10, 2, device=self.device, dtype=torch.double)
        # and test the old prior for completeness & test coverage
        gp1_gamma = SingleTaskGP(
            train_X,
            train_X.sum(dim=-1, keepdim=True),
            covar_module=get_matern_kernel_with_gamma_prior(train_X.shape[-1]),
        )
        gp2_gamma = SingleTaskGP(
            train_X,
            train_X.sum(dim=-1, keepdim=True),
            covar_module=get_matern_kernel_with_gamma_prior(train_X.shape[-1]),
        )
        gp1_gamma.covar_module.base_kernel.lengthscale_prior.rate.fill_(1.0)
        with self.assertRaises(UnsupportedError):
            model_list_to_batched(ModelListGP(gp1_gamma, gp2_gamma))

    def test_model_list_to_batched_with_different_prior(self) -> None:
        # The goal is to test priors that didn't have their parameters
        # recorded in the state dict prior to GPyTorch #2551.
        train_X = torch.rand(10, 2, device=self.device, dtype=torch.double)
        gp1 = SingleTaskGP(
            train_X=train_X,
            train_Y=train_X.sum(dim=-1, keepdim=True),
            covar_module=RBFKernel(
                ard_num_dims=2, lengthscale_prior=LogNormalPrior(3.0, 6.0)
            ),
            outcome_transform=None,
        )
        gp2 = SingleTaskGP(
            train_X=train_X,
            train_Y=train_X.max(dim=-1, keepdim=True).values,
            covar_module=RBFKernel(
                ard_num_dims=2, lengthscale_prior=LogNormalPrior(2.0, 4.0)
            ),
            outcome_transform=None,
        )
        with self.assertRaisesRegex(
            UnsupportedError, "All scalars must have the same value."
        ):
            model_list_to_batched(ModelListGP(gp1, gp2))

    def test_roundtrip(self):
        for dtype in (torch.float, torch.double):
            train_X = torch.rand(10, 2, device=self.device, dtype=dtype)
            train_Y1 = train_X.sum(dim=-1)
            train_Y2 = train_X[:, 0] - train_X[:, 1]
            train_Y = torch.stack([train_Y1, train_Y2], dim=-1)
            # SingleTaskGP
            batch_gp = SingleTaskGP(train_X, train_Y, outcome_transform=None)
            list_gp = batched_to_model_list(batch_gp)
            batch_gp_recov = model_list_to_batched(list_gp)
            sd_orig = batch_gp.state_dict()
            sd_recov = batch_gp_recov.state_dict()
            self.assertTrue(set(sd_orig) == set(sd_recov))
            self.assertTrue(all(torch.equal(sd_orig[k], sd_recov[k]) for k in sd_orig))
            # Observed noise
            batch_gp = SingleTaskGP(
                train_X, train_Y, torch.rand_like(train_Y), outcome_transform=None
            )
            list_gp = batched_to_model_list(batch_gp)
            batch_gp_recov = model_list_to_batched(list_gp)
            sd_orig = batch_gp.state_dict()
            sd_recov = batch_gp_recov.state_dict()
            self.assertTrue(set(sd_orig) == set(sd_recov))
            self.assertTrue(all(torch.equal(sd_orig[k], sd_recov[k]) for k in sd_orig))
            # SingleTaskMultiFidelityGP
            for lin_trunc in (False, True):
                batch_gp = SingleTaskMultiFidelityGP(
                    train_X=train_X,
                    train_Y=train_Y,
                    iteration_fidelity=1,
                    linear_truncated=lin_trunc,
                    outcome_transform=None,
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
            batched_mo_model = SingleTaskGP(train_X, train_Y, outcome_transform=None)
            with self.assertWarnsRegex(DeprecationWarning, DEPRECATION_MESSAGE):
                batched_so_model = batched_multi_output_to_single_output(
                    batched_mo_model
                )
            self.assertIsInstance(batched_so_model, SingleTaskGP)
            self.assertEqual(batched_so_model.num_outputs, 1)
            # test non-batched models
            non_batch_model = SimpleGPyTorchModel(train_X, train_Y[:, :1])
            with self.assertRaises(UnsupportedError):
                batched_multi_output_to_single_output(non_batch_model)
            # test custom likelihood
            gp2 = SingleTaskGP(train_X, train_Y, likelihood=GaussianLikelihood())
            with self.assertRaises(NotImplementedError):
                batched_multi_output_to_single_output(gp2)
            # test observed noise
            train_X = torch.rand(10, 2, device=self.device, dtype=dtype)
            batched_mo_model = SingleTaskGP(
                train_X, train_Y, torch.rand_like(train_Y), outcome_transform=None
            )
            batched_so_model = batched_multi_output_to_single_output(
                batched_mo_model,
            )
            self.assertIsInstance(batched_so_model, SingleTaskGP)
            self.assertIsInstance(
                batched_so_model.likelihood, FixedNoiseGaussianLikelihood
            )
            self.assertEqual(batched_so_model.num_outputs, 1)
            # test SingleTaskMultiFidelityGP
            batched_mo_model = SingleTaskMultiFidelityGP(
                train_X,
                train_Y,
                iteration_fidelity=1,
                outcome_transform=None,
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
            batched_mo_model = SingleTaskGP(
                train_X, train_Y, input_transform=input_tf, outcome_transform=None
            )
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
                train_X, train_Y, input_transform=input_tf, outcome_transform=None
            ).eval()
            batch_so_model = batched_multi_output_to_single_output(batched_mo_model)
            self.assertIsInstance(batch_so_model.input_transform, AppendFeatures)

            # test batched input transform
            input_tf = Normalize(
                d=2,
                bounds=torch.tensor(
                    [[-1.0, -1.0], [1.0, 1.0]], device=self.device, dtype=dtype
                ),
            )
            batched_mo_model = SingleTaskGP(
                train_X, train_Y, input_transform=input_tf, outcome_transform=None
            )
            batch_so_model = batched_multi_output_to_single_output(batched_mo_model)
            self.assertIsInstance(batch_so_model.input_transform, Normalize)
            self.assertTrue(
                torch.equal(batch_so_model.input_transform.bounds, input_tf.bounds)
            )
            # test outcome transform
            batched_mo_model = SingleTaskGP(
                train_X, train_Y, outcome_transform=Standardize(m=2)
            )
            with self.assertRaisesRegex(
                NotImplementedError,
                "Converting batched multi-output models with outcome transforms",
            ):
                batched_multi_output_to_single_output(batched_mo_model)
