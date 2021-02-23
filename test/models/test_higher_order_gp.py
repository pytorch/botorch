#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools

import torch
from botorch.models import HigherOrderGP
from botorch.models.higher_order_gp import FlattenedStandardize
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.optim.fit import fit_gpytorch_torch
from botorch.posteriors import GPyTorchPosterior, TransformedPosterior
from botorch.sampling import IIDNormalSampler
from botorch.utils.testing import BotorchTestCase
from gpytorch.kernels import RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.settings import skip_posterior_variances


class TestHigherOrderGP(BotorchTestCase):
    def setUp(self):
        super().setUp()
        torch.random.manual_seed(0)

        train_x = torch.rand(2, 10, 1, device=self.device)
        train_y = torch.randn(2, 10, 3, 5, device=self.device)

        self.model = HigherOrderGP(train_x, train_y)

        # check that we can assign different kernels and likelihoods
        model_2 = HigherOrderGP(
            train_X=train_x,
            train_Y=train_y,
            covar_modules=[RBFKernel(), RBFKernel(), RBFKernel()],
            likelihood=GaussianLikelihood(),
        )

        for m in [self.model, model_2]:
            mll = ExactMarginalLogLikelihood(m.likelihood, m)
            fit_gpytorch_torch(mll, options={"maxiter": 1, "disp": False})

    def test_num_output_dims(self):
        train_x = torch.rand(2, 10, 1, device=self.device)
        train_y = torch.randn(2, 10, 3, 5, device=self.device)
        model = HigherOrderGP(train_x, train_y)

        # check that it correctly inferred that this is a batched model
        self.assertEqual(model._num_outputs, 2)

        train_x = torch.rand(10, 1, device=self.device)
        train_y = torch.randn(10, 3, 5, 2, device=self.device)
        model = HigherOrderGP(train_x, train_y)

        # non-batched case
        self.assertEqual(model._num_outputs, 1)

        train_x = torch.rand(3, 2, 10, 1, device=self.device)
        train_y = torch.randn(3, 2, 10, 3, 5, device=self.device)

        # check the error when using multi-dim batch_shape
        with self.assertRaises(NotImplementedError):
            model = HigherOrderGP(train_x, train_y)

    def test_posterior(self):
        torch.random.manual_seed(0)
        test_x = torch.rand(2, 30, 1).to(device=self.device)

        # test the posterior works
        posterior = self.model.posterior(test_x)
        self.assertIsInstance(posterior, GPyTorchPosterior)

        # test the posterior works with observation noise
        posterior = self.model.posterior(test_x, observation_noise=True)
        self.assertIsInstance(posterior, GPyTorchPosterior)

        # test the posterior works with no variances
        # some funkiness in MVNs registration so the variance is non-zero.
        with skip_posterior_variances():
            posterior = self.model.posterior(test_x)
            self.assertIsInstance(posterior, GPyTorchPosterior)
            self.assertLessEqual(posterior.variance.max(), 1e-6)

    def test_transforms(self):
        train_x = torch.rand(10, 3, device=self.device)
        train_y = torch.randn(10, 4, 5, device=self.device)

        # test handling of Standardize
        with self.assertWarns(RuntimeWarning):
            model = HigherOrderGP(
                train_X=train_x, train_Y=train_y, outcome_transform=Standardize(m=5)
            )
        self.assertIsInstance(model.outcome_transform, FlattenedStandardize)
        self.assertEqual(model.outcome_transform.output_shape, train_y.shape[1:])
        self.assertEqual(model.outcome_transform.batch_shape, torch.Size())

        model = HigherOrderGP(
            train_X=train_x,
            train_Y=train_y,
            input_transform=Normalize(d=3),
            outcome_transform=FlattenedStandardize(train_y.shape[1:]),
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_torch(mll, options={"maxiter": 1, "disp": False})

        test_x = torch.rand(2, 5, 3, device=self.device)
        test_y = torch.randn(2, 5, 4, 5, device=self.device)
        posterior = model.posterior(test_x)
        self.assertIsInstance(posterior, TransformedPosterior)

        conditioned_model = model.condition_on_observations(test_x, test_y)
        self.assertIsInstance(conditioned_model, HigherOrderGP)

        self.check_transform_forward(model)
        self.check_transform_untransform(model)

    def check_transform_forward(self, model):
        train_y = torch.randn(2, 10, 4, 5, device=self.device)
        train_y_var = torch.rand(2, 10, 4, 5, device=self.device)

        output, output_var = model.outcome_transform.forward(train_y)
        self.assertEqual(output.shape, torch.Size((2, 10, 4, 5)))
        self.assertEqual(output_var, None)

        output, output_var = model.outcome_transform.forward(train_y, train_y_var)
        self.assertEqual(output.shape, torch.Size((2, 10, 4, 5)))
        self.assertEqual(output_var.shape, torch.Size((2, 10, 4, 5)))

    def check_transform_untransform(self, model):
        output, output_var = model.outcome_transform.untransform(
            torch.randn(2, 2, 4, 5, device=self.device)
        )
        self.assertEqual(output.shape, torch.Size((2, 2, 4, 5)))
        self.assertEqual(output_var, None)

        output, output_var = model.outcome_transform.untransform(
            torch.randn(2, 2, 4, 5, device=self.device),
            torch.rand(2, 2, 4, 5, device=self.device),
        )
        self.assertEqual(output.shape, torch.Size((2, 2, 4, 5)))
        self.assertEqual(output_var.shape, torch.Size((2, 2, 4, 5)))

    def test_condition_on_observations(self):
        torch.random.manual_seed(0)
        test_x = torch.rand(2, 5, 1, device=self.device)
        test_y = torch.randn(2, 5, 3, 5, device=self.device)

        # dummy call to ensure caches have been computed
        _ = self.model.posterior(test_x)
        conditioned_model = self.model.condition_on_observations(test_x, test_y)
        self.assertIsInstance(conditioned_model, HigherOrderGP)

    def test_fantasize(self):
        torch.random.manual_seed(0)
        test_x = torch.rand(2, 5, 1, device=self.device)
        sampler = IIDNormalSampler(num_samples=32).to(self.device)

        _ = self.model.posterior(test_x)
        fantasy_model = self.model.fantasize(test_x, sampler=sampler)
        self.assertIsInstance(fantasy_model, HigherOrderGP)
        self.assertEqual(fantasy_model.train_inputs[0].shape[:2], torch.Size((32, 2)))

    def test_initialize_latents(self):
        torch.random.manual_seed(0)

        train_x = torch.rand(10, 1, device=self.device)
        train_y = torch.randn(10, 3, 5, device=self.device)

        for latent_dim_sizes, latent_init in itertools.product(
            [[1, 1], [2, 3]],
            ["gp", "default"],
        ):
            self.model = HigherOrderGP(
                train_x,
                train_y,
                num_latent_dims=latent_dim_sizes,
                latent_init=latent_init,
            )
            self.assertEqual(
                self.model.latent_parameters[0].shape,
                torch.Size((3, latent_dim_sizes[0])),
            )
            self.assertEqual(
                self.model.latent_parameters[1].shape,
                torch.Size((5, latent_dim_sizes[1])),
            )
