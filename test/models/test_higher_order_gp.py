#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import itertools

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
from numpy import prod
from torch import Size, allclose, rand, randn
from torch.random import manual_seed


class TestHigherOrderGP(BotorchTestCase):
    def setUp(self):
        super().setUp()
        manual_seed(0)

        train_x = rand(2, 10, 1, device=self.device)
        train_y = randn(2, 10, 3, 5, device=self.device)

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
        train_x = rand(2, 10, 1, device=self.device)
        train_y = randn(2, 10, 3, 5, device=self.device)
        model = HigherOrderGP(train_x, train_y)

        # check that it correctly inferred that this is a batched model
        self.assertEqual(model._num_outputs, 2)

        train_x = rand(10, 1, device=self.device)
        train_y = randn(10, 3, 5, 2, device=self.device)
        model = HigherOrderGP(train_x, train_y)

        # non-batched case
        self.assertEqual(model._num_outputs, 1)

        train_x = rand(3, 2, 10, 1, device=self.device)
        train_y = randn(3, 2, 10, 3, 5, device=self.device)

        # check the error when using multi-dim batch_shape
        with self.assertRaises(NotImplementedError):
            model = HigherOrderGP(train_x, train_y)

    def test_posterior(self):
        manual_seed(0)
        test_x = rand(2, 30, 1).to(device=self.device)

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
        train_x = rand(10, 3, device=self.device)
        train_y = randn(10, 4, 5, device=self.device)

        # test handling of Standardize
        with self.assertWarns(RuntimeWarning):
            model = HigherOrderGP(
                train_X=train_x, train_Y=train_y, outcome_transform=Standardize(m=5)
            )
        self.assertIsInstance(model.outcome_transform, FlattenedStandardize)
        self.assertEqual(model.outcome_transform.output_shape, train_y.shape[1:])
        self.assertEqual(model.outcome_transform.batch_shape, Size())

        model = HigherOrderGP(
            train_X=train_x,
            train_Y=train_y,
            input_transform=Normalize(d=3),
            outcome_transform=FlattenedStandardize(train_y.shape[1:]),
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_torch(mll, options={"maxiter": 1, "disp": False})

        test_x = rand(2, 5, 3, device=self.device)
        test_y = randn(2, 5, 4, 5, device=self.device)
        posterior = model.posterior(test_x)
        self.assertIsInstance(posterior, TransformedPosterior)

        conditioned_model = model.condition_on_observations(test_x, test_y)
        self.assertIsInstance(conditioned_model, HigherOrderGP)

        self.check_transform_forward(model)
        self.check_transform_untransform(model)

    def check_transform_forward(self, model):
        train_y = randn(2, 10, 4, 5, device=self.device)
        train_y_var = rand(2, 10, 4, 5, device=self.device)

        output, output_var = model.outcome_transform.forward(train_y)
        self.assertEqual(output.shape, Size((2, 10, 4, 5)))
        self.assertEqual(output_var, None)

        output, output_var = model.outcome_transform.forward(train_y, train_y_var)
        self.assertEqual(output.shape, Size((2, 10, 4, 5)))
        self.assertEqual(output_var.shape, Size((2, 10, 4, 5)))

    def check_transform_untransform(self, model):
        output, output_var = model.outcome_transform.untransform(
            randn(2, 2, 4, 5, device=self.device)
        )
        self.assertEqual(output.shape, Size((2, 2, 4, 5)))
        self.assertEqual(output_var, None)

        output, output_var = model.outcome_transform.untransform(
            randn(2, 2, 4, 5, device=self.device),
            rand(2, 2, 4, 5, device=self.device),
        )
        self.assertEqual(output.shape, Size((2, 2, 4, 5)))
        self.assertEqual(output_var.shape, Size((2, 2, 4, 5)))

    def test_condition_on_observations(self):
        manual_seed(0)
        test_x = rand(2, 5, 1, device=self.device)
        test_y = randn(2, 5, 3, 5, device=self.device)

        # dummy call to ensure caches have been computed
        _ = self.model.posterior(test_x)
        conditioned_model = self.model.condition_on_observations(test_x, test_y)
        self.assertIsInstance(conditioned_model, HigherOrderGP)

    def test_fantasize(self):
        manual_seed(0)
        test_x = rand(2, 5, 1, device=self.device)
        sampler = IIDNormalSampler(num_samples=32).to(self.device)

        _ = self.model.posterior(test_x)
        fantasy_model = self.model.fantasize(test_x, sampler=sampler)
        self.assertIsInstance(fantasy_model, HigherOrderGP)
        self.assertEqual(fantasy_model.train_inputs[0].shape[:2], Size((32, 2)))

    def test_initialize_latents(self):
        manual_seed(0)

        train_x = rand(10, 1, device=self.device)
        train_y = randn(10, 3, 5, device=self.device)

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
                Size((3, latent_dim_sizes[0])),
            )
            self.assertEqual(
                self.model.latent_parameters[1].shape,
                Size((5, latent_dim_sizes[1])),
            )


class TestHigherOrderGPPosterior(BotorchTestCase):
    def setUp(self):
        super().setUp()
        manual_seed(0)

        train_x = rand(2, 10, 1, device=self.device)
        train_y = randn(2, 10, 3, 5, device=self.device)

        m1 = HigherOrderGP(train_x, train_y)
        m2 = HigherOrderGP(train_x[0], train_y[0])

        manual_seed(0)
        test_x = rand(2, 5, 1, device=self.device)

        posterior1 = m1.posterior(test_x)
        posterior2 = m2.posterior(test_x[0])
        posterior3 = m2.posterior(test_x)

        self.post_list = [
            [m1, test_x, posterior1],
            [m2, test_x[0], posterior2],
            [m2, test_x, posterior3],
        ]

    def test_posterior(self):
        # test the posterior works
        sample_shaping = [5, 3, 5]

        for post_collection in self.post_list:
            model, test_x, posterior = post_collection

            self.assertIsInstance(posterior, GPyTorchPosterior)

            correct_shape = [2] if test_x.shape[0] == 2 else []
            [correct_shape.append(s) for s in sample_shaping]

            # test providing no base samples
            samples_0 = posterior.rsample()
            self.assertEqual(samples_0.shape, Size((1, *correct_shape)))

            # test that providing all base samples produces non-random results
            if test_x.shape[0] == 2:
                base_samples = randn(8, 2, (5 + 10 + 10) * 3 * 5, device=self.device)
            else:
                base_samples = randn(8, (5 + 10 + 10) * 3 * 5, device=self.device)

            samples_1 = posterior.rsample(
                base_samples=base_samples, sample_shape=Size((8,))
            )
            samples_2 = posterior.rsample(
                base_samples=base_samples, sample_shape=Size((8,))
            )
            self.assertTrue(allclose(samples_1, samples_2))

            # test that botorch.sampler picks up the correct shapes
            sampler = IIDNormalSampler(num_samples=5)
            samples_det_shape = sampler(posterior).shape
            self.assertEqual(samples_det_shape, Size([5, *correct_shape]))

            # test that providing only some base samples is okay
            base_samples = randn(8, prod(correct_shape), device=self.device)
            samples_3 = posterior.rsample(
                base_samples=base_samples, sample_shape=Size((8,))
            )
            self.assertEqual(samples_3.shape, Size([8, *correct_shape]))

            # test that providing the wrong number base samples is okay
            base_samples = randn(8, 50 * 2 * 3 * 5, device=self.device)
            samples_4 = posterior.rsample(
                base_samples=base_samples, sample_shape=Size((8,))
            )
            self.assertEqual(samples_4.shape, Size([8, *correct_shape]))

            # test that providing the wrong shapes of base samples fails
            base_samples = randn(8, 5 * 2 * 3 * 5, device=self.device)
            with self.assertRaises(RuntimeError):
                samples_4 = posterior.rsample(
                    base_samples=base_samples, sample_shape=Size((4,))
                )

            # finally we check the quality of the variances and the samples
            # test that the posterior variances are the same as the evaluation variance
            posterior_variance = posterior.variance

            model.eval()
            eval_mode_variance = model(test_x).variance.reshape_as(posterior_variance)
            self.assertLess(
                (posterior_variance - eval_mode_variance).norm()
                / eval_mode_variance.norm(),
                4e-2,
            )

            # and finally test that sampling with no base samples is okay
            samples_3 = posterior.rsample(sample_shape=Size((5000,)))
            sampled_variance = samples_3.var(dim=0).view(-1)
            posterior_variance = posterior_variance.view(-1)
            self.assertLess(
                (posterior_variance - sampled_variance).norm()
                / posterior_variance.norm(),
                5e-2,
            )
