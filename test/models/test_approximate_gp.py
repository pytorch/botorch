#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from botorch.models.approximate_gp import (
    ApproximateGPyTorchModel,
    SingleTaskVariationalGP,
    _SingleTaskVariationalGP,
)
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Log
from botorch.posteriors import GPyTorchPosterior, TransformedPosterior
from botorch.sampling import IIDNormalSampler
from botorch.utils.testing import BotorchTestCase
from gpytorch.likelihoods import GaussianLikelihood, MultitaskGaussianLikelihood
from gpytorch.mlls import VariationalELBO
from gpytorch.variational import (
    IndependentMultitaskVariationalStrategy,
    VariationalStrategy,
)


class TestApproximateGP(BotorchTestCase):
    def setUp(self):
        super().setUp()
        self.train_X = torch.rand(10, 1, device=self.device)
        self.train_Y = torch.sin(self.train_X) + torch.randn_like(self.train_X) * 0.2

    def test_initialization(self):
        # test non batch case
        model = ApproximateGPyTorchModel(train_X=self.train_X, train_Y=self.train_Y)
        self.assertIsInstance(model.model, _SingleTaskVariationalGP)
        self.assertIsInstance(model.likelihood, GaussianLikelihood)
        self.assertIsInstance(model.model.variational_strategy, VariationalStrategy)
        self.assertEqual(model.num_outputs, 1)

        # test batch case
        stacked_y = torch.cat((self.train_Y, self.train_Y), dim=-1)
        model = ApproximateGPyTorchModel(
            train_X=self.train_X, train_Y=stacked_y, num_outputs=2
        )
        self.assertIsInstance(model.model, _SingleTaskVariationalGP)
        self.assertIsInstance(model.likelihood, MultitaskGaussianLikelihood)
        self.assertIsInstance(
            model.model.variational_strategy, IndependentMultitaskVariationalStrategy
        )
        self.assertEqual(model.num_outputs, 2)


class TestSingleTaskVariationalGP(BotorchTestCase):
    def setUp(self):
        super().setUp()
        train_X = torch.rand(10, 1, device=self.device)
        train_y = torch.sin(train_X) + torch.randn_like(train_X) * 0.2

        self.model = SingleTaskVariationalGP(
            train_X=train_X, likelihood=GaussianLikelihood()
        ).to(self.device)

        mll = VariationalELBO(self.model.likelihood, self.model.model, num_data=10)
        loss = -mll(self.model.likelihood(self.model(train_X)), train_y).sum()
        loss.backward()

    def test_posterior(self):
        # basic test of checking that the posterior works as intended
        test_x = torch.rand(30, 1, device=self.device)
        posterior = self.model.posterior(test_x)
        self.assertIsInstance(posterior, GPyTorchPosterior)

        posterior = self.model.posterior(test_x, observation_noise=True)
        self.assertIsInstance(posterior, GPyTorchPosterior)

        # now loop through all possibilities
        train_X = torch.rand(3, 10, 1, device=self.device)
        train_Y = torch.randn(3, 10, 2, device=self.device)
        test_X = torch.rand(3, 5, 1, device=self.device)

        non_batched = [train_X[0], train_Y[0, :, 0].unsqueeze(-1), test_X[0]]
        non_batched_mo = [train_X[0], train_Y[0], test_X[0]]
        batched = [train_X, train_Y[..., 0].unsqueeze(-1), test_X]
        # batched multi-output is not supported at this time
        # batched_mo = [train_X, train_Y, test_X]
        non_batched_to_batched = [train_X[0], train_Y[0], test_X]
        all_test_lists = [non_batched, non_batched_mo, batched, non_batched_to_batched]

        for [tx, ty, test] in all_test_lists:
            print(tx.shape, ty.shape, test.shape)
            model = SingleTaskVariationalGP(tx, ty, inducing_points=tx)
            posterior = model.posterior(test)
            self.assertIsInstance(posterior, GPyTorchPosterior)

    def test_variational_setUp(self):
        for dtype in [torch.float, torch.double]:
            train_X = torch.rand(10, 1, device=self.device, dtype=dtype)
            train_y = torch.randn(10, 3, device=self.device, dtype=dtype)

            for ty, num_out in [[train_y, 3], [train_y, 1], [None, 3]]:
                batched_model = SingleTaskVariationalGP(
                    train_X,
                    train_Y=ty,
                    num_outputs=num_out,
                    learn_inducing_points=False,
                ).to(self.device)
                mll = VariationalELBO(
                    batched_model.likelihood, batched_model.model, num_data=10
                )

                with torch.enable_grad():
                    loss = -mll(
                        batched_model.likelihood(batched_model(train_X)), train_y
                    ).sum()
                    loss.backward()

                # ensure that inducing points do not require grad
                model_var_strat = batched_model.model.variational_strategy
                self.assertEqual(
                    model_var_strat.base_variational_strategy.inducing_points.grad,
                    None,
                )

                # but that the covariance does have a gradient
                self.assertIsNotNone(
                    batched_model.model.covar_module.raw_outputscale.grad
                )

                # check that we always have three outputs
                self.assertEqual(batched_model._num_outputs, 3)
                self.assertIsInstance(
                    batched_model.likelihood, MultitaskGaussianLikelihood
                )

    def test_likelihood_and_fantasize(self):
        self.assertIsInstance(self.model.likelihood, GaussianLikelihood)
        self.assertTrue(self.model._is_custom_likelihood, True)

        test_X = torch.randn(5, 1, device=self.device)

        with self.assertRaises(NotImplementedError):
            self.model.fantasize(test_X, sampler=IIDNormalSampler(num_samples=32))

    def test_initializations(self):
        train_X = torch.rand(25, 1, device=self.device)
        train_Y = torch.rand(25, 1, device=self.device)

        stacked_train_X = torch.cat((train_X, train_X), dim=0)
        for X, num_ind in [[train_X, 10], [stacked_train_X, 30], [stacked_train_X, 10]]:
            model = SingleTaskVariationalGP(train_X=X, inducing_points=num_ind)
            if num_ind == 10:
                self.assertLessEqual(
                    model.model.variational_strategy.inducing_points.shape,
                    torch.Size((10, 1)),
                )
            else:
                # should not have 30 inducing points when 25 singular dimensions
                # are passed
                self.assertLess(
                    model.model.variational_strategy.inducing_points.shape[-2], num_ind
                )

        # test that only piv cholesky init is supported
        with self.assertRaises(AssertionError):
            model = SingleTaskVariationalGP(
                train_X=X, inducing_points=10, init_method="rand"
            )

        test_X = torch.rand(5, 1, device=self.device)

        # test transforms
        for inp_trans in [None, Normalize(d=1)]:
            for out_trans in [None, Log()]:
                model = SingleTaskVariationalGP(
                    train_X=train_X,
                    train_Y=train_Y,
                    outcome_transform=out_trans,
                    input_transform=inp_trans,
                )

                if inp_trans is not None:
                    self.assertIsInstance(model.input_transform, Normalize)
                else:
                    self.assertFalse(hasattr(model, "input_transform"))
                if out_trans is not None:
                    self.assertIsInstance(model.outcome_transform, Log)

                    posterior = model.posterior(test_X)
                    self.assertIsInstance(posterior, TransformedPosterior)
                else:
                    self.assertFalse(hasattr(model, "outcome_transform"))
