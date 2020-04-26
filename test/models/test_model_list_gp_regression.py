#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import warnings

import torch
from botorch.exceptions.errors import BotorchTensorDimensionError
from botorch.exceptions.warnings import OptimizationWarning
from botorch.fit import fit_gpytorch_model
from botorch.models import ModelListGP
from botorch.models.gp_regression import FixedNoiseGP, SingleTaskGP
from botorch.models.transforms import Standardize
from botorch.posteriors import GPyTorchPosterior
from botorch.utils.testing import BotorchTestCase, _get_random_data
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import LikelihoodList
from gpytorch.means import ConstantMean
from gpytorch.mlls import SumMarginalLogLikelihood
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.priors import GammaPrior


def _get_model(n, fixed_noise=False, use_octf=False, **tkwargs):
    train_x1, train_y1 = _get_random_data(
        batch_shape=torch.Size(), num_outputs=1, n=10, **tkwargs
    )
    train_x2, train_y2 = _get_random_data(
        batch_shape=torch.Size(), num_outputs=1, n=11, **tkwargs
    )
    octfs = [Standardize(m=1), Standardize(m=1)] if use_octf else [None, None]
    if fixed_noise:
        train_y1_var = 0.1 + 0.1 * torch.rand_like(train_y1, **tkwargs)
        train_y2_var = 0.1 + 0.1 * torch.rand_like(train_y2, **tkwargs)
        model1 = FixedNoiseGP(
            train_X=train_x1,
            train_Y=train_y1,
            train_Yvar=train_y1_var,
            outcome_transform=octfs[0],
        )
        model2 = FixedNoiseGP(
            train_X=train_x2,
            train_Y=train_y2,
            train_Yvar=train_y2_var,
            outcome_transform=octfs[1],
        )
    else:
        model1 = SingleTaskGP(
            train_X=train_x1, train_Y=train_y1, outcome_transform=octfs[0]
        )
        model2 = SingleTaskGP(
            train_X=train_x2, train_Y=train_y2, outcome_transform=octfs[1]
        )
    model = ModelListGP(model1, model2)
    return model.to(**tkwargs)


class TestModelListGP(BotorchTestCase):
    def test_ModelListGP(self):
        for dtype, use_octf in itertools.product(
            (torch.float, torch.double), (False, True)
        ):
            tkwargs = {"device": self.device, "dtype": dtype}
            model = _get_model(n=10, use_octf=use_octf, **tkwargs)
            self.assertIsInstance(model, ModelListGP)
            self.assertIsInstance(model.likelihood, LikelihoodList)
            for m in model.models:
                self.assertIsInstance(m.mean_module, ConstantMean)
                self.assertIsInstance(m.covar_module, ScaleKernel)
                matern_kernel = m.covar_module.base_kernel
                self.assertIsInstance(matern_kernel, MaternKernel)
                self.assertIsInstance(matern_kernel.lengthscale_prior, GammaPrior)
                if use_octf:
                    self.assertIsInstance(m.outcome_transform, Standardize)

            # test constructing likelihood wrapper
            mll = SumMarginalLogLikelihood(model.likelihood, model)
            for mll_ in mll.mlls:
                self.assertIsInstance(mll_, ExactMarginalLogLikelihood)

            # test model fitting (sequential)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=OptimizationWarning)
                mll = fit_gpytorch_model(mll, options={"maxiter": 1}, max_retries=1)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=OptimizationWarning)
                # test model fitting (joint)
                mll = fit_gpytorch_model(
                    mll, options={"maxiter": 1}, max_retries=1, sequential=False
                )

            # test subset outputs
            subset_model = model.subset_output([1])
            self.assertIsInstance(subset_model, ModelListGP)
            self.assertEqual(len(subset_model.models), 1)
            sd_subset = subset_model.models[0].state_dict()
            sd = model.models[1].state_dict()
            self.assertTrue(set(sd_subset.keys()) == set(sd.keys()))
            self.assertTrue(all(torch.equal(v, sd[k]) for k, v in sd_subset.items()))

            # test posterior
            test_x = torch.tensor([[0.25], [0.75]], **tkwargs)
            posterior = model.posterior(test_x)
            self.assertIsInstance(posterior, GPyTorchPosterior)
            self.assertIsInstance(posterior.mvn, MultitaskMultivariateNormal)
            if use_octf:
                # ensure un-transformation is applied
                submodel = model.models[0]
                p0 = submodel.posterior(test_x)
                tmp_tf = submodel.outcome_transform
                del submodel.outcome_transform
                p0_tf = submodel.posterior(test_x)
                submodel.outcome_transform = tmp_tf
                expected_var = tmp_tf.untransform_posterior(p0_tf).variance
                self.assertTrue(torch.allclose(p0.variance, expected_var))

            # test observation_noise
            posterior = model.posterior(test_x, observation_noise=True)
            self.assertIsInstance(posterior, GPyTorchPosterior)
            self.assertIsInstance(posterior.mvn, MultitaskMultivariateNormal)

            # test output_indices
            posterior = model.posterior(
                test_x, output_indices=[0], observation_noise=True
            )
            self.assertIsInstance(posterior, GPyTorchPosterior)
            self.assertIsInstance(posterior.mvn, MultivariateNormal)

            # test condition_on_observations
            f_x = torch.rand(2, 1, **tkwargs)
            f_y = torch.rand(2, 2, **tkwargs)
            cm = model.condition_on_observations(f_x, f_y)
            self.assertIsInstance(cm, ModelListGP)

            # test condition_on_observations batched
            f_x = torch.rand(3, 2, 1, **tkwargs)
            f_y = torch.rand(3, 2, 2, **tkwargs)
            cm = model.condition_on_observations(f_x, f_y)
            self.assertIsInstance(cm, ModelListGP)

            # test condition_on_observations batched (fast fantasies)
            f_x = torch.rand(2, 1, **tkwargs)
            f_y = torch.rand(3, 2, 2, **tkwargs)
            cm = model.condition_on_observations(f_x, f_y)
            self.assertIsInstance(cm, ModelListGP)

            # test condition_on_observations (incorrect input shape error)
            with self.assertRaises(BotorchTensorDimensionError):
                model.condition_on_observations(f_x, torch.rand(3, 2, 3, **tkwargs))

    def test_ModelListGP_fixed_noise(self):
        for dtype, use_octf in itertools.product(
            (torch.float, torch.double), (False, True)
        ):
            tkwargs = {"device": self.device, "dtype": dtype}
            model = _get_model(n=10, fixed_noise=True, use_octf=use_octf, **tkwargs)
            self.assertIsInstance(model, ModelListGP)
            self.assertIsInstance(model.likelihood, LikelihoodList)
            for m in model.models:
                self.assertIsInstance(m.mean_module, ConstantMean)
                self.assertIsInstance(m.covar_module, ScaleKernel)
                matern_kernel = m.covar_module.base_kernel
                self.assertIsInstance(matern_kernel, MaternKernel)
                self.assertIsInstance(matern_kernel.lengthscale_prior, GammaPrior)

            # test model fitting
            mll = SumMarginalLogLikelihood(model.likelihood, model)
            for mll_ in mll.mlls:
                self.assertIsInstance(mll_, ExactMarginalLogLikelihood)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=OptimizationWarning)
                mll = fit_gpytorch_model(mll, options={"maxiter": 1}, max_retries=1)

            # test posterior
            test_x = torch.tensor([[0.25], [0.75]], **tkwargs)
            posterior = model.posterior(test_x)
            self.assertIsInstance(posterior, GPyTorchPosterior)
            self.assertIsInstance(posterior.mvn, MultitaskMultivariateNormal)
            if use_octf:
                # ensure un-transformation is applied
                submodel = model.models[0]
                p0 = submodel.posterior(test_x)
                tmp_tf = submodel.outcome_transform
                del submodel.outcome_transform
                p0_tf = submodel.posterior(test_x)
                submodel.outcome_transform = tmp_tf
                expected_var = tmp_tf.untransform_posterior(p0_tf).variance
                self.assertTrue(torch.allclose(p0.variance, expected_var))

            # test output_indices
            posterior = model.posterior(
                test_x, output_indices=[0], observation_noise=True
            )
            self.assertIsInstance(posterior, GPyTorchPosterior)
            self.assertIsInstance(posterior.mvn, MultivariateNormal)

            # test condition_on_observations
            f_x = torch.rand(2, 1, **tkwargs)
            f_y = torch.rand(2, 2, **tkwargs)
            noise = 0.1 + 0.1 * torch.rand_like(f_y)
            cm = model.condition_on_observations(f_x, f_y, noise=noise)
            self.assertIsInstance(cm, ModelListGP)

            # test condition_on_observations batched
            f_x = torch.rand(3, 2, 1, **tkwargs)
            f_y = torch.rand(3, 2, 2, **tkwargs)
            noise = 0.1 + 0.1 * torch.rand_like(f_y)
            cm = model.condition_on_observations(f_x, f_y, noise=noise)
            self.assertIsInstance(cm, ModelListGP)

            # test condition_on_observations batched (fast fantasies)
            f_x = torch.rand(2, 1, **tkwargs)
            f_y = torch.rand(3, 2, 2, **tkwargs)
            noise = 0.1 + 0.1 * torch.rand(2, 2, **tkwargs)
            cm = model.condition_on_observations(f_x, f_y, noise=noise)
            self.assertIsInstance(cm, ModelListGP)

            # test condition_on_observations (incorrect input shape error)
            with self.assertRaises(BotorchTensorDimensionError):
                model.condition_on_observations(
                    f_x, torch.rand(3, 2, 3, **tkwargs), noise=noise
                )
            # test condition_on_observations (incorrect noise shape error)
            f_y = torch.rand(2, 2, **tkwargs)
            with self.assertRaises(BotorchTensorDimensionError):
                model.condition_on_observations(
                    f_x, f_y, noise=torch.rand(2, 3, **tkwargs)
                )

    def test_ModelListGP_single(self):
        tkwargs = {"device": self.device, "dtype": torch.float}
        train_x1, train_y1 = _get_random_data(
            batch_shape=torch.Size(), num_outputs=1, n=10, **tkwargs
        )
        train_x2, train_y2 = _get_random_data(
            batch_shape=torch.Size(), num_outputs=1, n=11, **tkwargs
        )
        model1 = SingleTaskGP(train_X=train_x1, train_Y=train_y1)
        model = ModelListGP(model1)
        model.to(**tkwargs)
        test_x = torch.tensor([[0.25], [0.75]], **tkwargs)
        posterior = model.posterior(test_x)
        self.assertIsInstance(posterior, GPyTorchPosterior)
        self.assertIsInstance(posterior.mvn, MultivariateNormal)
