#! /usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import math
import unittest

import torch
from botorch import fit_gpytorch_model
from botorch.models.gp_regression import (
    FixedNoiseGP,
    HeteroskedasticSingleTaskGP,
    SingleTaskGP,
)
from botorch.posteriors import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import (
    FixedNoiseGaussianLikelihood,
    GaussianLikelihood,
    HeteroskedasticNoise,
    _GaussianLikelihoodBase,
)
from gpytorch.means import ConstantMean
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.priors import GammaPrior


def _get_random_data(batch_shape, num_outputs, n=10, **tkwargs):
    train_x = torch.linspace(0, 0.95, n, **tkwargs).unsqueeze(-1) + 0.05 * torch.rand(
        n, 1, **tkwargs
    ).repeat(batch_shape + torch.Size([1, 1]))
    train_y = torch.sin(train_x * (2 * math.pi)) + 0.2 * torch.randn(
        n, num_outputs, **tkwargs
    ).repeat(batch_shape + torch.Size([1, 1]))

    if num_outputs == 1:
        train_y = train_y.squeeze(-1)
    return train_x, train_y


class TestSingleTaskGP(unittest.TestCase):
    def _get_model_and_data(self, batch_shape, num_outputs, **tkwargs):
        train_X, train_Y = _get_random_data(
            batch_shape=batch_shape, num_outputs=num_outputs, **tkwargs
        )
        model_kwargs = {"train_X": train_X, "train_Y": train_Y}
        model = SingleTaskGP(**model_kwargs)
        return model, model_kwargs

    def test_gp(self, cuda=False):
        for batch_shape in (torch.Size([]), torch.Size([2])):
            for num_outputs in (1, 2):
                for double in (False, True):
                    tkwargs = {
                        "device": torch.device("cuda") if cuda else torch.device("cpu"),
                        "dtype": torch.double if double else torch.float,
                    }
                    model, _ = self._get_model_and_data(
                        batch_shape=batch_shape, num_outputs=num_outputs, **tkwargs
                    )
                    mll = ExactMarginalLogLikelihood(model.likelihood, model).to(
                        **tkwargs
                    )
                    fit_gpytorch_model(mll, options={"maxiter": 1})
                    # test init
                    self.assertIsInstance(model.mean_module, ConstantMean)
                    self.assertIsInstance(model.covar_module, ScaleKernel)
                    matern_kernel = model.covar_module.base_kernel
                    self.assertIsInstance(matern_kernel, MaternKernel)
                    self.assertIsInstance(matern_kernel.lengthscale_prior, GammaPrior)

                    # Test forward
                    test_x = torch.rand(batch_shape + torch.Size([3, 1]), **tkwargs)
                    posterior = model(test_x)
                    self.assertIsInstance(posterior, MultivariateNormal)

                    # test param sizes
                    params = dict(model.named_parameters())
                    for p in params:
                        self.assertEqual(
                            params[p].numel(),
                            num_outputs * torch.tensor(batch_shape).prod().item(),
                        )

                    # test posterior
                    # test non batch evaluation
                    X = torch.rand(batch_shape + torch.Size([3, 1]), **tkwargs)
                    posterior = model.posterior(X)
                    self.assertIsInstance(posterior, GPyTorchPosterior)
                    self.assertEqual(
                        posterior.mean.shape, batch_shape + torch.Size([3, num_outputs])
                    )
                    # test batch evaluation
                    X = torch.rand(
                        torch.Size([2]) + batch_shape + torch.Size([3, 1]), **tkwargs
                    )
                    posterior = model.posterior(X)
                    self.assertIsInstance(posterior, GPyTorchPosterior)
                    self.assertEqual(
                        posterior.mean.shape,
                        torch.Size([2]) + batch_shape + torch.Size([3, num_outputs]),
                    )

    def test_gp_cuda(self):
        if torch.cuda.is_available():
            self.test_gp(cuda=True)

    def test_get_fantasy_model(self, cuda=False):
        for batch_shape in (torch.Size([]), torch.Size([2])):
            for num_outputs in (1, 2):
                for double in (False, True):
                    tkwargs = {
                        "device": torch.device("cuda") if cuda else torch.device("cpu"),
                        "dtype": torch.double if double else torch.float,
                    }
                    model, model_kwargs = self._get_model_and_data(
                        batch_shape=batch_shape, num_outputs=num_outputs, **tkwargs
                    )
                    # evaluate model
                    model.posterior(torch.rand(torch.Size([4, 1]), **tkwargs))
                    # test get_fantasy_model
                    fant_shape = torch.Size([2])
                    # fantasize at different input points
                    X_fant, Y_fant = _get_random_data(
                        fant_shape + batch_shape, num_outputs, n=3, **tkwargs
                    )
                    fant_kwargs = (
                        {"noise": torch.full_like(Y_fant, 0.01)}
                        if isinstance(model, FixedNoiseGP)
                        else {}
                    )
                    fm = model.get_fantasy_model(X_fant, Y_fant, **fant_kwargs)
                    # fantasize at different same input points
                    fant_kwargs_same_inputs = (
                        {"noise": torch.full_like(Y_fant[0], 0.01)}
                        if isinstance(model, FixedNoiseGP)
                        else {}
                    )
                    fm_same_inputs = model.get_fantasy_model(
                        X_fant[0], Y_fant, **fant_kwargs_same_inputs
                    )

                    test_Xs = [
                        # test broadcasting single input across fantasy and
                        # model batches
                        torch.rand(4, 1, **tkwargs),
                        # separate input for each model batch and broadcast across
                        # fantasy batches
                        torch.rand(batch_shape + torch.Size([4, 1]), **tkwargs),
                        # separate input for each model and fantasy batch
                        torch.rand(
                            fant_shape + batch_shape + torch.Size([4, 1]), **tkwargs
                        ),
                    ]
                    for test_X in test_Xs:
                        posterior = fm.posterior(test_X)
                        self.assertEqual(
                            posterior.mean.shape,
                            fant_shape + batch_shape + torch.Size([4, num_outputs]),
                        )
                        posterior_same_inputs = fm_same_inputs.posterior(test_X)
                        self.assertEqual(
                            posterior_same_inputs.mean.shape,
                            fant_shape + batch_shape + torch.Size([4, num_outputs]),
                        )

                        # check that fantasies of batched model are correct
                        if len(batch_shape) > 0 and test_X.dim() == 2:
                            state_dict_non_batch = {
                                key: (val[0] if val.numel() > 1 else val)
                                for key, val in model.state_dict().items()
                            }
                            model_kwargs_non_batch = {
                                "train_X": model_kwargs["train_X"][0],
                                "train_Y": model_kwargs["train_Y"][0],
                            }
                            if "train_Yvar" in model_kwargs:
                                model_kwargs_non_batch["train_Yvar"] = model_kwargs[
                                    "train_Yvar"
                                ][0]
                            model_non_batch = type(model)(**model_kwargs_non_batch)
                            model_non_batch.load_state_dict(state_dict_non_batch)
                            model_non_batch.eval()
                            model_non_batch.likelihood.eval()
                            model_non_batch.posterior(
                                torch.rand(torch.Size([4, 1]), **tkwargs)
                            )
                            fant_kwargs = (
                                {"noise": torch.full_like(Y_fant[0, 0, :], 0.01)}
                                if isinstance(model, FixedNoiseGP)
                                else {}
                            )
                            fm_non_batch = model_non_batch.get_fantasy_model(
                                X_fant[0][0], Y_fant[:, 0, :], **fant_kwargs
                            )
                            non_batch_posterior = fm_non_batch.posterior(test_X)
                            self.assertTrue(
                                torch.allclose(
                                    posterior_same_inputs.mean[:, 0, ...],
                                    non_batch_posterior.mean,
                                    atol=1e-3,
                                )
                            )
                            self.assertTrue(
                                torch.allclose(
                                    posterior_same_inputs.mvn.covariance_matrix[
                                        :, 0, :, :
                                    ],
                                    non_batch_posterior.mvn.covariance_matrix,
                                    atol=1e-3,
                                )
                            )

    def test_get_fantasy_model_cuda(self):
        if torch.cuda.is_available():
            self.test_get_fantasy_model(cuda=True)


class TestFixedNoiseGP(TestSingleTaskGP):
    def _get_model_and_data(self, batch_shape, num_outputs, **tkwargs):
        train_X, train_Y = _get_random_data(
            batch_shape=batch_shape, num_outputs=num_outputs, **tkwargs
        )
        model_kwargs = {
            "train_X": train_X,
            "train_Y": train_Y,
            "train_Yvar": torch.full_like(train_Y, 0.01),
        }
        model = FixedNoiseGP(**model_kwargs)
        return model, model_kwargs

    def test_fixed_noise_likelihood(self, cuda=False):
        for batch_shape in (torch.Size([]), torch.Size([2])):
            for num_outputs in (1, 2):
                for double in (False, True):
                    tkwargs = {
                        "device": torch.device("cuda") if cuda else torch.device("cpu"),
                        "dtype": torch.double if double else torch.float,
                    }
                    model, model_kwargs = self._get_model_and_data(
                        batch_shape=batch_shape, num_outputs=num_outputs, **tkwargs
                    )
                    self.assertIsInstance(
                        model.likelihood, FixedNoiseGaussianLikelihood
                    )
                    self.assertTrue(
                        torch.equal(
                            model.likelihood.noise.contiguous().view(-1),
                            model_kwargs["train_Yvar"].contiguous().view(-1),
                        )
                    )

    def test_fixed_noise_likelihood_cuda(self):
        if torch.cuda.is_available():
            self.test_fixed_noise_likelihood(cuda=True)


class TestHeteroskedasticSingleTaskGP(TestSingleTaskGP):
    def _get_model_and_data(self, batch_shape, num_outputs, **tkwargs):
        train_X, train_Y = _get_random_data(
            batch_shape=batch_shape, num_outputs=num_outputs, **tkwargs
        )
        train_Yvar = (0.1 + 0.1 * torch.rand_like(train_Y)) ** 2
        model_kwargs = {
            "train_X": train_X,
            "train_Y": train_Y,
            "train_Yvar": train_Yvar,
        }
        model = HeteroskedasticSingleTaskGP(**model_kwargs)
        return model, model_kwargs

    def test_heteroskedastic_likelihood(self, cuda=False):
        for batch_shape in (torch.Size([]), torch.Size([2])):
            for num_outputs in (1, 2):
                for double in (False, True):
                    tkwargs = {
                        "device": torch.device("cuda") if cuda else torch.device("cpu"),
                        "dtype": torch.double if double else torch.float,
                    }
                    model, _ = self._get_model_and_data(
                        batch_shape=batch_shape, num_outputs=num_outputs, **tkwargs
                    )
                    self.assertIsInstance(model.likelihood, _GaussianLikelihoodBase)
                    self.assertFalse(isinstance(model.likelihood, GaussianLikelihood))
                    self.assertIsInstance(
                        model.likelihood.noise_covar, HeteroskedasticNoise
                    )
                    self.assertIsInstance(
                        model.likelihood.noise_covar.noise_model, SingleTaskGP
                    )

    def test_heteroskedastic_likelihood_cuda(self):
        if torch.cuda.is_available():
            self.test_heteroskedastic_likelihood(cuda=True)
