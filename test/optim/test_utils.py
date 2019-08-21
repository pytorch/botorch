#! /usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest
import warnings
from copy import deepcopy

import torch
from botorch import settings
from botorch.exceptions import BotorchError
from botorch.models import ModelListGP, SingleTaskGP
from botorch.optim.utils import (
    _expand_bounds,
    _get_extra_mll_args,
    check_convergence,
    columnwise_clamp,
    fix_features,
    sample_all_priors,
)
from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.mlls.marginal_log_likelihood import MarginalLogLikelihood
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from gpytorch.mlls.variational_elbo import VariationalELBO
from gpytorch.priors.smoothed_box_prior import SmoothedBoxPrior
from gpytorch.priors.torch_priors import GammaPrior


class TestCheckConvergence(unittest.TestCase):
    def test_check_convergence(self, cuda=False):
        losses = torch.rand(5).tolist()
        self.assertTrue(
            check_convergence(
                loss_trajectory=losses, param_trajectory=[], options={"maxiter": 5}
            )
        )
        self.assertFalse(
            check_convergence(
                loss_trajectory=losses, param_trajectory=[], options={"maxiter": 6}
            )
        )

    def test_check_convergence_cuda(self):
        if torch.cuda.is_available():
            self.test_check_convergence(cuda=True)


class TestColumnWiseClamp(unittest.TestCase):
    def setUp(self):
        self.X = torch.tensor([[-2, 1], [0.5, -0.5]])
        self.X_expected = torch.tensor([[-1, 0.5], [0.5, -0.5]])

    def test_column_wise_clamp_scalars(self, cuda=False):
        X = self.X.cuda() if cuda else self.X
        X_expected = self.X_expected.cuda() if cuda else self.X_expected
        with self.assertRaises(ValueError):
            X_clmp = columnwise_clamp(X, 1, -1)
        X_clmp = columnwise_clamp(X, -1, 0.5)
        self.assertTrue(torch.equal(X_clmp, X_expected))
        X_clmp = columnwise_clamp(X, -3, 3)
        self.assertTrue(torch.equal(X_clmp, X))

    def test_column_wise_clamp_scalars_cuda(self):
        if torch.cuda.is_available():
            self.test_column_wise_clamp_scalars(cuda=True)

    def test_column_wise_clamp_scalar_tensors(self, cuda=False):
        X = self.X.cuda() if cuda else self.X
        X_expected = self.X_expected.cuda() if cuda else self.X_expected
        with self.assertRaises(ValueError):
            X_clmp = columnwise_clamp(X, torch.tensor(1), torch.tensor(-1))
        X_clmp = columnwise_clamp(X, torch.tensor(-1), torch.tensor(0.5))
        self.assertTrue(torch.equal(X_clmp, X_expected))
        X_clmp = columnwise_clamp(X, torch.tensor(-3), torch.tensor(3))
        self.assertTrue(torch.equal(X_clmp, X))

    def test_column_wise_clamp_scalar_tensors_cuda(self):
        if torch.cuda.is_available():
            self.test_column_wise_clamp_scalar_tensors(cuda=True)

    def test_column_wise_clamp_tensors(self, cuda=False):
        X = self.X.cuda() if cuda else self.X
        X_expected = self.X_expected.cuda() if cuda else self.X_expected
        with self.assertRaises(ValueError):
            X_clmp = columnwise_clamp(X, torch.ones(2), torch.zeros(2))
        with self.assertRaises(RuntimeError):
            X_clmp = columnwise_clamp(X, torch.zeros(3), torch.ones(3))
        X_clmp = columnwise_clamp(X, torch.tensor([-1, -1]), torch.tensor([0.5, 0.5]))
        self.assertTrue(torch.equal(X_clmp, X_expected))
        X_clmp = columnwise_clamp(X, torch.tensor([-3, -3]), torch.tensor([3, 3]))
        self.assertTrue(torch.equal(X_clmp, X))

    def test_column_wise_clamp_tensors_cuda(self):
        if torch.cuda.is_available():
            self.test_column_wise_clamp_tensors(cuda=True)

    def test_column_wise_clamp_raise_on_violation(self, cuda=False):
        X = self.X.cuda() if cuda else self.X
        with self.assertRaises(BotorchError):
            X_clmp = columnwise_clamp(
                X, torch.zeros(2), torch.ones(2), raise_on_violation=True
            )
        X_clmp = columnwise_clamp(
            X, torch.tensor([-3, -3]), torch.tensor([3, 3]), raise_on_violation=True
        )
        self.assertTrue(torch.equal(X_clmp, X))

    def test_column_wise_clamp_raise_on_violation_cuda(self):
        if torch.cuda.is_available():
            self.test_column_wise_clamp_raise_on_violation(cuda=True)


class TestFixFeatures(unittest.TestCase):
    def _getTensors(self, device):
        X = torch.tensor([[-2, 1, 3], [0.5, -0.5, 1.0]], device=device)
        X_null_two = torch.tensor([[-2, 1, 3], [0.5, -0.5, 1.0]], device=device)
        X_expected = torch.tensor([[-1, 1, -2], [-1, -0.5, -2]], device=device)
        X_expected_null_two = torch.tensor([[-1, 1, 3], [-1, -0.5, 1.0]], device=device)
        return X, X_null_two, X_expected, X_expected_null_two

    def test_fix_features(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        X, X_null_two, X_expected, X_expected_null_two = self._getTensors(device)
        X.requires_grad_(True)
        X_null_two.requires_grad_(True)

        X_fix = fix_features(X, {0: -1, 2: -2})
        X_fix_null_two = fix_features(X_null_two, {0: -1, 2: None})

        self.assertTrue(torch.equal(X_fix, X_expected))
        self.assertTrue(torch.equal(X_fix_null_two, X_expected_null_two))

        def f(X):
            return X.sum()

        f(X).backward()
        self.assertTrue(torch.equal(X.grad, torch.ones_like(X)))
        X.grad.zero_()

        f(X_fix).backward()
        self.assertTrue(
            torch.equal(
                X.grad, torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]], device=device)
            )
        )

        f(X_null_two).backward()
        self.assertTrue(torch.equal(X_null_two.grad, torch.ones_like(X)))
        X_null_two.grad.zero_()
        f(X_fix_null_two).backward()
        self.assertTrue(
            torch.equal(
                X_null_two.grad,
                torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]], device=device),
            )
        )

    def test_fix_features_cuda(self):
        if torch.cuda.is_available():
            self.test_fix_features(cuda=True)


class testGetExtraMllArgs(unittest.TestCase):
    def test_get_extra_mll_args(self):
        train_X = torch.rand(3, 5)
        train_Y = torch.rand(3, 1)
        model = SingleTaskGP(train_X=train_X, train_Y=train_Y)
        # test ExactMarginalLogLikelihood
        exact_mll = ExactMarginalLogLikelihood(model.likelihood, model)
        exact_extra_args = _get_extra_mll_args(mll=exact_mll)
        self.assertEqual(len(exact_extra_args), 1)
        self.assertTrue(torch.equal(exact_extra_args[0], train_X))

        # test VariationalELBO
        elbo = VariationalELBO(model.likelihood, model, num_data=train_X.shape[0])
        elbo_extra_args = _get_extra_mll_args(mll=elbo)
        self.assertEqual(len(elbo_extra_args), 0)

        # test SumMarginalLogLikelihood
        model2 = ModelListGP(model)
        sum_mll = SumMarginalLogLikelihood(model2.likelihood, model2)
        sum_mll_extra_args = _get_extra_mll_args(mll=sum_mll)
        self.assertEqual(len(sum_mll_extra_args), 1)
        self.assertEqual(len(sum_mll_extra_args[0]), 1)
        self.assertTrue(torch.equal(sum_mll_extra_args[0][0], train_X))

        # test unsupported MarginalLogLikelihood type
        unsupported_mll = MarginalLogLikelihood(model.likelihood, model)
        with self.assertRaises(ValueError):
            _get_extra_mll_args(mll=unsupported_mll)


class testExpandBounds(unittest.TestCase):
    def test_expand_bounds(self):
        X = torch.zeros(2, 3)
        expected_bounds = torch.zeros(1, 3)
        # bounds is float
        bounds = 0.0
        expanded_bounds = _expand_bounds(bounds=bounds, X=X)
        self.assertTrue(torch.equal(expected_bounds, expanded_bounds))
        # bounds is 0-d
        bounds = torch.tensor(0.0)
        expanded_bounds = _expand_bounds(bounds=bounds, X=X)
        self.assertTrue(torch.equal(expected_bounds, expanded_bounds))
        # bounds is 1-d
        bounds = torch.zeros(3)
        expanded_bounds = _expand_bounds(bounds=bounds, X=X)
        self.assertTrue(torch.equal(expected_bounds, expanded_bounds))
        # bounds is > 1-d
        bounds = torch.zeros(1, 3)
        expanded_bounds = _expand_bounds(bounds=bounds, X=X)
        self.assertTrue(torch.equal(expected_bounds, expanded_bounds))
        # bounds is None
        expanded_bounds = _expand_bounds(bounds=None, X=X)
        self.assertIsNone(expanded_bounds)


class TestSampleAllPriors(unittest.TestCase):
    def test_sample_all_priors(self, cuda=False):
        device = torch.device("cuda" if cuda else "cpu")
        for dtype in (torch.float, torch.double):
            train_X = torch.rand(3, 5, device=device, dtype=dtype)
            train_Y = torch.rand(3, 1, device=device, dtype=dtype)
            model = SingleTaskGP(train_X=train_X, train_Y=train_Y)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            mll.to(device=device, dtype=dtype)
            original_state_dict = dict(deepcopy(mll.model.state_dict()))
            sample_all_priors(model)

            # make sure one of the hyperparameters changed
            self.assertTrue(
                dict(model.state_dict())["likelihood.noise_covar.raw_noise"]
                != original_state_dict["likelihood.noise_covar.raw_noise"]
            )

            # change one of the priors to SmoothedBoxPrior
            model.covar_module = ScaleKernel(
                MaternKernel(
                    nu=2.5,
                    ard_num_dims=model.train_inputs[0].shape[-1],
                    batch_shape=model._aug_batch_shape,
                    lengthscale_prior=SmoothedBoxPrior(3.0, 6.0),
                ),
                batch_shape=model._aug_batch_shape,
                outputscale_prior=GammaPrior(2.0, 0.15),
            )
            original_state_dict = dict(deepcopy(mll.model.state_dict()))
            with warnings.catch_warnings(record=True) as ws, settings.debug(True):
                sample_all_priors(model)
                self.assertEqual(len(ws), 1)
                self.assertTrue("rsample" in str(ws[0].message))

            # the lengthscale should not have changed because sampling is
            # not implemented for SmoothedBoxPrior
            self.assertTrue(
                torch.equal(
                    dict(model.state_dict())[
                        "covar_module.base_kernel.raw_lengthscale"
                    ],
                    original_state_dict["covar_module.base_kernel.raw_lengthscale"],
                )
            )

            # set setting_closure to None and make sure RuntimeError is raised
            prior_tuple = model.likelihood.noise_covar._priors["noise_prior"]
            model.likelihood.noise_covar._priors["noise_prior"] = (
                prior_tuple[0],
                prior_tuple[1],
                None,
            )
            with self.assertRaises(RuntimeError):
                sample_all_priors(model)

    def test_sample_all_priors_cuda(self):
        if torch.cuda.is_available():
            self.test_sample_all_priors(cuda=True)
