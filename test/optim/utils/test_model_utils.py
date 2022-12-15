#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import re
import warnings
from copy import deepcopy
from string import ascii_lowercase
from unittest.mock import MagicMock, patch

import torch
from botorch import settings
from botorch.models import ModelListGP, SingleTaskGP
from botorch.optim.utils import (
    _get_extra_mll_args,
    get_data_loader,
    get_name_filter,
    get_parameters,
    get_parameters_and_bounds,
    model_utils,
    sample_all_priors,
)
from botorch.utils.testing import BotorchTestCase
from gpytorch.constraints import GreaterThan
from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.mlls.marginal_log_likelihood import MarginalLogLikelihood
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from gpytorch.priors import UniformPrior
from gpytorch.priors.prior import Prior
from gpytorch.priors.torch_priors import GammaPrior


class DummyPrior(Prior):
    arg_constraints = {}

    def rsample(self, sample_shape=torch.Size()):  # noqa: B008
        raise NotImplementedError


class DummyPriorRuntimeError(Prior):
    arg_constraints = {}

    def rsample(self, sample_shape=torch.Size()):  # noqa: B008
        raise RuntimeError("Another runtime error.")


class TestGetExtraMllArgs(BotorchTestCase):
    def test_get_extra_mll_args(self):
        train_X = torch.rand(3, 5)
        train_Y = torch.rand(3, 1)
        model = SingleTaskGP(train_X=train_X, train_Y=train_Y)

        # test ExactMarginalLogLikelihood
        exact_mll = ExactMarginalLogLikelihood(model.likelihood, model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)
            exact_extra_args = _get_extra_mll_args(mll=exact_mll)
        self.assertEqual(len(exact_extra_args), 1)
        self.assertTrue(torch.equal(exact_extra_args[0], train_X))

        # test SumMarginalLogLikelihood
        model2 = ModelListGP(model)
        sum_mll = SumMarginalLogLikelihood(model2.likelihood, model2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)
            sum_mll_extra_args = _get_extra_mll_args(mll=sum_mll)
        self.assertEqual(len(sum_mll_extra_args), 1)
        self.assertEqual(len(sum_mll_extra_args[0]), 1)
        self.assertTrue(torch.equal(sum_mll_extra_args[0][0], train_X))

        # test unsupported MarginalLogLikelihood type
        unsupported_mll = MarginalLogLikelihood(model.likelihood, model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)
            unsupported_mll_extra_args = _get_extra_mll_args(mll=unsupported_mll)
        self.assertEqual(unsupported_mll_extra_args, [])


class TestGetDataLoader(BotorchTestCase):
    def setUp(self):
        super().setUp()
        with torch.random.fork_rng():
            torch.random.manual_seed(0)
            train_X = torch.rand(3, 5, device=self.device)
            train_Y = torch.rand(3, 1, device=self.device)

        self.model = SingleTaskGP(train_X=train_X, train_Y=train_Y).to(torch.float64)

    def test_get_data_loader(self):
        data_loader = get_data_loader(self.model)
        self.assertEqual(data_loader.batch_size, len(self.model.train_targets))

        train_X, train_Y = next(iter(data_loader))
        self.assertTrue(self.model.train_inputs[0].equal(train_X))
        self.assertTrue(self.model.train_targets.equal(train_Y))

        _TensorDataset = MagicMock(return_value="foo")
        _DataLoader = MagicMock()
        with patch.multiple(
            model_utils, TensorDataset=_TensorDataset, DataLoader=_DataLoader
        ):
            model_utils.get_data_loader(self.model, batch_size=2, shuffle=True)
            _DataLoader.assert_called_once_with(
                dataset="foo",
                batch_size=2,
                shuffle=True,
            )


class TestGetParameters(BotorchTestCase):
    def setUp(self):
        self.module = GaussianLikelihood(
            noise_constraint=GreaterThan(1e-6, initial_value=0.123),
        )

    def test_get_parameters(self):
        self.assertEqual(0, len(get_parameters(self.module, requires_grad=False)))

        params = get_parameters(self.module)
        self.assertTrue(1 == len(params))
        self.assertEqual(next(iter(params)), "noise_covar.raw_noise")
        self.assertTrue(
            self.module.noise_covar.raw_noise.equal(next(iter(params.values())))
        )

    def test_get_parameters_and_bounds(self):
        param_dict, bounds_dict = get_parameters_and_bounds(self.module)
        self.assertTrue(1 == len(param_dict) == len(bounds_dict))

        name, bounds = next(iter(bounds_dict.items()))
        self.assertEqual(name, "noise_covar.raw_noise")
        self.assertEqual(bounds, (-float("inf"), float("inf")))

        mock_module = torch.nn.Module()
        mock_module.named_parameters = MagicMock(
            return_value=self.module.named_parameters()
        )
        param_dict2, bounds_dict2 = get_parameters_and_bounds(mock_module)
        self.assertEqual(param_dict, param_dict2)
        self.assertTrue(len(bounds_dict2) == 0)


class TestGetNameFilter(BotorchTestCase):
    def test_get_name_filter(self):
        with self.assertRaisesRegex(TypeError, "Expected `patterns` to contain"):
            get_name_filter(("foo", re.compile("bar"), 1))

        names = ascii_lowercase
        name_filter = get_name_filter(iter(names[1::2]))
        self.assertEqual(names[::2], "".join(filter(name_filter, names)))

        items = tuple(zip(names, range(len(names))))
        self.assertEqual(items[::2], tuple(filter(name_filter, items)))


class TestSampleAllPriors(BotorchTestCase):
    def test_sample_all_priors(self):
        for dtype in (torch.float, torch.double):
            train_X = torch.rand(3, 5, device=self.device, dtype=dtype)
            train_Y = torch.rand(3, 1, device=self.device, dtype=dtype)
            model = SingleTaskGP(train_X=train_X, train_Y=train_Y)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            mll.to(device=self.device, dtype=dtype)
            original_state_dict = dict(deepcopy(mll.model.state_dict()))
            sample_all_priors(model)

            # make sure one of the hyperparameters changed
            self.assertTrue(
                dict(model.state_dict())["likelihood.noise_covar.raw_noise"]
                != original_state_dict["likelihood.noise_covar.raw_noise"]
            )
            # check that lengthscales are all different
            ls = model.covar_module.base_kernel.raw_lengthscale.view(-1).tolist()
            self.assertTrue(all(ls[0] != ls[i]) for i in range(1, len(ls)))

            # change one of the priors to a dummy prior that does not support sampling
            model.covar_module = ScaleKernel(
                MaternKernel(
                    nu=2.5,
                    ard_num_dims=model.train_inputs[0].shape[-1],
                    batch_shape=model._aug_batch_shape,
                    lengthscale_prior=DummyPrior(),
                ),
                batch_shape=model._aug_batch_shape,
                outputscale_prior=GammaPrior(2.0, 0.15),
            )
            original_state_dict = dict(deepcopy(mll.model.state_dict()))
            with warnings.catch_warnings(record=True) as ws, settings.debug(True):
                sample_all_priors(model)
                self.assertEqual(len(ws), 1)
                self.assertTrue("rsample" in str(ws[0].message))

            # change to dummy prior that raises an unrecognized RuntimeError
            model.covar_module = ScaleKernel(
                MaternKernel(
                    nu=2.5,
                    ard_num_dims=model.train_inputs[0].shape[-1],
                    batch_shape=model._aug_batch_shape,
                    lengthscale_prior=DummyPriorRuntimeError(),
                ),
                batch_shape=model._aug_batch_shape,
                outputscale_prior=GammaPrior(2.0, 0.15),
            )
            with self.assertRaises(RuntimeError):
                sample_all_priors(model)

            # the lengthscale should not have changed because sampling is
            # not implemented for DummyPrior
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

            # test for error when sampling violates constraint
            model = SingleTaskGP(train_X=train_X, train_Y=train_Y)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            mll.to(device=self.device, dtype=dtype)
            model.covar_module = ScaleKernel(
                MaternKernel(
                    nu=2.5,
                    ard_num_dims=model.train_inputs[0].shape[-1],
                    batch_shape=model._aug_batch_shape,
                    lengthscale_prior=GammaPrior(3.0, 6.0),
                ),
                batch_shape=model._aug_batch_shape,
                outputscale_prior=UniformPrior(1.0, 2.0),
                outputscale_constraint=GreaterThan(3.0),
            )
            original_state_dict = dict(deepcopy(mll.model.state_dict()))
            with self.assertRaises(RuntimeError):
                sample_all_priors(model)
