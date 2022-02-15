#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
import warnings
from copy import deepcopy

import numpy as np
import torch
from botorch import settings
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.acquisition.monte_carlo import (
    qExpectedImprovement,
    qNoisyExpectedImprovement,
)
from botorch.acquisition.multi_objective.max_value_entropy_search import (
    qMultiObjectiveMaxValueEntropy,
)
from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement,
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.exceptions import BotorchError
from botorch.exceptions.warnings import BotorchWarning
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms.input import Warp
from botorch.optim.utils import (
    _expand_bounds,
    _get_extra_mll_args,
    _handle_numerical_errors,
    columnwise_clamp,
    fix_features,
    get_X_baseline,
    sample_all_priors,
)
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from botorch.utils.testing import BotorchTestCase, MockModel, MockPosterior
from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.mlls.marginal_log_likelihood import MarginalLogLikelihood
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from gpytorch.priors.prior import Prior
from gpytorch.priors.torch_priors import GammaPrior
from gpytorch.utils.errors import NanError, NotPSDError


class DummyPrior(Prior):
    arg_constraints = {}

    def rsample(self, sample_shape=torch.Size()):  # noqa: B008
        raise NotImplementedError


class TestColumnWiseClamp(BotorchTestCase):
    def setUp(self):
        super().setUp()
        self.X = torch.tensor([[-2, 1], [0.5, -0.5]], device=self.device)
        self.X_expected = torch.tensor([[-1, 0.5], [0.5, -0.5]], device=self.device)

    def test_column_wise_clamp_scalars(self):
        X, X_expected = self.X, self.X_expected
        with self.assertRaises(ValueError):
            X_clmp = columnwise_clamp(X, 1, -1)
        X_clmp = columnwise_clamp(X, -1, 0.5)
        self.assertTrue(torch.equal(X_clmp, X_expected))
        X_clmp = columnwise_clamp(X, -3, 3)
        self.assertTrue(torch.equal(X_clmp, X))

    def test_column_wise_clamp_scalar_tensors(self):
        X, X_expected = self.X, self.X_expected
        with self.assertRaises(ValueError):
            X_clmp = columnwise_clamp(X, torch.tensor(1), torch.tensor(-1))
        X_clmp = columnwise_clamp(X, torch.tensor(-1), torch.tensor(0.5))
        self.assertTrue(torch.equal(X_clmp, X_expected))
        X_clmp = columnwise_clamp(X, torch.tensor(-3), torch.tensor(3))
        self.assertTrue(torch.equal(X_clmp, X))

    def test_column_wise_clamp_tensors(self):
        X, X_expected = self.X, self.X_expected
        with self.assertRaises(ValueError):
            X_clmp = columnwise_clamp(X, torch.ones(2), torch.zeros(2))
        with self.assertRaises(RuntimeError):
            X_clmp = columnwise_clamp(X, torch.zeros(3), torch.ones(3))
        X_clmp = columnwise_clamp(X, torch.tensor([-1, -1]), torch.tensor([0.5, 0.5]))
        self.assertTrue(torch.equal(X_clmp, X_expected))
        X_clmp = columnwise_clamp(X, torch.tensor([-3, -3]), torch.tensor([3, 3]))
        self.assertTrue(torch.equal(X_clmp, X))

    def test_column_wise_clamp_full_dim_tensors(self):
        X = torch.tensor([[[-1, 2, 0.5], [0.5, 3, 1.5]], [[0.5, 1, 0], [2, -2, 3]]])
        lower = torch.tensor([[[0, 0.5, 1], [0, 2, 2]], [[0, 2, 0], [1, -1, 0]]])
        upper = torch.tensor([[[1, 1.5, 1], [1, 4, 3]], [[1, 3, 0.5], [3, 1, 2.5]]])
        X_expected = torch.tensor(
            [[[0, 1.5, 1], [0.5, 3, 2]], [[0.5, 2, 0], [2, -1, 2.5]]]
        )
        X_clmp = columnwise_clamp(X, lower, upper)
        self.assertTrue(torch.equal(X_clmp, X_expected))
        X_clmp = columnwise_clamp(X, lower - 5, upper + 5)
        self.assertTrue(torch.equal(X_clmp, X))
        with self.assertRaises(ValueError):
            X_clmp = columnwise_clamp(X, torch.ones_like(X), torch.zeros_like(X))
        with self.assertRaises(RuntimeError):
            X_clmp = columnwise_clamp(X, lower.unsqueeze(-3), upper.unsqueeze(-3))

    def test_column_wise_clamp_raise_on_violation(self):
        X = self.X
        with self.assertRaises(BotorchError):
            X_clmp = columnwise_clamp(
                X, torch.zeros(2), torch.ones(2), raise_on_violation=True
            )
        X_clmp = columnwise_clamp(
            X, torch.tensor([-3, -3]), torch.tensor([3, 3]), raise_on_violation=True
        )
        self.assertTrue(torch.equal(X_clmp, X))


class TestFixFeatures(BotorchTestCase):
    def _getTensors(self):
        X = torch.tensor([[-2, 1, 3], [0.5, -0.5, 1.0]], device=self.device)
        X_null_two = torch.tensor([[-2, 1, 3], [0.5, -0.5, 1.0]], device=self.device)
        X_expected = torch.tensor([[-1, 1, -2], [-1, -0.5, -2]], device=self.device)
        X_expected_null_two = torch.tensor(
            [[-1, 1, 3], [-1, -0.5, 1.0]], device=self.device
        )
        return X, X_null_two, X_expected, X_expected_null_two

    def test_fix_features(self):
        X, X_null_two, X_expected, X_expected_null_two = self._getTensors()
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
                X.grad,
                torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]], device=self.device),
            )
        )

        f(X_null_two).backward()
        self.assertTrue(torch.equal(X_null_two.grad, torch.ones_like(X)))
        X_null_two.grad.zero_()
        f(X_fix_null_two).backward()
        self.assertTrue(
            torch.equal(
                X_null_two.grad,
                torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]], device=self.device),
            )
        )


class TestGetExtraMllArgs(BotorchTestCase):
    def test_get_extra_mll_args(self):
        train_X = torch.rand(3, 5)
        train_Y = torch.rand(3, 1)
        model = SingleTaskGP(train_X=train_X, train_Y=train_Y)

        # test ExactMarginalLogLikelihood
        exact_mll = ExactMarginalLogLikelihood(model.likelihood, model)
        exact_extra_args = _get_extra_mll_args(mll=exact_mll)
        self.assertEqual(len(exact_extra_args), 1)
        self.assertTrue(torch.equal(exact_extra_args[0], train_X))

        # test SumMarginalLogLikelihood
        model2 = ModelListGP(model)
        sum_mll = SumMarginalLogLikelihood(model2.likelihood, model2)
        sum_mll_extra_args = _get_extra_mll_args(mll=sum_mll)
        self.assertEqual(len(sum_mll_extra_args), 1)
        self.assertEqual(len(sum_mll_extra_args[0]), 1)
        self.assertTrue(torch.equal(sum_mll_extra_args[0][0], train_X))

        # test unsupported MarginalLogLikelihood type
        unsupported_mll = MarginalLogLikelihood(model.likelihood, model)
        unsupported_mll_extra_args = _get_extra_mll_args(mll=unsupported_mll)
        self.assertEqual(unsupported_mll_extra_args, [])


class TestExpandBounds(BotorchTestCase):
    def test_expand_bounds(self):
        X = torch.zeros(2, 3)
        expected_bounds = torch.zeros(2, 3)
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
        # bounds is 2-d
        bounds = torch.zeros(1, 3)
        expanded_bounds = _expand_bounds(bounds=bounds, X=X)
        self.assertTrue(torch.equal(expected_bounds, expanded_bounds))
        # bounds is > 2-d
        bounds = torch.zeros(1, 1, 3)
        with self.assertRaises(RuntimeError):
            # X does not have a t-batch
            expanded_bounds = _expand_bounds(bounds=bounds, X=X)
        X = torch.zeros(4, 2, 3)
        expanded_bounds = _expand_bounds(bounds=bounds, X=X)
        self.assertTrue(torch.equal(expanded_bounds, torch.zeros_like(X)))
        with self.assertRaises(RuntimeError):
            # bounds is not broadcastable to X
            expanded_bounds = _expand_bounds(bounds=torch.zeros(2, 1, 3), X=X)
        # bounds is None
        expanded_bounds = _expand_bounds(bounds=None, X=X)
        self.assertIsNone(expanded_bounds)


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


class TestHelpers(BotorchTestCase):
    def test_handle_numerical_errors(self):
        x = np.zeros(1)

        with self.assertRaisesRegex(NotPSDError, "foo"):
            _handle_numerical_errors(error=NotPSDError("foo"), x=x)

        for error in (
            NanError(),
            RuntimeError("singular"),
            RuntimeError("input is not positive-definite"),
        ):
            fake_loss, fake_grad = _handle_numerical_errors(error=error, x=x)
            self.assertTrue(math.isnan(fake_loss))
            self.assertEqual(fake_grad.shape, x.shape)
            self.assertTrue(np.isnan(fake_grad).all())

        with self.assertRaisesRegex(RuntimeError, "foo"):
            _handle_numerical_errors(error=RuntimeError("foo"), x=x)


class TestGetXBaseline(BotorchTestCase):
    def test_get_X_baseline(self):
        tkwargs = {"device": self.device}
        for dtype in (torch.float, torch.double):
            tkwargs["dtype"] = dtype
            X_train = torch.rand(20, 2, **tkwargs)
            model = MockModel(
                MockPosterior(mean=(2 * X_train + 1).sum(dim=-1, keepdim=True))
            )
            # test NEI with X_baseline
            acqf = qNoisyExpectedImprovement(
                model, X_baseline=X_train[:2], cache_root=False
            )
            X = get_X_baseline(acq_function=acqf)
            self.assertTrue(torch.equal(X, acqf.X_baseline))
            # test EI without X_baseline
            acqf = qExpectedImprovement(model, best_f=0.0)

            with warnings.catch_warnings(record=True) as w, settings.debug(True):

                X_rnd = get_X_baseline(
                    acq_function=acqf,
                )
                self.assertEqual(len(w), 1)
                self.assertTrue(issubclass(w[-1].category, BotorchWarning))
                self.assertIsNone(X_rnd)

            # set train inputs
            model.train_inputs = (X_train,)
            X = get_X_baseline(
                acq_function=acqf,
            )
            self.assertTrue(torch.equal(X, X_train))
            # test that we fail back to train_inputs if X_baseline is an empty tensor
            acqf.register_buffer("X_baseline", X_train[:0])
            X = get_X_baseline(
                acq_function=acqf,
            )
            self.assertTrue(torch.equal(X, X_train))

            # test acquisitipon function without X_baseline or model
            acqf = FixedFeatureAcquisitionFunction(acqf, d=2, columns=[0], values=[0])
            with warnings.catch_warnings(record=True) as w, settings.debug(True):
                X_rnd = get_X_baseline(
                    acq_function=acqf,
                )
                self.assertEqual(len(w), 1)
                self.assertTrue(issubclass(w[-1].category, BotorchWarning))
                self.assertIsNone(X_rnd)

            Y_train = 2 * X_train[:2] + 1
            moo_model = MockModel(MockPosterior(mean=Y_train, samples=Y_train))
            ref_point = torch.zeros(2, **tkwargs)
            # test NEHVI with X_baseline
            acqf = qNoisyExpectedHypervolumeImprovement(
                moo_model,
                ref_point=ref_point,
                X_baseline=X_train[:2],
                cache_root=False,
            )
            X = get_X_baseline(
                acq_function=acqf,
            )
            self.assertTrue(torch.equal(X, acqf.X_baseline))
            # test qEHVI without train_inputs
            acqf = qExpectedHypervolumeImprovement(
                moo_model,
                ref_point=ref_point,
                partitioning=FastNondominatedPartitioning(
                    ref_point=ref_point,
                    Y=Y_train,
                ),
            )
            # test extracting train_inputs from model list GP
            model_list = ModelListGP(
                SingleTaskGP(X_train, Y_train[:, :1]),
                SingleTaskGP(X_train, Y_train[:, 1:]),
            )
            acqf = qExpectedHypervolumeImprovement(
                model_list,
                ref_point=ref_point,
                partitioning=FastNondominatedPartitioning(
                    ref_point=ref_point,
                    Y=Y_train,
                ),
            )
            X = get_X_baseline(
                acq_function=acqf,
            )
            self.assertTrue(torch.equal(X, X_train))

            # test MESMO for which we need to use
            # `acqf.mo_model`
            batched_mo_model = SingleTaskGP(X_train, Y_train)
            acqf = qMultiObjectiveMaxValueEntropy(
                batched_mo_model,
                sample_pareto_frontiers=lambda model: torch.rand(10, 2, **tkwargs),
            )
            X = get_X_baseline(
                acq_function=acqf,
            )
            self.assertTrue(torch.equal(X, X_train))
            # test that if there is an input transform that is applied
            # to the train_inputs when the model is in eval mode, we
            # extract the untransformed train_inputs
            model = SingleTaskGP(
                X_train, Y_train[:, :1], input_transform=Warp(indices=[0, 1])
            )
            model.eval()
            self.assertFalse(torch.equal(model.train_inputs[0], X_train))
            acqf = qExpectedImprovement(model, best_f=0.0)
            X = get_X_baseline(
                acq_function=acqf,
            )
            self.assertTrue(torch.equal(X, X_train))
