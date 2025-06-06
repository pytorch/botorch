#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from itertools import product

import torch
from botorch.acquisition.thompson_sampling import PathwiseThompsonSampling
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.model import Model
from botorch.utils.test_helpers import get_model
from botorch.utils.testing import BotorchTestCase


def _get_mcmc_samples(num_samples: int, dim: int, infer_noise: bool, **tkwargs):
    mcmc_samples = {
        "lengthscale": torch.rand(num_samples, 1, dim, **tkwargs),
        "outputscale": torch.rand(num_samples, **tkwargs),
        "mean": torch.randn(num_samples, **tkwargs),
    }
    if infer_noise:
        mcmc_samples["noise"] = torch.rand(num_samples, 1, **tkwargs)
    return mcmc_samples


def get_fully_bayesian_model(
    train_X,
    train_Y,
    num_models,
    **tkwargs,
):
    model = SaasFullyBayesianSingleTaskGP(
        train_X=train_X,
        train_Y=train_Y,
    )
    mcmc_samples = _get_mcmc_samples(
        num_samples=num_models,
        dim=train_X.shape[-1],
        infer_noise=True,
        **tkwargs,
    )
    model.load_mcmc_samples(mcmc_samples)
    return model


class TestPathwiseThompsonSampling(BotorchTestCase):
    def _test_thompson_sampling_base(self, model: Model):
        acq = PathwiseThompsonSampling(
            model=model,
        )
        X_observed = model.train_inputs[0]
        input_dim = X_observed.shape[-1]
        test_X = torch.rand(4, 1, input_dim).to(X_observed)
        # re-draw samples and expect other output
        acq_pass = acq(test_X)
        self.assertTrue(acq_pass.shape == test_X.shape[:-2])

        acq_pass1 = acq(test_X)
        self.assertAllClose(acq_pass1, acq(test_X))
        acq.redraw()
        acq_pass2 = acq(test_X)
        self.assertFalse(torch.allclose(acq_pass1, acq_pass2))

    def _test_thompson_sampling_batch(self, model: Model):
        X_observed = model.train_inputs[0]
        input_dim = X_observed.shape[-1]
        batch_acq = PathwiseThompsonSampling(
            model=model,
        )
        self.assertEqual(batch_acq.batch_size, None)
        test_X = torch.rand(4, 5, input_dim).to(X_observed)
        batch_acq(test_X)
        self.assertEqual(batch_acq.batch_size, 5)
        test_X = torch.rand(4, 7, input_dim).to(X_observed)
        with self.assertRaisesRegex(
            ValueError,
            "The batch size of PathwiseThompsonSampling should not "
            "change during a forward pass - was 5, now 7. Please re-initialize "
            "the acquisition if you want to change the batch size.",
        ):
            batch_acq(test_X)

        batch_acq2 = PathwiseThompsonSampling(model)
        test_X = torch.rand(4, 7, 1, input_dim).to(X_observed)
        self.assertEqual(batch_acq2(test_X).shape, test_X.shape[:-2])

        batch_acq3 = PathwiseThompsonSampling(model)
        test_X = torch.rand(4, 7, 3, input_dim).to(X_observed)
        self.assertEqual(batch_acq3(test_X).shape, test_X.shape[:-2])

    def test_thompson_sampling_single_task(self):
        input_dim = 2
        num_objectives = 1
        for dtype, standardize_model in product(
            (torch.float32, torch.float64), (True, False)
        ):
            tkwargs = {"device": self.device, "dtype": dtype}
            train_X = torch.rand(4, input_dim, **tkwargs)
            train_Y = 10 * torch.rand(4, num_objectives, **tkwargs)
            model = get_model(train_X, train_Y, standardize_model=standardize_model)
            self._test_thompson_sampling_base(model)
            self._test_thompson_sampling_batch(model)

    def test_thompson_sampling_fully_bayesian(self):
        input_dim = 2
        num_objectives = 1
        tkwargs = {"device": self.device, "dtype": torch.float64}
        train_X = torch.rand(4, input_dim, **tkwargs)
        train_Y = 10 * torch.rand(4, num_objectives, **tkwargs)

        fb_model = get_fully_bayesian_model(train_X, train_Y, num_models=3, **tkwargs)
        with self.assertRaisesRegex(
            NotImplementedError,
            "PathwiseThompsonSampling is not supported for fully Bayesian models",
        ):
            PathwiseThompsonSampling(model=fb_model)
