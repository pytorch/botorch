#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.utils.testing import BotorchTestCase
from botorch_community.acquisition.scorebo import qSelfCorrectingBayesianOptimization


def _get_mcmc_samples(num_samples: int, dim: int, infer_noise: bool, **tkwargs):

    mcmc_samples = {
        "lengthscale": torch.rand(num_samples, 1, dim, **tkwargs),
        "outputscale": torch.rand(num_samples, **tkwargs),
        "mean": torch.randn(num_samples, **tkwargs),
    }
    if infer_noise:
        mcmc_samples["noise"] = torch.rand(num_samples, 1, **tkwargs)
    return mcmc_samples


def get_model(
    train_X,
    train_Y,
    num_models,
    standardize_model,
    infer_noise,
    **tkwargs,
):
    num_objectives = train_Y.shape[-1]

    if standardize_model:
        outcome_transform = Standardize(m=num_objectives)
    else:
        outcome_transform = None

    mcmc_samples = _get_mcmc_samples(
        num_samples=num_models,
        dim=train_X.shape[-1],
        infer_noise=infer_noise,
        **tkwargs,
    )

    model = SaasFullyBayesianSingleTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        outcome_transform=outcome_transform,
    )
    model.load_mcmc_samples(mcmc_samples)

    return model


class TestQSelfCorrectingBayesianOptimization(BotorchTestCase):
    def test_q_self_correcting_bayesian_optimization(self):
        torch.manual_seed(5)
        tkwargs = {"device": self.device}
        num_objectives = 1
        num_models = 3
        for (
            dtype,
            distance_metric,
            only_maxval,
            standardize_model,
            maximize,
        ) in [
            (torch.float, "hellinger", False, True, True),
            (torch.double, "hellinger", True, False, False),
            (torch.float, "kl_divergence", False, True, True),
            (torch.double, "kl_divergence", True, False, False),
            (torch.double, "kl_divergence", True, True, False),
        ]:
            tkwargs["dtype"] = dtype
            input_dim = 2
            train_X = torch.rand(5, input_dim, **tkwargs)
            train_Y = torch.rand(5, num_objectives, **tkwargs)

            model = get_model(
                train_X=train_X,
                train_Y=train_Y,
                num_models=num_models,
                standardize_model=standardize_model,
                infer_noise=True,
                **tkwargs,
            )

            num_optimal_samples = 7
            optimal_inputs = torch.rand(
                num_optimal_samples, num_models, input_dim, **tkwargs
            )

            # SCoreBO can work with only max-value, so we're testing that too
            if only_maxval:
                optimal_inputs = None
            optimal_outputs = torch.rand(
                num_optimal_samples, num_models, num_objectives, **tkwargs
            )

            # test acquisition
            X_pending_list = [None, torch.rand(2, input_dim, **tkwargs)]
            for i in range(len(X_pending_list)):
                X_pending = X_pending_list[i]

                acq = qSelfCorrectingBayesianOptimization(
                    model=model,
                    optimal_inputs=optimal_inputs,
                    optimal_outputs=optimal_outputs,
                    distance_metric=distance_metric,
                    X_pending=X_pending,
                    maximize=maximize,
                )

                test_Xs = [
                    torch.rand(4, 1, input_dim, **tkwargs),
                    torch.rand(4, 3, input_dim, **tkwargs),
                    torch.rand(4, 5, 1, input_dim, **tkwargs),
                    torch.rand(4, 5, 3, input_dim, **tkwargs),
                ]

                for j in range(len(test_Xs)):
                    acq_X = acq(test_Xs[j])
                    # assess shape
                    self.assertTrue(acq_X.shape == test_Xs[j].shape[:-2])

        with self.assertRaises(ValueError):
            acq = qSelfCorrectingBayesianOptimization(
                model=model,
                optimal_inputs=optimal_inputs,
                optimal_outputs=optimal_outputs,
                distance_metric="NOT_A_DISTANCE",
                X_pending=X_pending,
                maximize=maximize,
            )

        # Support with non-fully bayesian models is not possible. Thus, we
        # throw an error.
        non_fully_bayesian_model = SingleTaskGP(train_X, train_Y)
        with self.assertRaises(ValueError):
            acq = qSelfCorrectingBayesianOptimization(
                model=non_fully_bayesian_model,
                optimal_inputs=optimal_inputs,
                optimal_outputs=optimal_outputs,
            )
