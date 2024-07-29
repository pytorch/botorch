#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from itertools import product

import torch
from botorch.acquisition.bayesian_active_learning import (
    FullyBayesianAcquisitionFunction,
)
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.utils.testing import BotorchTestCase
from botorch_community.acquisition.bayesian_active_learning import (
    qBayesianQueryByComittee,
    qBayesianVarianceReduction,
    qStatisticalDistanceActiveLearning,
)


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


class TestFullyBayesianActuisitionFunction(BotorchTestCase):
    def test_abstract_raises(self):
        with self.assertRaises(TypeError):
            FullyBayesianAcquisitionFunction()


class TestQStatisticalDistanceActiveLearning(BotorchTestCase):
    def test_q_statistical_distance_active_learning(self):
        torch.manual_seed(1)
        tkwargs = {"device": self.device}
        distance_metrics = ("hellinger", "kl_divergence")
        num_objectives = 1
        num_models = 3
        for (
            dtype,
            distance_metric,
            standardize_model,
            infer_noise,
        ) in product(
            (torch.float, torch.double),
            distance_metrics,
            (False, True),  # standardize_model
            (True,),  # infer_noise - only one option avail in PyroModels
        ):
            tkwargs["dtype"] = dtype
            input_dim = 2
            train_X = torch.rand(4, input_dim, **tkwargs)
            train_Y = torch.rand(4, num_objectives, **tkwargs)

            model = get_model(
                train_X,
                train_Y,
                num_models,
                standardize_model,
                infer_noise,
                **tkwargs,
            )

            # test acquisition
            X_pending_list = [None, torch.rand(2, input_dim, **tkwargs)]
            for i in range(len(X_pending_list)):
                X_pending = X_pending_list[i]

                acq = qStatisticalDistanceActiveLearning(
                    model=model,
                    X_pending=X_pending,
                    distance_metric=distance_metric,
                )

                test_Xs = [
                    torch.rand(4, 1, input_dim, **tkwargs),
                    torch.rand(4, 3, input_dim, **tkwargs),
                    torch.rand(4, 5, 1, input_dim, **tkwargs),
                    torch.rand(4, 5, 3, input_dim, **tkwargs),
                ]

                for j in range(len(test_Xs)):
                    acq_X = acq.forward(test_Xs[j])
                    acq_X = acq(test_Xs[j])
                    # assess shape
                    self.assertTrue(acq_X.shape == test_Xs[j].shape[:-2])

        with self.assertRaises(ValueError):
            acq = qStatisticalDistanceActiveLearning(
                model=model,
                distance_metric="NOT_A_DISTANCE",
                X_pending=X_pending,
            )

        # Support with non-fully bayesian models is not possible. Thus, we
        # throw an error.
        non_fully_bayesian_model = SingleTaskGP(train_X, train_Y)
        with self.assertRaises(ValueError):
            acq = qStatisticalDistanceActiveLearning(
                model=non_fully_bayesian_model,
            )


class TestQBayesianQueryByComittee(BotorchTestCase):
    def test_q_bayesian_query_by_comittee(self):
        torch.manual_seed(1)
        tkwargs = {"device": self.device}
        num_objectives = 1
        num_models = 3
        for (
            dtype,
            standardize_model,
            infer_noise,
        ) in product(
            (torch.float, torch.double),
            (False, True),  # standardize_model
            (True,),  # infer_noise - only one option avail in PyroModels
        ):
            tkwargs["dtype"] = dtype
            input_dim = 2
            train_X = torch.rand(4, input_dim, **tkwargs)
            train_Y = torch.rand(4, num_objectives, **tkwargs)

            model = get_model(
                train_X,
                train_Y,
                num_models,
                standardize_model,
                infer_noise,
                **tkwargs,
            )

            # test acquisition
            X_pending_list = [None, torch.rand(2, input_dim, **tkwargs)]
            for i in range(len(X_pending_list)):
                X_pending = X_pending_list[i]

                acq = qBayesianQueryByComittee(
                    model=model,
                    X_pending=X_pending,
                )

                test_Xs = [
                    torch.rand(4, 1, input_dim, **tkwargs),
                    torch.rand(4, 3, input_dim, **tkwargs),
                    torch.rand(4, 5, 1, input_dim, **tkwargs),
                    torch.rand(4, 5, 3, input_dim, **tkwargs),
                ]

                for j in range(len(test_Xs)):
                    acq_X = acq.forward(test_Xs[j])
                    acq_X = acq(test_Xs[j])
                    # assess shape
                    self.assertTrue(acq_X.shape == test_Xs[j].shape[:-2])

        # Support with non-fully bayesian models is not possible. Thus, we
        # throw an error.
        non_fully_bayesian_model = SingleTaskGP(train_X, train_Y)
        with self.assertRaises(ValueError):
            acq = qBayesianQueryByComittee(
                model=non_fully_bayesian_model,
            )


class TestQBayesianVarianceReduction(BotorchTestCase):
    def test_q_bayesian_variance_reduction(self):
        torch.manual_seed(1)
        tkwargs = {"device": self.device}
        num_objectives = 1
        num_models = 3
        for (
            dtype,
            standardize_model,
            infer_noise,
        ) in product(
            (torch.float, torch.double),
            (False, True),  # standardize_model
            (True,),  # infer_noise - only one option avail in PyroModels
        ):
            tkwargs["dtype"] = dtype
            input_dim = 2
            train_X = torch.rand(4, input_dim, **tkwargs)
            train_Y = torch.rand(4, num_objectives, **tkwargs)

            model = get_model(
                train_X,
                train_Y,
                num_models,
                standardize_model,
                infer_noise,
                **tkwargs,
            )

            # test acquisition
            X_pending_list = [None, torch.rand(2, input_dim, **tkwargs)]
            for i in range(len(X_pending_list)):
                X_pending = X_pending_list[i]

                acq = qBayesianVarianceReduction(
                    model=model,
                    X_pending=X_pending,
                )

                test_Xs = [
                    torch.rand(4, 1, input_dim, **tkwargs),
                    torch.rand(4, 3, input_dim, **tkwargs),
                    torch.rand(4, 5, 1, input_dim, **tkwargs),
                    torch.rand(4, 5, 3, input_dim, **tkwargs),
                ]

                for j in range(len(test_Xs)):
                    acq_X = acq.forward(test_Xs[j])
                    acq_X = acq(test_Xs[j])
                    # assess shape
                    self.assertTrue(acq_X.shape == test_Xs[j].shape[:-2])

        # Support with non-fully bayesian models is not possible. Thus, we
        # throw an error.
        non_fully_bayesian_model = SingleTaskGP(train_X, train_Y)
        with self.assertRaises(ValueError):
            acq = qBayesianVarianceReduction(
                model=non_fully_bayesian_model,
            )
