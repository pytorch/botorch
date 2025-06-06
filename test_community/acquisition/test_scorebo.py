#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from botorch.acquisition.objective import ScalarizedPosteriorTransform
from botorch.utils.test_helpers import get_fully_bayesian_model
from botorch.utils.testing import BotorchTestCase
from botorch_community.acquisition.scorebo import qSelfCorrectingBayesianOptimization


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
        ) in [
            (torch.float, "hellinger", True, True),
            (torch.double, "hellinger", False, False),
            (torch.float, "kl_divergence", True, True),
            (torch.double, "kl_divergence", False, False),
            (torch.double, "kl_divergence", True, False),
        ]:
            tkwargs["dtype"] = dtype
            input_dim = 2
            train_X = torch.rand(5, input_dim, **tkwargs)
            train_Y = torch.rand(5, num_objectives, **tkwargs)

            model = get_fully_bayesian_model(
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

        acq = qSelfCorrectingBayesianOptimization(
            model=model,
            optimal_inputs=optimal_inputs,
            optimal_outputs=optimal_outputs,
            posterior_transform=ScalarizedPosteriorTransform(
                weights=-torch.ones(1, **tkwargs)
            ),
        )
        self.assertTrue(torch.all(acq.optimal_output_values == -acq.optimal_outputs))
        acq_X = acq(test_Xs[j])
        self.assertTrue(acq_X.shape == test_Xs[j].shape[:-2])

        with self.assertRaises(ValueError):
            acq = qSelfCorrectingBayesianOptimization(
                model=model,
                optimal_inputs=optimal_inputs,
                optimal_outputs=optimal_outputs,
                distance_metric="NOT_A_DISTANCE",
                X_pending=X_pending,
            )
