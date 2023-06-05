#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from itertools import product

import torch
from botorch.acquisition.joint_entropy_search import qJointEntropySearch
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP

from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.testing import BotorchTestCase


def get_model(train_X, train_Y, use_model_list, standardize_model):
    num_objectives = train_Y.shape[-1]

    if standardize_model:
        if use_model_list:
            outcome_transform = Standardize(m=1)
        else:
            outcome_transform = Standardize(m=num_objectives)
    else:
        outcome_transform = None

    if use_model_list:
        model = ModelListGP(
            *[
                SingleTaskGP(
                    train_X=train_X,
                    train_Y=train_Y[:, i : i + 1],
                    outcome_transform=outcome_transform,
                )
                for i in range(num_objectives)
            ]
        )
    else:
        model = SingleTaskGP(
            train_X=train_X,
            train_Y=train_Y,
            outcome_transform=outcome_transform,
        )

    return model


class TestQJointEntropySearch(BotorchTestCase):
    def test_joint_entropy_search(self):
        torch.manual_seed(1)
        tkwargs = {"device": self.device}
        estimation_types = ("LB", "MC")

        num_objectives = 1
        for (
            dtype,
            estimation_type,
            use_model_list,
            standardize_model,
            maximize,
            condition_noiseless,
        ) in product(
            (torch.float, torch.double),
            estimation_types,
            (False, True),
            (False, True),
            (False, True),
            (False, True),
        ):
            tkwargs["dtype"] = dtype
            input_dim = 2
            train_X = torch.rand(4, input_dim, **tkwargs)
            train_Y = torch.rand(4, num_objectives, **tkwargs)

            model = get_model(train_X, train_Y, use_model_list, standardize_model)

            num_samples = 20

            optimal_inputs = torch.rand(num_samples, input_dim, **tkwargs)
            optimal_outputs = torch.rand(num_samples, num_objectives, **tkwargs)

            # test acquisition
            X_pending_list = [None, torch.rand(2, input_dim, **tkwargs)]
            for i in range(len(X_pending_list)):
                X_pending = X_pending_list[i]

                acq = qJointEntropySearch(
                    model=model,
                    optimal_inputs=optimal_inputs,
                    optimal_outputs=optimal_outputs,
                    estimation_type=estimation_type,
                    num_samples=64,
                    X_pending=X_pending,
                    condition_noiseless=condition_noiseless,
                    maximize=maximize,
                )
                self.assertIsInstance(acq.sampler, SobolQMCNormalSampler)

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
            acq = qJointEntropySearch(
                model=model,
                optimal_inputs=optimal_inputs,
                optimal_outputs=optimal_outputs,
                estimation_type="NO_EST",
                num_samples=64,
                X_pending=X_pending,
                condition_noiseless=condition_noiseless,
                maximize=maximize,
            )
            acq_X = acq(test_Xs[j])

        # Support with fully bayesian models is not yet implemented. Thus, we
        # throw an error for now.
        fully_bayesian_model = SaasFullyBayesianSingleTaskGP(train_X, train_Y)
        with self.assertRaises(NotImplementedError):
            acq = qJointEntropySearch(
                model=fully_bayesian_model,
                optimal_inputs=optimal_inputs,
                optimal_outputs=optimal_outputs,
                estimation_type="LB",
            )
