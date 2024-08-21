# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from itertools import product

import torch
from botorch.acquisition.bayesian_active_learning import (
    FullyBayesianAcquisitionFunction,
    qBayesianActiveLearningByDisagreement,
)
from botorch.sampling.normal import IIDNormalSampler
from botorch.utils.test_helpers import get_fully_bayesian_model, get_model
from botorch.utils.testing import BotorchTestCase


class TestFullyBayesianActuisitionFunction(BotorchTestCase):
    def test_abstract_raises(self):
        with self.assertRaises(TypeError):
            FullyBayesianAcquisitionFunction()


class TestQBayesianActiveLearningByDisagreement(BotorchTestCase):
    def test_q_bayesian_active_learning_by_disagreement(self):
        torch.manual_seed(1)
        tkwargs = {"device": self.device}
        num_objectives = 1
        num_models = 3
        input_dim = 2

        X_pending_list = [None, torch.rand(2, input_dim)]
        for (
            dtype,
            standardize_model,
            infer_noise,
            X_pending,
        ) in product(
            (torch.float, torch.double),
            (False, True),  # standardize_model
            (True,),  # infer_noise - only one option avail in PyroModels
            X_pending_list,
        ):
            X_pending = X_pending.to(**tkwargs) if X_pending is not None else None
            tkwargs["dtype"] = dtype
            train_X = torch.rand(4, input_dim, **tkwargs)
            train_Y = torch.rand(4, num_objectives, **tkwargs)

            model = get_fully_bayesian_model(
                train_X,
                train_Y,
                num_models,
                standardize_model,
                infer_noise,
                **tkwargs,
            )

            # test acquisition
            acq = qBayesianActiveLearningByDisagreement(
                model=model,
                X_pending=X_pending,
            )

            acq2 = qBayesianActiveLearningByDisagreement(
                model=model, sampler=IIDNormalSampler(torch.Size([9]))
            )
            self.assertIsInstance(acq2.sampler, IIDNormalSampler)

            test_Xs = [
                torch.rand(4, 1, input_dim, **tkwargs),
                torch.rand(4, 3, input_dim, **tkwargs),
                torch.rand(4, 5, 1, input_dim, **tkwargs),
                torch.rand(4, 5, 3, input_dim, **tkwargs),
                torch.rand(5, 13, input_dim, **tkwargs),
            ]

            for j in range(len(test_Xs)):
                acq_X = acq.forward(test_Xs[j])
                acq_X = acq(test_Xs[j])
                # assess shape
                self.assertTrue(acq_X.shape == test_Xs[j].shape[:-2])

                self.assertTrue(torch.all(acq_X > 0))

        # Support with non-fully bayesian models is not possible. Thus, we
        # throw an error.
        non_fully_bayesian_model = get_model(train_X, train_Y, False)
        with self.assertRaisesRegex(
            ValueError,
            "Fully Bayesian acquisition functions require a "
            "SaasFullyBayesianSingleTaskGP to run.",
        ):
            acq = qBayesianActiveLearningByDisagreement(
                model=non_fully_bayesian_model,
            )
