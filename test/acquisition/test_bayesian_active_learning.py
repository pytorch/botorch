# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from itertools import product

import torch
from botorch.acquisition.bayesian_active_learning import (
    check_negative_info_gain,
    FULLY_BAYESIAN_ERROR_MSG,
    FullyBayesianAcquisitionFunction,
    NEGATIVE_INFOGAIN_WARNING,
    qBayesianActiveLearningByDisagreement,
)
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.sampling.normal import IIDNormalSampler
from botorch.utils.test_helpers import (
    get_fully_bayesian_model,
    get_fully_bayesian_model_list,
    get_model,
)
from botorch.utils.testing import BotorchTestCase


class TestFullyBayesianActuisitionFunction(BotorchTestCase):
    def test_abstract_raises(self):
        with self.assertRaises(TypeError):
            FullyBayesianAcquisitionFunction()

    def test_fully_bayesian_model_can_init(self):
        tkwargs = {"device": self.device, "dtype": torch.double}
        train_X = torch.rand(4, 2, **tkwargs)
        train_Y = torch.rand(4, 1, **tkwargs)

        acq = qBayesianActiveLearningByDisagreement(
            model=get_fully_bayesian_model(
                train_X=train_X,
                train_Y=train_Y,
                num_models=3,
                standardize_model=True,
                infer_noise=True,
                **tkwargs,
            ),
        )
        acq_with_model_list = qBayesianActiveLearningByDisagreement(
            get_fully_bayesian_model_list(
                train_X=train_X,
                train_Y=train_Y,
                num_models=3,
                standardize_model=True,
                infer_noise=True,
                **tkwargs,
            ),
        )
        self.assertIsInstance(acq.model, SaasFullyBayesianSingleTaskGP)
        self.assertIsInstance(acq_with_model_list.model, ModelListGP)

        # Support with non-fully bayesian models is not possible. Thus, we
        # throw an error.

    def test_non_fully_bayesian_model_raises(self):
        train_X = torch.rand(4, 2)
        train_Y = torch.rand(4, 1)
        non_fully_bayesian_model = get_model(train_X=train_X, train_Y=train_Y)
        with self.assertRaisesRegex(
            RuntimeError,
            FULLY_BAYESIAN_ERROR_MSG,
        ):
            qBayesianActiveLearningByDisagreement(model=non_fully_bayesian_model)


class TestQBayesianActiveLearningByDisagreement(BotorchTestCase):
    def test_check_negative_info_gain(self):
        negative_ig_tensor = torch.tensor([0.0, 1.2, -3.0])
        with self.assertWarnsRegex(RuntimeWarning, NEGATIVE_INFOGAIN_WARNING):
            check_negative_info_gain(negative_ig_tensor)

        # test that it doesn't raise a warning if the tensor is all positive
        positive_ig_tensor = torch.tensor([0.01, 1.2, 3.0])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            check_negative_info_gain(positive_ig_tensor)

    def _test_q_bayesian_active_learning_by_disagreement_base(self, model_getter):
        torch.manual_seed(1)
        tkwargs = {"device": self.device}
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
            train_Y = torch.rand(4, 1, **tkwargs)

            model = model_getter(
                train_X=train_X,
                train_Y=train_Y,
                num_models=num_models,
                standardize_model=standardize_model,
                infer_noise=infer_noise,
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

                # for a model with sensible lengthscales (such as the ones sampled)
                # this test should always pass since the EIG is always postitive
                # in theory.
                self.assertTrue(torch.all(acq_X > 0))

    def test_q_bayesian_active_learning_by_disagreement_multi_objective(self):
        self._test_q_bayesian_active_learning_by_disagreement_base(
            model_getter=get_fully_bayesian_model_list
        )

    def test_q_bayesian_active_learning_by_disagreement_single_objective(self):
        self._test_q_bayesian_active_learning_by_disagreement_base(
            model_getter=get_fully_bayesian_model
        )
