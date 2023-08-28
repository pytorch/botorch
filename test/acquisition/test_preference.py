#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.preference import (
    AnalyticExpectedUtilityOfBestOption,
    PairwiseBayesianActiveLearningByDisagreement,
)
from botorch.exceptions.errors import UnsupportedError
from botorch.models import SingleTaskGP
from botorch.models.deterministic import FixedSingleSampleModel
from botorch.models.pairwise_gp import PairwiseGP
from botorch.utils.testing import BotorchTestCase


class TestPreferenceAcquisitionFunctions(BotorchTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.twargs = {"dtype": torch.double}
        self.X_dim = 3
        self.Y_dim = 2
        X = torch.rand(2, self.X_dim, **self.twargs)
        Y = torch.rand(2, self.Y_dim, **self.twargs)
        comps = torch.tensor([[1, 0]], dtype=torch.long)

        self.model = SingleTaskGP(X, Y)
        self.pref_model_on_X = PairwiseGP(X, comps)
        self.pref_model_on_Y = PairwiseGP(Y, comps)
        self.deterministic_model = FixedSingleSampleModel(model=self.model)

    def pairwise_preference_acqf_test(
        self, acqf_class: AcquisitionFunction, test_previous_winner: bool
    ):
        for outcome_model in [self.deterministic_model, None]:
            pref_model = (
                self.pref_model_on_X if outcome_model is None else self.pref_model_on_Y
            )
            # Test with an outcome model and a preference model
            acqf = acqf_class(pref_model=pref_model, outcome_model=outcome_model)

            # test forward with different number of points
            X1 = torch.rand(1, self.X_dim, **self.twargs)
            X2 = torch.rand(2, self.X_dim, **self.twargs)
            X3 = torch.rand(3, self.X_dim, **self.twargs)

            # q = 1
            with self.assertRaises((UnsupportedError, AssertionError)):
                acqf(X1)
            # q = 2
            acqf(X2)
            # q > 2
            with self.assertRaises((UnsupportedError, AssertionError)):
                acqf(X3)

            if test_previous_winner:
                previous_winner = (
                    torch.rand(1, self.X_dim, **self.twargs)
                    if outcome_model is None
                    else torch.rand(1, self.Y_dim, **self.twargs)
                )
                acqf = acqf_class(
                    pref_model=pref_model,
                    outcome_model=outcome_model,
                    previous_winner=previous_winner,
                )
                # q = 1
                acqf(X1)
                # q = 2
                with self.assertRaises((UnsupportedError, AssertionError)):
                    acqf(X2)
                # q > 2
                with self.assertRaises((UnsupportedError, AssertionError)):
                    acqf(X3)

    def test_analytic_eubo(self):
        self.pairwise_preference_acqf_test(
            acqf_class=AnalyticExpectedUtilityOfBestOption,
            test_previous_winner=True,
        )

    def test_analytic_bald(self):
        self.pairwise_preference_acqf_test(
            acqf_class=PairwiseBayesianActiveLearningByDisagreement,
            test_previous_winner=False,
        )
