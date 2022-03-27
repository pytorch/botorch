#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from botorch.acquisition.preference import AnalyticExpectedUtilityOfBestOption
from botorch.exceptions.errors import UnsupportedError
from botorch.models import SingleTaskGP
from botorch.models.deterministic import FixedSingleSampleModel
from botorch.models.pairwise_gp import PairwiseGP
from botorch.utils.testing import BotorchTestCase


class TestAnalyticExpectedUtilityOfBestOption(BotorchTestCase):
    def test_analytic_eubo(self):
        twargs = {"dtype": torch.double}
        X_dim = 3
        Y_dim = 2
        X = torch.rand(2, X_dim, **twargs)
        Y = torch.rand(2, Y_dim, **twargs)
        comps = torch.tensor([[1, 0]], dtype=torch.long)

        standard_bounds = torch.zeros(2, X.shape[-1])
        standard_bounds[1] = 1

        model = SingleTaskGP(X, Y)
        pref_model = PairwiseGP(Y, comps)

        # Test with an outcome model and a preference model
        one_sample_outcome_model = FixedSingleSampleModel(model=model)
        eubo = AnalyticExpectedUtilityOfBestOption(
            pref_model=pref_model, outcome_model=one_sample_outcome_model
        )

        # test forward with different number of points
        good_X = torch.rand(2, X_dim, **twargs)
        eubo(good_X)

        bad_X = torch.rand(3, X_dim, **twargs)
        with self.assertRaises(UnsupportedError):
            eubo(bad_X)

        good_X = torch.rand(1, X_dim, **twargs)
        previous_winner = torch.rand(1, Y_dim, **twargs)
        eubo_with_winner = AnalyticExpectedUtilityOfBestOption(
            pref_model=pref_model,
            outcome_model=one_sample_outcome_model,
            previous_winner=previous_winner,
        )
        eubo_with_winner(good_X)

        # Test model=None
        AnalyticExpectedUtilityOfBestOption(pref_model=pref_model, outcome_model=None)
