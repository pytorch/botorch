#! /usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest import mock

import torch
from botorch.acquisition.analytic import PosteriorMean
from botorch.acquisition.cost_aware import GenericCostAwareUtility
from botorch.acquisition.knowledge_gradient import (
    _get_value_function,
    _split_fantasy_points,
    qKnowledgeGradient,
    qMultiFidelityKnowledgeGradient,
)
from botorch.acquisition.monte_carlo import qSimpleRegret
from botorch.acquisition.objective import GenericMCObjective, ScalarizedObjective
from botorch.exceptions.errors import UnsupportedError
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.sampling.samplers import IIDNormalSampler, SobolQMCNormalSampler
from botorch.utils.testing import BotorchTestCase, MockModel, MockPosterior
from gpytorch.distributions import MultitaskMultivariateNormal


NO = "botorch.utils.testing.MockModel.num_outputs"


class TestQMultiStepLookahead(BotorchTestCase):
    def test_step(self):
        raise NotImplementedError

    def test_initialize_q_multi_step_lookahead(self):
        raise NotImplementedError

    def test_evaluate_q_multi_step_lookahead(self):
        raise NotImplementedError
