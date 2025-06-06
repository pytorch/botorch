#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from botorch.utils.testing import BotorchTestCase
from botorch_community.models.example import ExampleModel
from gpytorch.kernels import RBFKernel, ScaleKernel


class TestExampleModel(BotorchTestCase):
    def test_example_gp(self) -> None:
        model = ExampleModel(
            train_X=torch.rand(2, 2),
            train_Y=torch.rand(2, 1),
        )
        self.assertIsInstance(model.covar_module, ScaleKernel)
        self.assertIsInstance(model.covar_module.base_kernel, RBFKernel)
