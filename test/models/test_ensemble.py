#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from botorch.models.ensemble import EnsembleModel
from botorch.utils.testing import BotorchTestCase


class DummyEnsembleModel(EnsembleModel):
    r"""A dummy ensemble model."""

    def __init__(self):
        r"""Init model."""
        super().__init__()
        self._num_outputs = 2
        self.a = torch.rand(4, 3, 2)

    def forward(self, X):
        return torch.stack(
            [torch.einsum("...d,dm", X, self.a[i]) for i in range(4)], dim=-3
        )


class TestEnsembleModels(BotorchTestCase):
    def test_abstract_base_model(self):
        with self.assertRaises(TypeError):
            EnsembleModel()

    def test_DummyEnsembleModel(self):
        for shape in [(10, 3), (5, 10, 3)]:
            e = DummyEnsembleModel()
            X = torch.randn(*shape)
            p = e.posterior(X)
            self.assertEqual(p.ensemble_size, 4)
