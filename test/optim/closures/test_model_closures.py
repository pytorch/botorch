#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from itertools import zip_longest
from math import pi

import torch
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.optim.closures.model_closures import (
    get_loss_closure,
    get_loss_closure_with_grads,
)
from botorch.utils.testing import BotorchTestCase
from gpytorch import settings as gpytorch_settings
from gpytorch.mlls import ExactMarginalLogLikelihood, SumMarginalLogLikelihood
from torch.utils.data import DataLoader, TensorDataset


class TestLossClosures(BotorchTestCase):
    def setUp(self):
        super().setUp()
        with torch.random.fork_rng():
            torch.manual_seed(0)
            train_X = torch.linspace(0, 1, 10).unsqueeze(-1)
            train_Y = torch.sin((2 * pi) * train_X)
            train_Y = train_Y + 0.1 * torch.randn_like(train_Y)

        self.mlls = {}
        model = SingleTaskGP(
            train_X=train_X,
            train_Y=train_Y,
            input_transform=Normalize(d=1),
            outcome_transform=Standardize(m=1),
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        self.mlls[type(mll), type(model.likelihood), type(model)] = mll.to(self.device)

        model = ModelListGP(model, model)
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        self.mlls[type(mll), type(model.likelihood), type(model)] = mll.to(self.device)

    def test_main(self):
        for mll in self.mlls.values():
            out = mll.model(*mll.model.train_inputs)
            loss = -mll(out, mll.model.train_targets).sum()
            loss.backward()
            params = {n: p for n, p in mll.named_parameters() if p.requires_grad}
            grads = [
                torch.zeros_like(p) if p.grad is None else p.grad
                for p in params.values()
            ]

            closure = get_loss_closure(mll)
            self.assertTrue(loss.equal(closure()))

            closure = get_loss_closure_with_grads(mll, params)
            _loss, _grads = closure()
            self.assertTrue(loss.equal(_loss))
            self.assertTrue(all(a.equal(b) for a, b in zip_longest(grads, _grads)))

    def test_data_loader(self):
        for mll in self.mlls.values():
            if type(mll) != ExactMarginalLogLikelihood:
                continue

            dataset = TensorDataset(*mll.model.train_inputs, mll.model.train_targets)
            loader = DataLoader(dataset, batch_size=len(mll.model.train_targets))
            params = {n: p for n, p in mll.named_parameters() if p.requires_grad}
            A = get_loss_closure_with_grads(mll, params)
            (a, das) = A()

            B = get_loss_closure_with_grads(mll, params, data_loader=loader)
            with gpytorch_settings.debug(False):  # disables GPyTorch's internal check
                (b, dbs) = B()

            self.assertTrue(a.allclose(b))
            for da, db in zip_longest(das, dbs):
                self.assertTrue(da.allclose(db))

        loader = DataLoader(mll.model.train_targets, len(mll.model.train_targets))
        closure = get_loss_closure_with_grads(mll, params, data_loader=loader)
        with self.assertRaisesRegex(TypeError, "Expected .* a batch of tensors"):
            closure()
