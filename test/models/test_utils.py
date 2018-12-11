#! /usr/bin/env python3

import unittest
from copy import deepcopy

import torch
from botorch.models import SingleTaskGP
from botorch.models.fantasy_utils import _get_fantasy_state, _load_fantasy_state_dict
from gpytorch.likelihoods import GaussianLikelihood


class BotorchModelUtilsTest(unittest.TestCase):
    def setUp(self):
        self.train_x = torch.randn(2, 3)
        train_y = torch.randn(2)

        likelihood = GaussianLikelihood()
        model = SingleTaskGP(self.train_x, train_y, likelihood).eval()

        # choose some random parameter values
        for _, v in model.named_parameters():
            v = torch.randn_like(v).requires_grad_(True)
        for _, v in model.named_buffers():
            v = torch.randn_like(v).requires_grad_(True)

        self.model = model
        self.likelihood = likelihood
        self.X = torch.randn(2, 3)

    def test_get_fantasy_state(self):
        num_samples = 2
        state_dict, train_X, train_Y = _get_fantasy_state(
            model=self.model, X=self.X, num_samples=num_samples
        )
        # make sure we correctly extracted the state dict
        model_state_dict = self.model.state_dict()
        self.assertTrue(set(state_dict.keys()) == set(model_state_dict.keys()))
        self.assertTrue(
            all(torch.equal(model_state_dict[k], v) for k, v in state_dict.items())
        )

        # make sure the training data for the fantasized model is correct
        self.assertTrue(
            torch.equal(
                train_X, torch.cat([self.train_x, self.X]).expand(num_samples, -1, 3)
            )
        )

    def test_load_fantasy_state_dict(self):
        state_dict, train_X, train_Y = _get_fantasy_state(
            model=self.model, X=self.X, num_samples=2
        )
        fantasy_model = SingleTaskGP(
            train_X=train_X, train_Y=train_Y, likelihood=deepcopy(self.likelihood)
        )
        fantasy_model = _load_fantasy_state_dict(
            model=fantasy_model, state_dict=state_dict
        )

        # check that parameters were updated
        for k, v in fantasy_model.named_parameters():
            self.assertTrue(torch.equal(state_dict[k].expand_as(v), v))
        for k, v in fantasy_model.named_buffers():
            self.assertTrue(torch.equal(state_dict[k].expand_as(v), v))
