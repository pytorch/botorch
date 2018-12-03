#! /usr/bin/env python3

import unittest

import torch
from botorch.models import SingleTaskGP
from botorch.models.utils import initialize_batch_fantasy_GP
from gpytorch.likelihoods import GaussianLikelihood


class BotorchModelUtilsTest(unittest.TestCase):
    def test_initialize_batch_fantasy_GP(self):
        train_x = torch.randn(2, 3)
        train_y = torch.randn(2)

        likelihood = GaussianLikelihood()
        model = SingleTaskGP(train_x, train_y, likelihood).eval()

        # choose some random parameter values
        for _, v in model.named_parameters():
            v = torch.randn_like(v).requires_grad_(True)
        for _, v in model.named_buffers():
            v = torch.randn_like(v).requires_grad_(True)

        X = torch.randn(2, 3)
        num_samples = 2
        fantasy_model = initialize_batch_fantasy_GP(
            model=model, X=X, num_samples=num_samples
        )

        # check that parameters were updated
        state_dict = model.state_dict()
        for k, v in fantasy_model.named_parameters():
            self.assertTrue(torch.equal(state_dict[k].expand_as(v), v))
        for k, v in fantasy_model.named_buffers():
            self.assertTrue(torch.equal(state_dict[k].expand_as(v), v))

        # make sure the training data is correct
        self.assertTrue(
            torch.equal(
                fantasy_model.train_inputs[0],
                torch.cat([train_x, X]).expand(num_samples, -1, 3),
            )
        )
