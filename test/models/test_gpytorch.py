#! /usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

import torch
from botorch.exceptions.errors import UnsupportedError
from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel, GPyTorchModel
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP
from torch import Tensor


class SimpleGPyTorchModel(ExactGP, GPyTorchModel):
    def __init__(self, train_X, train_Y):
        likelihood = GaussianLikelihood()
        super().__init__(train_X, train_Y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class TestGPyTorchModel(unittest.TestCase):
    def test_gpytorch_model(self):
        train_X = torch.rand(5, 1)
        train_Y = torch.sin(train_X.squeeze())
        # basic test
        model = SimpleGPyTorchModel(train_X, train_Y)
        test_X = torch.rand(2, 1)
        posterior = model.posterior(test_X)
        self.assertIsInstance(posterior, GPyTorchPosterior)
        self.assertEqual(posterior.mean.shape, torch.Size([2, 1]))
        # test observation noise
        posterior = model.posterior(test_X, observation_noise=True)
        self.assertIsInstance(posterior, GPyTorchPosterior)
        self.assertEqual(posterior.mean.shape, torch.Size([2, 1]))


class DummyParentModel:
    def get_fantasy_model(self, inputs: Tensor, targets: Tensor, **kwargs):
        raise AttributeError


class DummyModel(BatchedMultiOutputGPyTorchModel, DummyParentModel):
    def __init__(self):
        self._num_outputs = 1


class TestBatchedMultiOutputGPyTorchModel(unittest.TestCase):
    def test_get_fantasy_model_errors(self):
        model = DummyModel()
        fant_inputs = torch.zeros(2, 1)
        fant_targets = torch.zeros(2)
        # test when parent has `get_fantasy_model` method
        with self.assertRaises(AttributeError):
            model.get_fantasy_model(fant_inputs, fant_targets)
        # test when parent does not have `get_fantasy_model` method
        model2 = BatchedMultiOutputGPyTorchModel()
        model2._num_outputs = 1
        with self.assertRaises(UnsupportedError):
            model2.get_fantasy_model(fant_inputs, fant_targets)
