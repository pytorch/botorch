#! /usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

import torch
from botorch.models.gpytorch import (
    BatchedMultiOutputGPyTorchModel,
    GPyTorchModel,
    ModelListGPyTorchModel,
)
from botorch.models.utils import multioutput_to_batch_mode_transform
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.sampling.samplers import SobolQMCNormalSampler
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP, IndependentModelList


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


class SimpleBatchedMultiOutputGPyTorchModel(ExactGP, BatchedMultiOutputGPyTorchModel):
    def __init__(self, train_X, train_Y):
        train_X, train_Y, _ = self._set_dimensions(train_X=train_X, train_Y=train_Y)
        train_X, train_Y, _ = multioutput_to_batch_mode_transform(
            train_X=train_X, train_Y=train_Y, num_outputs=self._num_outputs
        )
        likelihood = GaussianLikelihood(batch_shape=self._aug_batch_shape)
        super().__init__(train_X, train_Y, likelihood)
        self.mean_module = ConstantMean(batch_shape=self._aug_batch_shape)
        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=self._aug_batch_shape),
            batch_shape=self._aug_batch_shape,
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class SimpleModelListGPyTorchModel(IndependentModelList, ModelListGPyTorchModel):
    def __init__(self, *gp_models: GPyTorchModel):
        super().__init__(*gp_models)


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
        # test conditioning on observations
        cm = model.condition_on_observations(torch.rand(2, 1), torch.rand(2))
        self.assertIsInstance(cm, SimpleGPyTorchModel)
        self.assertEqual(cm.train_targets.shape, torch.Size([7]))
        # test fantasize
        sampler = SobolQMCNormalSampler(num_samples=2)
        cm = model.fantasize(torch.rand(2, 1), sampler=sampler)
        self.assertIsInstance(cm, SimpleGPyTorchModel)
        self.assertEqual(cm.train_targets.shape, torch.Size([2, 7]))
        cm = model.fantasize(torch.rand(2, 1), sampler=sampler, observation_noise=True)
        self.assertIsInstance(cm, SimpleGPyTorchModel)
        self.assertEqual(cm.train_targets.shape, torch.Size([2, 7]))


class TestBatchedMultiOutputGPyTorchModel(unittest.TestCase):
    def test_batched_multi_output_gpytorch_model(self):
        train_X = torch.rand(5, 1)
        train_Y = torch.cat([torch.sin(train_X), torch.cos(train_X)], dim=-1)
        # basic test
        model = SimpleBatchedMultiOutputGPyTorchModel(train_X, train_Y)
        test_X = torch.rand(2, 1)
        posterior = model.posterior(test_X)
        self.assertIsInstance(posterior, GPyTorchPosterior)
        self.assertEqual(posterior.mean.shape, torch.Size([2, 2]))
        # test observation noise
        posterior = model.posterior(test_X, observation_noise=True)
        self.assertIsInstance(posterior, GPyTorchPosterior)
        self.assertEqual(posterior.mean.shape, torch.Size([2, 2]))
        # test conditioning on observations
        cm = model.condition_on_observations(torch.rand(2, 1), torch.rand(2, 2))
        self.assertIsInstance(cm, SimpleBatchedMultiOutputGPyTorchModel)
        self.assertEqual(cm.train_targets.shape, torch.Size([2, 7]))
        # test fantasize
        sampler = SobolQMCNormalSampler(num_samples=2)
        cm = model.fantasize(torch.rand(2, 1), sampler=sampler)
        self.assertIsInstance(cm, SimpleBatchedMultiOutputGPyTorchModel)
        self.assertEqual(cm.train_targets.shape, torch.Size([2, 2, 7]))
        cm = model.fantasize(torch.rand(2, 1), sampler=sampler, observation_noise=True)
        self.assertIsInstance(cm, SimpleBatchedMultiOutputGPyTorchModel)
        self.assertEqual(cm.train_targets.shape, torch.Size([2, 2, 7]))


class TestModelListGPyTorchModel(unittest.TestCase):
    def test_model_list_gpytorch_model(self):
        train_X1, train_X2 = torch.rand(5, 1), torch.rand(5, 1)
        train_Y1 = torch.sin(train_X1).squeeze()
        train_Y2 = torch.cos(train_X2).squeeze()
        m1 = SimpleGPyTorchModel(train_X1, train_Y1)
        m2 = SimpleGPyTorchModel(train_X2, train_Y2)
        model = SimpleModelListGPyTorchModel(m1, m2)
        test_X = torch.rand(2, 1)
        posterior = model.posterior(test_X)
        self.assertIsInstance(posterior, GPyTorchPosterior)
        self.assertEqual(posterior.mean.shape, torch.Size([2, 2]))
        # test observation noise
        posterior = model.posterior(test_X, observation_noise=True)
        self.assertIsInstance(posterior, GPyTorchPosterior)
        self.assertEqual(posterior.mean.shape, torch.Size([2, 2]))
        # conditioning is not implemented (see ModelListGP for tests)
        with self.assertRaises(NotImplementedError):
            model.condition_on_observations(X=torch.rand(2, 1), Y=torch.rand(2, 2))
