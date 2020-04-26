#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools

import torch
from botorch import settings
from botorch.exceptions import (
    BotorchTensorDimensionError,
    BotorchTensorDimensionWarning,
)
from botorch.models.gpytorch import (
    BatchedMultiOutputGPyTorchModel,
    GPyTorchModel,
    ModelListGPyTorchModel,
)
from botorch.models.transforms import Standardize
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.utils.testing import BotorchTestCase
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP, IndependentModelList


class SimpleGPyTorchModel(GPyTorchModel, ExactGP):
    def __init__(self, train_X, train_Y, outcome_transform=None):
        if outcome_transform is not None:
            train_Y, _ = outcome_transform(train_Y)
        self._validate_tensor_args(train_X, train_Y)
        train_Y = train_Y.squeeze(-1)
        likelihood = GaussianLikelihood()
        super().__init__(train_X, train_Y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel())
        if outcome_transform is not None:
            self.outcome_transform = outcome_transform
        self._num_outputs = 1
        self.to(train_X)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class SimpleBatchedMultiOutputGPyTorchModel(BatchedMultiOutputGPyTorchModel, ExactGP):
    def __init__(self, train_X, train_Y):
        self._validate_tensor_args(train_X, train_Y)
        self._set_dimensions(train_X=train_X, train_Y=train_Y)
        train_X, train_Y, _ = self._transform_tensor_args(X=train_X, Y=train_Y)
        likelihood = GaussianLikelihood(batch_shape=self._aug_batch_shape)
        super().__init__(train_X, train_Y, likelihood)
        self.mean_module = ConstantMean(batch_shape=self._aug_batch_shape)
        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=self._aug_batch_shape),
            batch_shape=self._aug_batch_shape,
        )
        self.to(train_X)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class SimpleModelListGPyTorchModel(IndependentModelList, ModelListGPyTorchModel):
    def __init__(self, *gp_models: GPyTorchModel):
        super().__init__(*gp_models)


class TestGPyTorchModel(BotorchTestCase):
    def test_gpytorch_model(self):
        for dtype, use_octf in itertools.product(
            (torch.float, torch.double), (False, True)
        ):
            tkwargs = {"device": self.device, "dtype": dtype}
            octf = Standardize(m=1) if use_octf else None
            train_X = torch.rand(5, 1, **tkwargs)
            train_Y = torch.sin(train_X)
            # basic test
            model = SimpleGPyTorchModel(train_X, train_Y, octf)
            self.assertEqual(model.num_outputs, 1)
            test_X = torch.rand(2, 1, **tkwargs)
            posterior = model.posterior(test_X)
            self.assertIsInstance(posterior, GPyTorchPosterior)
            self.assertEqual(posterior.mean.shape, torch.Size([2, 1]))
            if use_octf:
                # ensure un-transformation is applied
                tmp_tf = model.outcome_transform
                del model.outcome_transform
                p_tf = model.posterior(test_X)
                model.outcome_transform = tmp_tf
                expected_var = tmp_tf.untransform_posterior(p_tf).variance
                self.assertTrue(torch.allclose(posterior.variance, expected_var))
            # test observation noise
            posterior = model.posterior(test_X, observation_noise=True)
            self.assertIsInstance(posterior, GPyTorchPosterior)
            self.assertEqual(posterior.mean.shape, torch.Size([2, 1]))
            posterior = model.posterior(
                test_X, observation_noise=torch.rand(2, 1, **tkwargs)
            )
            self.assertIsInstance(posterior, GPyTorchPosterior)
            self.assertEqual(posterior.mean.shape, torch.Size([2, 1]))
            # test noise shape validation
            with self.assertRaises(BotorchTensorDimensionError):
                model.posterior(test_X, observation_noise=torch.rand(2, **tkwargs))
            # test conditioning on observations
            cm = model.condition_on_observations(
                torch.rand(2, 1, **tkwargs), torch.rand(2, 1, **tkwargs)
            )
            self.assertIsInstance(cm, SimpleGPyTorchModel)
            self.assertEqual(cm.train_targets.shape, torch.Size([7]))
            # test subset_output
            with self.assertRaises(NotImplementedError):
                model.subset_output([0])
            # test fantasize
            sampler = SobolQMCNormalSampler(num_samples=2)
            cm = model.fantasize(torch.rand(2, 1, **tkwargs), sampler=sampler)
            self.assertIsInstance(cm, SimpleGPyTorchModel)
            self.assertEqual(cm.train_targets.shape, torch.Size([2, 7]))
            cm = model.fantasize(
                torch.rand(2, 1, **tkwargs), sampler=sampler, observation_noise=True
            )
            self.assertIsInstance(cm, SimpleGPyTorchModel)
            self.assertEqual(cm.train_targets.shape, torch.Size([2, 7]))
            cm = model.fantasize(
                torch.rand(2, 1, **tkwargs),
                sampler=sampler,
                observation_noise=torch.rand(2, 1, **tkwargs),
            )
            self.assertIsInstance(cm, SimpleGPyTorchModel)
            self.assertEqual(cm.train_targets.shape, torch.Size([2, 7]))

    def test_validate_tensor_args(self):
        n, d = 3, 2
        for batch_shape, output_dim_shape, dtype in itertools.product(
            (torch.Size(), torch.Size([2])),
            (torch.Size(), torch.Size([1]), torch.Size([2])),
            (torch.float, torch.double),
        ):
            tkwargs = {"device": self.device, "dtype": dtype}
            X = torch.empty(batch_shape + torch.Size([n, d]), **tkwargs)
            # test using the same batch_shape as X
            Y = torch.empty(batch_shape + torch.Size([n]) + output_dim_shape, **tkwargs)
            if len(output_dim_shape) > 0:
                # check that no exception is raised
                GPyTorchModel._validate_tensor_args(X, Y)
                with settings.debug(True), self.assertWarns(
                    BotorchTensorDimensionWarning
                ):
                    GPyTorchModel._validate_tensor_args(X, Y, strict=False)
            else:
                with self.assertRaises(BotorchTensorDimensionError):
                    GPyTorchModel._validate_tensor_args(X, Y)
                with settings.debug(True), self.assertWarns(
                    BotorchTensorDimensionWarning
                ):
                    GPyTorchModel._validate_tensor_args(X, Y, strict=False)
            # test using different batch_shape
            if len(batch_shape) > 0:
                with self.assertRaises(BotorchTensorDimensionError):
                    GPyTorchModel._validate_tensor_args(X, Y[0])
                with settings.debug(True), self.assertWarns(
                    BotorchTensorDimensionWarning
                ):
                    GPyTorchModel._validate_tensor_args(X, Y[0], strict=False)


class TestBatchedMultiOutputGPyTorchModel(BotorchTestCase):
    def test_batched_multi_output_gpytorch_model(self):
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            train_X = torch.rand(5, 1, **tkwargs)
            train_Y = torch.cat([torch.sin(train_X), torch.cos(train_X)], dim=-1)
            # basic test
            model = SimpleBatchedMultiOutputGPyTorchModel(train_X, train_Y)
            self.assertEqual(model.num_outputs, 2)
            test_X = torch.rand(2, 1, **tkwargs)
            posterior = model.posterior(test_X)
            self.assertIsInstance(posterior, GPyTorchPosterior)
            self.assertEqual(posterior.mean.shape, torch.Size([2, 2]))
            # test observation noise
            posterior = model.posterior(test_X, observation_noise=True)
            self.assertIsInstance(posterior, GPyTorchPosterior)
            self.assertEqual(posterior.mean.shape, torch.Size([2, 2]))
            posterior = model.posterior(
                test_X, observation_noise=torch.rand(2, 2, **tkwargs)
            )
            self.assertIsInstance(posterior, GPyTorchPosterior)
            self.assertEqual(posterior.mean.shape, torch.Size([2, 2]))
            # test subset_output
            with self.assertRaises(NotImplementedError):
                model.subset_output([0])
            # test conditioning on observations
            cm = model.condition_on_observations(
                torch.rand(2, 1, **tkwargs), torch.rand(2, 2, **tkwargs)
            )
            self.assertIsInstance(cm, SimpleBatchedMultiOutputGPyTorchModel)
            self.assertEqual(cm.train_targets.shape, torch.Size([2, 7]))
            # test fantasize
            sampler = SobolQMCNormalSampler(num_samples=2)
            cm = model.fantasize(torch.rand(2, 1, **tkwargs), sampler=sampler)
            self.assertIsInstance(cm, SimpleBatchedMultiOutputGPyTorchModel)
            self.assertEqual(cm.train_targets.shape, torch.Size([2, 2, 7]))
            cm = model.fantasize(
                torch.rand(2, 1, **tkwargs), sampler=sampler, observation_noise=True
            )
            self.assertIsInstance(cm, SimpleBatchedMultiOutputGPyTorchModel)
            self.assertEqual(cm.train_targets.shape, torch.Size([2, 2, 7]))
            cm = model.fantasize(
                torch.rand(2, 1, **tkwargs),
                sampler=sampler,
                observation_noise=torch.rand(2, 2, **tkwargs),
            )
            self.assertIsInstance(cm, SimpleBatchedMultiOutputGPyTorchModel)
            self.assertEqual(cm.train_targets.shape, torch.Size([2, 2, 7]))

            # test get_batch_dimensions
            get_batch_dims = SimpleBatchedMultiOutputGPyTorchModel.get_batch_dimensions
            for input_batch_dim in (0, 3):
                for num_outputs in (1, 2):
                    input_batch_shape, aug_batch_shape = get_batch_dims(
                        train_X=train_X.unsqueeze(0).expand(3, 5, 1)
                        if input_batch_dim == 3
                        else train_X,
                        train_Y=train_Y[:, 0:1] if num_outputs == 1 else train_Y,
                    )
                    expected_input_batch_shape = (
                        torch.Size([3]) if input_batch_dim == 3 else torch.Size([])
                    )
                    self.assertEqual(input_batch_shape, expected_input_batch_shape)
                    self.assertEqual(
                        aug_batch_shape,
                        expected_input_batch_shape + torch.Size([])
                        if num_outputs == 1
                        else expected_input_batch_shape + torch.Size([2]),
                    )


class TestModelListGPyTorchModel(BotorchTestCase):
    def test_model_list_gpytorch_model(self):
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            train_X1, train_X2 = (
                torch.rand(5, 1, **tkwargs),
                torch.rand(5, 1, **tkwargs),
            )
            train_Y1 = torch.sin(train_X1)
            train_Y2 = torch.cos(train_X2)
            m1 = SimpleGPyTorchModel(train_X1, train_Y1)
            m2 = SimpleGPyTorchModel(train_X2, train_Y2)
            model = SimpleModelListGPyTorchModel(m1, m2)
            self.assertEqual(model.num_outputs, 2)
            test_X = torch.rand(2, 1, **tkwargs)
            posterior = model.posterior(test_X)
            self.assertIsInstance(posterior, GPyTorchPosterior)
            self.assertEqual(posterior.mean.shape, torch.Size([2, 2]))
            # test output indices
            posterior = model.posterior(test_X, output_indices=[0])
            self.assertIsInstance(posterior, GPyTorchPosterior)
            self.assertEqual(posterior.mean.shape, torch.Size([2, 1]))
            # test observation noise
            posterior = model.posterior(test_X, observation_noise=True)
            self.assertIsInstance(posterior, GPyTorchPosterior)
            self.assertEqual(posterior.mean.shape, torch.Size([2, 2]))
            posterior = model.posterior(
                test_X, observation_noise=torch.rand(2, **tkwargs)
            )
            self.assertIsInstance(posterior, GPyTorchPosterior)
            self.assertEqual(posterior.mean.shape, torch.Size([2, 2]))
            posterior = model.posterior(
                test_X, output_indices=[0], observation_noise=torch.rand(2, **tkwargs)
            )
            self.assertIsInstance(posterior, GPyTorchPosterior)
            self.assertEqual(posterior.mean.shape, torch.Size([2, 1]))
            # conditioning is not implemented (see ModelListGP for tests)
            with self.assertRaises(NotImplementedError):
                model.condition_on_observations(
                    X=torch.rand(2, 1, **tkwargs), Y=torch.rand(2, 2, **tkwargs)
                )
