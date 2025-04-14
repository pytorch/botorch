#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from botorch.posteriors import GPyTorchPosterior
from botorch.utils.testing import BotorchTestCase
from botorch_community.models.vblls import VBLLModel
from botorch_community.posteriors.bll_posterior import BLLPosterior
from gpytorch.distributions import MultitaskMultivariateNormal
from linear_operator.operators import to_linear_operator

from torch.distributions import MultivariateNormal


def _get_vbll_model(num_inputs: int = 2, num_hidden: int = 3, **tkwargs) -> VBLLModel:
    test_backbone = torch.nn.Sequential(
        torch.nn.Linear(num_inputs, num_hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(num_hidden, num_hidden),
    ).to(**tkwargs)
    model = VBLLModel(backbone=test_backbone, hidden_features=num_hidden)
    return model


class TestBLLPosterior(BotorchTestCase):
    def setUp(self):
        """Set up common test components."""
        super().setUp()
        self.tkwargs = {"dtype": torch.float64, "device": self.device}
        self.X = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], **self.tkwargs)
        self.num_X = self.X.shape[0]
        self.output_dim = 1
        self.model = _get_vbll_model(num_inputs=2, **self.tkwargs)

        # Create a proper MultivariateNormal distribution for GPyTorchPosterior
        mean = torch.tensor([1.0, 2.0, 3.0], **self.tkwargs)

        # Create a positive-definite covariance matrix
        cov_diag = torch.tensor([0.1, 0.2, 0.3], **self.tkwargs)
        cov = torch.diag(cov_diag)

        mvn_dist = MultivariateNormal(mean, cov)
        self.gpt_posterior = GPyTorchPosterior(mvn_dist)

        self.bll_posterior = BLLPosterior(
            posterior=self.gpt_posterior,
            model=self.model,
            X=self.X,
            output_dim=self.model.num_outputs,
        )

    def test_initialization(self):
        """Test that BLLPosterior initializes correctly."""
        self.assertEqual(self.bll_posterior.posterior, self.gpt_posterior)
        self.assertEqual(self.bll_posterior.model, self.model)
        self.assertEqual(self.bll_posterior.output_dim, self.output_dim)
        self.assertTrue(torch.equal(self.bll_posterior.X, self.X))

    def test_rsample_default(self):
        """Test rsample method with default sample_shape."""
        samples = self.bll_posterior.rsample()

        # Check shape: should be [1, num_X, output_dim]
        expected_shape = torch.Size([1, self.num_X, 1])
        self.assertEqual(samples.shape, expected_shape)

    def test_rsample_with_explicit_shape(self):
        """Test rsample method with custom sample_shape."""
        for sample_shape in ([3], [2, 3]):
            samples = self.bll_posterior.rsample(sample_shape=torch.Size(sample_shape))

            # Check shape: should be [sample_shape, num_X, output_dim]
            expected_shape = torch.Size(sample_shape + [self.num_X, 1])
            self.assertEqual(samples.shape, expected_shape)

    def test_mean(self):
        """Test mean property."""
        mean = self.bll_posterior.mean

        # Check shape: should be [num_X, output_dim]
        expected_shape = torch.Size([self.num_X, 1])
        self.assertEqual(mean.shape, expected_shape)

    def test_variance(self):
        """Test variance property."""
        variance = self.bll_posterior.variance

        # Check shape: should be [num_X, output_dim]
        expected_shape = torch.Size([self.num_X, 1])
        self.assertEqual(variance.shape, expected_shape)

    def test_device(self):
        """Test device property."""
        self.assertEqual(self.bll_posterior.device, self.device)

    def test_dtype(self):
        """Test dtype property."""
        self.assertEqual(self.bll_posterior.dtype, self.tkwargs["dtype"])

    def test_multi_output(self):
        """Test with multiple output dimensions."""
        output_dim = 2

        # Create a model with multiple outputs
        backbone = torch.nn.Sequential(
            torch.nn.Linear(2, 3),
            torch.nn.ReLU(),
            torch.nn.Linear(3, 3),
        ).to(**self.tkwargs)

        model = VBLLModel(backbone=backbone, hidden_features=3, out_features=output_dim)

        # Create mean tensor with shape [batch_size, output_dim]
        mean = torch.rand(3, 2, **self.tkwargs)
        variance = 1 + torch.rand(3, 2, **self.tkwargs)
        covar = variance.view(-1).diag()
        mvn = MultitaskMultivariateNormal(mean, to_linear_operator(covar))
        gpt_posterior = GPyTorchPosterior(mvn)

        bll_posterior = BLLPosterior(
            posterior=gpt_posterior, model=model, X=self.X, output_dim=output_dim
        )

        # Test rsample
        samples = bll_posterior.rsample()
        expected_shape = torch.Size(
            [1, self.num_X, output_dim]
        )  # [samples, batch, output_dim]
        self.assertEqual(samples.shape, expected_shape)

        # Test rsample with explicit shape
        for sample_shape in ([3], [2, 3]):
            samples = bll_posterior.rsample(sample_shape=torch.Size(sample_shape))

            # Check shape: should be [sample_shape, num_X, output_dim]
            expected_shape = torch.Size(sample_shape + [self.num_X, output_dim])
            self.assertEqual(samples.shape, expected_shape)

        # Test mean with multi-output
        multi_mean = bll_posterior.mean
        expected_mean_shape = torch.Size(
            [self.num_X, output_dim]
        )  # [batch, output_dim]
        self.assertEqual(multi_mean.shape, expected_mean_shape)

        # check mean values
        self.assertTrue(torch.allclose(multi_mean, mean))

        # Test variance with multi-output
        multi_variance = bll_posterior.variance
        expected_var_shape = torch.Size([self.num_X, output_dim])  # [batch, output_dim]
        self.assertEqual(multi_variance.shape, expected_var_shape)

        # check variance values
        expected_variance = variance
        self.assertTrue(torch.allclose(multi_variance, expected_variance))
