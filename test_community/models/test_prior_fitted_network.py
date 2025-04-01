#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
from unittest.mock import MagicMock, mock_open, patch

import torch
from botorch.acquisition.objective import ScalarizedPosteriorTransform
from botorch.exceptions.errors import UnsupportedError
from botorch.utils.testing import BotorchTestCase
from botorch_community.models.prior_fitted_network import PFNModel
from botorch_community.models.utils.prior_fitted_network import (
    download_model,
    ModelPaths,
)
from torch import nn, Tensor


class DummyPFN(nn.Module):
    def __init__(self, n_buckets: int = 1000):
        """A dummy PFN model for testing purposes.

        This class implements a mocked PFN model that returns
        constant values for testing. It mimics the interface of actual PFN models
        but with simplified behavior.

        Args:
            n_buckets: Number of buckets for the output distribution. Default is 1000.
        """

        super().__init__()
        self.n_buckets = n_buckets
        self.criterion = MagicMock()
        self.criterion.borders = torch.linspace(0, 1, n_buckets + 1)

    def forward(self, train_X: Tensor, train_Y: Tensor, test_X: Tensor) -> Tensor:
        return torch.zeros(*test_X.shape[:-1], self.n_buckets, device=test_X.device)


class TestPriorFittedNetwork(BotorchTestCase):
    def test_raises(self):
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            train_X = torch.rand(10, 3, **tkwargs)
            train_Y = torch.rand(10, 1, **tkwargs)
            test_X = torch.rand(5, 3, **tkwargs)

            pfn = PFNModel(train_X, train_Y, DummyPFN())

            with self.assertRaises(RuntimeError):
                pfn.posterior(test_X, output_indices=[0, 1])
            with self.assertRaises(UnsupportedError):
                pfn.posterior(test_X, observation_noise=True)
            with self.assertRaises(UnsupportedError):
                pfn.posterior(
                    test_X,
                    posterior_transform=ScalarizedPosteriorTransform(
                        weights=torch.ones(1)
                    ),
                )

            # q should be 1
            test_X = torch.rand(5, 4, 2, **tkwargs)
            with self.assertRaises(NotImplementedError):
                pfn.posterior(test_X)

    def test_shapes(self):
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}

            # no batch dimension
            train_X = torch.rand(10, 3, **tkwargs)
            train_Y = torch.rand(10, 1, **tkwargs)
            test_X = torch.rand(5, 3, **tkwargs)

            pfn = PFNModel(train_X, train_Y, DummyPFN(n_buckets=100))
            posterior = pfn.posterior(test_X)

            self.assertEqual(posterior.probabilities.shape, torch.Size([5, 100]))
            self.assertEqual(posterior.mean.shape, torch.Size([5, 1]))

            # batch dimensions
            train_X = torch.rand(2, 3, 4, 10, 3, **tkwargs)
            train_Y = torch.rand(2, 3, 4, 10, 1, **tkwargs)
            test_X = torch.rand(2, 3, 4, 1, 3, **tkwargs)

            pfn = PFNModel(train_X, train_Y, DummyPFN(n_buckets=100))
            posterior = pfn.posterior(test_X)

            self.assertEqual(
                posterior.probabilities.shape, torch.Size([2, 3, 4, 1, 100])
            )
            self.assertEqual(posterior.mean.shape, torch.Size([2, 3, 4, 1, 1]))


class TestPriorFittedNetworkUtils(BotorchTestCase):
    @patch("botorch_community.models.utils.prior_fitted_network.requests.get")
    @patch("botorch_community.models.utils.prior_fitted_network.gzip.GzipFile")
    @patch("botorch_community.models.utils.prior_fitted_network.torch.load")
    @patch("botorch_community.models.utils.prior_fitted_network.torch.save")
    @patch("botorch_community.models.utils.prior_fitted_network.os.path.exists")
    @patch("botorch_community.models.utils.prior_fitted_network.os.makedirs")
    def test_download_model_cache_miss(
        self,
        _mock_makedirs,
        mock_exists,
        mock_torch_save,
        mock_torch_load,
        mock_gzip,
        mock_requests_get,
    ):
        # Simulate cache miss
        mock_exists.return_value = False

        # Mock the requests.get to simulate a network call
        mock_requests_get.return_value = MagicMock(
            status_code=200, content=b"fake content"
        )

        # Mock the gzip.GzipFile to simulate decompression
        mock_gzip.return_value.__enter__.return_value = mock_open(
            read_data=b"fake model data"
        ).return_value

        # Mock torch.load to simulate loading a model
        fake_model = MagicMock(spec=torch.nn.Module)
        mock_torch_load.return_value = fake_model

        # Call the function
        model = download_model(
            ModelPaths.pfns4bo_hebo,
            cache_dir=os.environ.get("RUNNER_TEMP", "/tmp") + "/test_cache",
            # $RUNNER_TEMP is set by GitHub Actions as tmp, /tmp does not work there
        )

        # Assertions for cache miss
        mock_requests_get.assert_called_once()
        mock_gzip.assert_called_once()
        mock_torch_load.assert_called_once()
        mock_torch_save.assert_called_once()
        self.assertEqual(model, fake_model)

    @patch("botorch_community.models.utils.prior_fitted_network.torch.load")
    @patch("botorch_community.models.utils.prior_fitted_network.os.path.exists")
    def test_download_model_cache_hit(self, mock_exists, mock_torch_load):
        # Simulate cache hit
        mock_exists.return_value = True

        # Mock torch.load to simulate loading a model
        fake_model = MagicMock(spec=torch.nn.Module)
        mock_torch_load.return_value = fake_model

        # Call the function
        model = download_model(
            ModelPaths.pfns4bo_hebo,
            cache_dir=os.environ.get("RUNNER_TEMP", "/tmp") + "/test_cache",
            # $RUNNER_TEMP is set by GitHub Actions as tmp, /tmp does not work there
        )

        # Assertions for cache hit
        # mock_exists is called once here and once through os.makedirs
        # which checks if directory exists
        self.assertEqual(mock_exists.call_count, 2)
        mock_torch_load.assert_called_once()
        self.assertEqual(model, fake_model)
