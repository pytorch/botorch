#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
from logging import DEBUG, WARN
from unittest.mock import MagicMock, mock_open, patch

import torch
from botorch.acquisition.objective import ScalarizedPosteriorTransform
from botorch.exceptions.errors import UnsupportedError
from botorch.models.transforms.input import Normalize
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
            train_Yvar = torch.rand(10, 1, **tkwargs)
            test_X = torch.rand(5, 3, **tkwargs)

            with self.assertLogs(logger="botorch", level=DEBUG) as log:
                PFNModel(train_X, train_Y, DummyPFN(), train_Yvar=train_Yvar)
                self.assertIn(
                    "train_Yvar provided but ignored for PFNModel.",
                    log.output[0],
                )

            train_Y_4d = torch.rand(10, 2, 2, 1, **tkwargs)
            with self.assertRaises(UnsupportedError):
                PFNModel(train_X, train_Y_4d, DummyPFN())

            train_Y_2d = torch.rand(10, 2, **tkwargs)
            with self.assertRaises(UnsupportedError):
                PFNModel(train_X, train_Y_2d, DummyPFN())

            with self.assertRaises(UnsupportedError):
                PFNModel(torch.rand(10, 3, 3, 2, **tkwargs), train_Y, DummyPFN())

            with self.assertRaises(UnsupportedError):
                PFNModel(train_X, torch.rand(11, **tkwargs), DummyPFN())

            pfn = PFNModel(train_X, train_Y, DummyPFN())

            with self.assertRaises(RuntimeError):
                pfn.posterior(test_X, output_indices=[0, 1])
            with self.assertLogs(logger="botorch", level=WARN) as log:
                pfn.posterior(test_X, observation_noise=True)
                self.assertIn(
                    "observation_noise is not supported for PFNModel",
                    log.output[0],
                )
            with self.assertRaises(UnsupportedError):
                pfn.posterior(
                    test_X,
                    posterior_transform=ScalarizedPosteriorTransform(
                        weights=torch.ones(1)
                    ),
                )

            # q should be 1
            test_X = torch.rand(5, 4, 2, **tkwargs)
            with self.assertRaises(UnsupportedError):
                pfn.posterior(test_X)

            # X dims should be 1 to 4
            test_X = torch.rand(5, 4, 2, 1, 2, **tkwargs)
            with self.assertRaises(UnsupportedError):
                pfn.posterior(test_X)

    def test_shapes(self):
        tkwargs = {"device": self.device, "dtype": torch.float}

        # no q dimension
        train_X = torch.rand(10, 3, **tkwargs)
        train_Y = torch.rand(10, 1, **tkwargs)
        test_X = torch.rand(5, 3, **tkwargs)

        pfn = PFNModel(train_X, train_Y, DummyPFN(n_buckets=100))

        for batch_first in [True, False]:
            with self.subTest(batch_first=batch_first):
                pfn.batch_first = batch_first
                posterior = pfn.posterior(test_X)

                self.assertEqual(posterior.probabilities.shape, torch.Size([5, 100]))
                self.assertEqual(posterior.mean.shape, torch.Size([5, 1]))

        # q=1
        test_X = torch.rand(5, 1, 3, **tkwargs)
        posterior = pfn.posterior(test_X)

        self.assertEqual(posterior.probabilities.shape, torch.Size([5, 100]))
        self.assertEqual(posterior.mean.shape, torch.Size([5, 1]))

        # no shape basically
        test_X = torch.rand(3, **tkwargs)
        posterior = pfn.posterior(test_X)

        self.assertEqual(posterior.probabilities.shape, torch.Size([100]))
        self.assertEqual(posterior.mean.shape, torch.Size([1]))

    def test_batching(self):
        tkwargs = {"device": self.device, "dtype": torch.float}

        # no q dimension
        train_X = torch.rand(2, 10, 3, **tkwargs)
        train_Y = torch.rand(2, 10, 1, **tkwargs)

        pfn = PFNModel(train_X, train_Y, DummyPFN(n_buckets=100))

        test_X = torch.rand(5, 2, 1, 3, **tkwargs)
        posterior = pfn.posterior(test_X)

        self.assertEqual(posterior.probabilities.shape, torch.Size([5, 2, 100]))

        test_X = torch.rand(2, 1, 3, **tkwargs)
        posterior = pfn.posterior(test_X)

        self.assertEqual(posterior.probabilities.shape, torch.Size([2, 100]))

    def test_input_transform(self):
        model = PFNModel(
            train_X=torch.rand(10, 3),
            train_Y=torch.rand(10, 1),
            input_transform=Normalize(d=3),
            model=DummyPFN(),
        )
        self.assertIsInstance(model.input_transform, Normalize)
        self.assertEqual(model.input_transform.bounds.shape, torch.Size([2, 3]))


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

        # Test loading in model init
        model = PFNModel(
            train_X=torch.rand(10, 3),
            train_Y=torch.rand(10, 1),
        )
        self.assertEqual(model.pfn, fake_model)

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
