#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import tempfile
import unittest

import torch
from botorch.posteriors.torch import TorchPosterior
from botorch.utils.testing import BotorchTestCase
from torch.distributions.exponential import Exponential


class TestTorchPosterior(BotorchTestCase):
    def test_torch_posterior(self):
        for dtype in (torch.float, torch.double):
            tkwargs = {"dtype": dtype, "device": self.device}
            rate = torch.rand(1, 2, **tkwargs)
            posterior = TorchPosterior(Exponential(rate=rate))
            self.assertEqual(posterior.dtype, dtype)
            self.assertEqual(posterior.device.type, self.device.type)
            self.assertEqual(
                posterior.rsample(torch.Size([5])).shape, torch.Size([5, 1, 2])
            )
            self.assertTrue(torch.equal(posterior.rate, rate))
            # test sampling with no size provided
            samples_with_unspecified_size = posterior.rsample()
            self.assertEqual(
                samples_with_unspecified_size.shape,
                posterior._batch_shape + posterior._event_shape,
            )
            # Single quantile & density.
            q_value = torch.tensor([0.5], **tkwargs)
            self.assertTrue(
                torch.allclose(
                    posterior.quantile(q_value), posterior.distribution.icdf(q_value)
                )
            )
            self.assertTrue(
                torch.allclose(
                    posterior.density(q_value),
                    posterior.distribution.log_prob(q_value).exp(),
                )
            )
            self.assertEqual(
                posterior._extended_shape(), posterior.distribution._extended_shape()
            )
            # Batch quantile & density.
            q_value = torch.tensor([0.25, 0.5], **tkwargs)
            expected = torch.stack(
                [posterior.distribution.icdf(q) for q in q_value], dim=0
            )
            self.assertAllClose(posterior.quantile(q_value), expected)
            expected = torch.stack(
                [posterior.distribution.log_prob(q).exp() for q in q_value], dim=0
            )
            self.assertAllClose(posterior.density(q_value), expected)

    @unittest.skipIf(os.name == "nt", "Pickle test is not supported on Windows.")
    def test_pickle(self) -> None:
        for dtype in (torch.float, torch.double):
            tkwargs = {"dtype": dtype, "device": self.device}
            posterior = TorchPosterior(Exponential(rate=torch.rand(1, 2, **tkwargs)))
            with tempfile.NamedTemporaryFile() as tmp_file:
                torch.save(posterior, tmp_file.name)
                loaded_posterior = torch.load(tmp_file.name, weights_only=False)
            self.assertEqual(posterior.dtype, loaded_posterior.dtype)
            self.assertEqual(posterior.device, loaded_posterior.device)
            self.assertTrue(
                torch.equal(
                    posterior.distribution.rate,
                    loaded_posterior.distribution.rate,
                )
            )
