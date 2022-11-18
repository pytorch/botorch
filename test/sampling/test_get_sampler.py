#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from botorch.posteriors.deterministic import DeterministicPosterior
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.posteriors.posterior_list import PosteriorList
from botorch.posteriors.torch import TorchPosterior
from botorch.posteriors.transformed import TransformedPosterior
from botorch.sampling.get_sampler import get_sampler
from botorch.sampling.list_sampler import ListSampler
from botorch.sampling.normal import IIDNormalSampler, SobolQMCNormalSampler
from botorch.sampling.stochastic_samplers import StochasticSampler
from botorch.utils.testing import BotorchTestCase
from gpytorch.distributions import MultivariateNormal
from torch.distributions.gamma import Gamma


class TestGetSampler(BotorchTestCase):
    def test_get_sampler(self):
        # Basic usage w/ gpytorch posterior.
        posterior = GPyTorchPosterior(
            distribution=MultivariateNormal(torch.rand(2), torch.eye(2))
        )
        sampler = get_sampler(
            posterior=posterior, sample_shape=torch.Size([10]), seed=2
        )
        self.assertIsInstance(sampler, SobolQMCNormalSampler)
        self.assertEqual(sampler.seed, 2)
        self.assertEqual(sampler.sample_shape, torch.Size([10]))

        # Fallback to IID sampler.
        posterior = GPyTorchPosterior(
            distribution=MultivariateNormal(torch.rand(22000), torch.eye(22000))
        )
        sampler = get_sampler(posterior=posterior, sample_shape=torch.Size([10]))
        self.assertIsInstance(sampler, IIDNormalSampler)
        self.assertEqual(sampler.sample_shape, torch.Size([10]))

        # Transformed posterior.
        tf_post = TransformedPosterior(
            posterior=posterior, sample_transform=lambda X: X
        )
        sampler = get_sampler(posterior=tf_post, sample_shape=torch.Size([10]))
        self.assertIsInstance(sampler, IIDNormalSampler)
        self.assertEqual(sampler.sample_shape, torch.Size([10]))

        # PosteriorList with transformed & deterministic.
        post_list = PosteriorList(
            tf_post, DeterministicPosterior(values=torch.rand(1, 2))
        )
        sampler = get_sampler(posterior=post_list, sample_shape=torch.Size([5]))
        self.assertIsInstance(sampler, ListSampler)
        self.assertIsInstance(sampler.samplers[0], IIDNormalSampler)
        self.assertIsInstance(sampler.samplers[1], StochasticSampler)
        for s in sampler.samplers:
            self.assertEqual(s.sample_shape, torch.Size([5]))

        # Unknown torch posterior.
        posterior = TorchPosterior(distribution=Gamma(torch.rand(2), torch.rand(2)))
        with self.assertRaisesRegex(NotImplementedError, "A registered `MCSampler`"):
            get_sampler(posterior=posterior, sample_shape=torch.Size([5]))
