#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from botorch.posteriors import GPyTorchPosterior
from botorch.sampling.pairwise_samplers import (
    PairwiseIIDNormalSampler,
    PairwiseMCSampler,
    PairwiseSobolQMCNormalSampler,
)
from botorch.utils.testing import BotorchTestCase
from gpytorch.distributions import MultivariateNormal


def _get_test_posterior(device, n=3, dtype=torch.float, batched=False):
    mean = torch.zeros(n, device=device, dtype=dtype)
    cov = torch.eye(n, device=device, dtype=dtype)
    if batched:
        cov = cov.repeat(3, 1, 1)
    mvn = MultivariateNormal(mean, cov)
    return GPyTorchPosterior(mvn)


class TestBasePairwiseMCSampler(BotorchTestCase):
    def test_PairwiseMCSampler_abstract_raises(self):
        with self.assertRaises(TypeError):
            PairwiseMCSampler()


class TestPairwiseIIDNormalSampler(BotorchTestCase):
    def test_get_base_sample_shape(self):
        sampler = PairwiseIIDNormalSampler(num_samples=4)
        self.assertFalse(sampler.resample)
        self.assertEqual(sampler.sample_shape, torch.Size([4]))
        self.assertTrue(sampler.collapse_batch_dims)
        # check sample shape non-batched
        posterior = _get_test_posterior(self.device)
        bss = sampler._get_base_sample_shape(posterior=posterior)
        self.assertEqual(bss, torch.Size([4, 3, 1]))
        # check sample shape batched
        posterior = _get_test_posterior(self.device, batched=True)
        bss = sampler._get_base_sample_shape(posterior=posterior)
        self.assertEqual(bss, torch.Size([4, 1, 3, 1]))

    def test_get_base_sample_shape_no_collapse(self):
        sampler = PairwiseIIDNormalSampler(num_samples=4, collapse_batch_dims=False)
        self.assertFalse(sampler.resample)
        self.assertEqual(sampler.sample_shape, torch.Size([4]))
        self.assertFalse(sampler.collapse_batch_dims)
        # check sample shape non-batched
        posterior = _get_test_posterior(self.device)
        bss = sampler._get_base_sample_shape(posterior=posterior)
        self.assertEqual(bss, torch.Size([4, 3, 1]))
        # check sample shape batched
        posterior = _get_test_posterior(self.device, batched=True)
        bss = sampler._get_base_sample_shape(posterior=posterior)
        self.assertEqual(bss, torch.Size([4, 3, 3, 1]))

    def test_forward(self):
        for dtype in (torch.float, torch.double):
            # no resample
            sampler = PairwiseIIDNormalSampler(num_samples=4, seed=1234)
            self.assertFalse(sampler.resample)
            self.assertEqual(sampler.seed, 1234)
            self.assertTrue(sampler.collapse_batch_dims)
            # check samples non-batched
            posterior = _get_test_posterior(device=self.device, dtype=dtype)
            samples = sampler(posterior)
            self.assertEqual(samples.shape, torch.Size([4, 3, 2]))
            self.assertEqual(sampler.seed, 1235)
            # ensure samples are the same
            samples2 = sampler(posterior)
            self.assertTrue(torch.allclose(samples, samples2))
            self.assertEqual(sampler.seed, 1235)
            # ensure this works with a differently shaped posterior
            posterior_batched = _get_test_posterior(
                device=self.device, dtype=dtype, batched=True
            )
            samples_batched = sampler(posterior_batched)
            self.assertEqual(samples_batched.shape, torch.Size([4, 3, 3, 2]))
            self.assertEqual(sampler.seed, 1235)
            # ensure this works when changing the dtype
            new_dtype = torch.float if dtype == torch.double else torch.double
            posterior_batched = _get_test_posterior(
                device=self.device, dtype=new_dtype, batched=True
            )
            samples_batched = sampler(posterior_batched)
            self.assertEqual(samples_batched.shape, torch.Size([4, 3, 3, 2]))
            self.assertEqual(sampler.seed, 1235)
            # ensure error is rasied when number of points are < 2
            posterior = _get_test_posterior(device=self.device, n=1, dtype=dtype)
            with self.assertRaises(RuntimeError):
                sampler(posterior)

            # resample
            sampler = PairwiseIIDNormalSampler(num_samples=4, resample=True, seed=None)
            self.assertTrue(sampler.resample)
            self.assertTrue(sampler.collapse_batch_dims)
            initial_seed = sampler.seed
            # check samples non-batched
            posterior = _get_test_posterior(device=self.device, dtype=dtype)
            samples = sampler(posterior)
            self.assertEqual(samples.shape, torch.Size([4, 3, 2]))
            self.assertEqual(sampler.seed, initial_seed + 1)
            # ensure samples are different
            samples2 = sampler(posterior)
            self.assertFalse(torch.allclose(samples, samples2))
            self.assertEqual(sampler.seed, initial_seed + 2)
            # ensure this works with a differently shaped posterior
            posterior_batched = _get_test_posterior(
                device=self.device, dtype=dtype, batched=True
            )
            samples_batched = sampler(posterior_batched)
            self.assertEqual(samples_batched.shape, torch.Size([4, 3, 3, 2]))
            self.assertEqual(sampler.seed, initial_seed + 3)
            # ensure error is rasied when number of points are < 2
            posterior = _get_test_posterior(device=self.device, n=1, dtype=dtype)
            with self.assertRaises(RuntimeError):
                sampler(posterior)

            # check max_num_comparisons
            sampler = PairwiseIIDNormalSampler(num_samples=4, max_num_comparisons=2)
            self.assertFalse(sampler.resample)
            self.assertTrue(sampler.collapse_batch_dims)
            # check samples non-batched
            posterior = _get_test_posterior(device=self.device, dtype=dtype)
            samples = sampler(posterior)
            self.assertEqual(samples.shape, torch.Size([4, 2, 2]))

    def test_forward_no_collapse(self):
        for dtype in (torch.float, torch.double):
            # no resample
            sampler = PairwiseIIDNormalSampler(
                num_samples=4, seed=1234, collapse_batch_dims=False
            )
            self.assertFalse(sampler.resample)
            self.assertEqual(sampler.seed, 1234)
            self.assertFalse(sampler.collapse_batch_dims)
            # check samples non-batched
            posterior = _get_test_posterior(device=self.device, dtype=dtype)
            samples = sampler(posterior)
            self.assertEqual(samples.shape, torch.Size([4, 3, 2]))
            self.assertEqual(sampler.seed, 1235)
            # ensure samples are the same
            samples2 = sampler(posterior)
            self.assertTrue(torch.allclose(samples, samples2))
            self.assertEqual(sampler.seed, 1235)
            # ensure this works with a differently shaped posterior
            posterior_batched = _get_test_posterior(
                device=self.device, dtype=dtype, batched=True
            )
            samples_batched = sampler(posterior_batched)
            self.assertEqual(samples_batched.shape, torch.Size([4, 3, 3, 2]))
            self.assertEqual(sampler.seed, 1236)
            # ensure error is rasied when number of points are < 2
            posterior = _get_test_posterior(device=self.device, n=1, dtype=dtype)
            with self.assertRaises(RuntimeError):
                sampler(posterior)

            # resample
            sampler = PairwiseIIDNormalSampler(
                num_samples=4, resample=True, collapse_batch_dims=False
            )
            self.assertTrue(sampler.resample)
            self.assertFalse(sampler.collapse_batch_dims)
            initial_seed = sampler.seed
            # check samples non-batched
            posterior = _get_test_posterior(device=self.device, dtype=dtype)
            samples = sampler(posterior=posterior)
            self.assertEqual(samples.shape, torch.Size([4, 3, 2]))
            self.assertEqual(sampler.seed, initial_seed + 1)
            # ensure samples are not the same
            samples2 = sampler(posterior)
            self.assertFalse(torch.allclose(samples, samples2))
            self.assertEqual(sampler.seed, initial_seed + 2)
            # ensure this works with a differeantly shaped posterior
            posterior_batched = _get_test_posterior(
                device=self.device, dtype=dtype, batched=True
            )
            samples_batched = sampler(posterior_batched)
            self.assertEqual(samples_batched.shape, torch.Size([4, 3, 3, 2]))
            self.assertEqual(sampler.seed, initial_seed + 3)
            # ensure error is rasied when number of points are < 2
            posterior = _get_test_posterior(device=self.device, n=1, dtype=dtype)
            with self.assertRaises(RuntimeError):
                sampler(posterior)

            # check max_num_comparisons
            sampler = PairwiseIIDNormalSampler(
                num_samples=4, max_num_comparisons=2, collapse_batch_dims=False
            )
            self.assertFalse(sampler.resample)
            self.assertFalse(sampler.collapse_batch_dims)
            # check samples non-batched
            posterior = _get_test_posterior(device=self.device, dtype=dtype)
            samples = sampler(posterior)
            self.assertEqual(samples.shape, torch.Size([4, 2, 2]))


class TestPairwiseSobolQMCNormalSampler(BotorchTestCase):
    def test_get_base_sample_shape(self):
        sampler = PairwiseSobolQMCNormalSampler(num_samples=4)
        self.assertFalse(sampler.resample)
        self.assertEqual(sampler.sample_shape, torch.Size([4]))
        self.assertTrue(sampler.collapse_batch_dims)
        # check sample shape non-batched
        posterior = _get_test_posterior(self.device)
        bss = sampler._get_base_sample_shape(posterior=posterior)
        self.assertEqual(bss, torch.Size([4, 3, 1]))
        # check sample shape batched
        posterior = _get_test_posterior(self.device, batched=True)
        bss = sampler._get_base_sample_shape(posterior=posterior)
        self.assertEqual(bss, torch.Size([4, 1, 3, 1]))

    def test_get_base_sample_shape_no_collapse(self):
        sampler = PairwiseSobolQMCNormalSampler(
            num_samples=4, collapse_batch_dims=False
        )
        self.assertFalse(sampler.resample)
        self.assertEqual(sampler.sample_shape, torch.Size([4]))
        self.assertFalse(sampler.collapse_batch_dims)
        # check sample shape non-batched
        posterior = _get_test_posterior(self.device)
        bss = sampler._get_base_sample_shape(posterior=posterior)
        self.assertEqual(bss, torch.Size([4, 3, 1]))
        # check sample shape batched
        posterior = _get_test_posterior(self.device, batched=True)
        bss = sampler._get_base_sample_shape(posterior=posterior)
        self.assertEqual(bss, torch.Size([4, 3, 3, 1]))

    def test_forward(self):
        for dtype in (torch.float, torch.double):
            # no resample
            sampler = PairwiseSobolQMCNormalSampler(num_samples=4, seed=1234)
            self.assertFalse(sampler.resample)
            self.assertEqual(sampler.seed, 1234)
            self.assertTrue(sampler.collapse_batch_dims)
            # check samples non-batched
            posterior = _get_test_posterior(device=self.device, dtype=dtype)
            samples = sampler(posterior)
            self.assertEqual(samples.shape, torch.Size([4, 3, 2]))
            self.assertEqual(sampler.seed, 1235)
            # ensure samples are the same
            samples2 = sampler(posterior)
            self.assertTrue(torch.allclose(samples, samples2))
            self.assertEqual(sampler.seed, 1235)
            # ensure this works with a differently shaped posterior
            posterior_batched = _get_test_posterior(
                device=self.device, dtype=dtype, batched=True
            )
            samples_batched = sampler(posterior_batched)
            self.assertEqual(samples_batched.shape, torch.Size([4, 3, 3, 2]))
            self.assertEqual(sampler.seed, 1235)
            # ensure this works when changing the dtype
            new_dtype = torch.float if dtype == torch.double else torch.double
            posterior_batched = _get_test_posterior(
                device=self.device, dtype=new_dtype, batched=True
            )
            samples_batched = sampler(posterior_batched)
            self.assertEqual(samples_batched.shape, torch.Size([4, 3, 3, 2]))
            self.assertEqual(sampler.seed, 1235)
            # ensure error is rasied when number of points are < 2
            posterior = _get_test_posterior(device=self.device, n=1, dtype=dtype)
            with self.assertRaises(RuntimeError):
                sampler(posterior)

            # resample
            sampler = PairwiseSobolQMCNormalSampler(
                num_samples=4, resample=True, seed=None
            )
            self.assertTrue(sampler.resample)
            self.assertTrue(sampler.collapse_batch_dims)
            initial_seed = sampler.seed
            # check samples non-batched
            posterior = _get_test_posterior(device=self.device, dtype=dtype)
            samples = sampler(posterior)
            self.assertEqual(samples.shape, torch.Size([4, 3, 2]))
            self.assertEqual(sampler.seed, initial_seed + 1)
            # ensure samples are different
            samples2 = sampler(posterior)
            self.assertFalse(torch.allclose(samples, samples2))
            self.assertEqual(sampler.seed, initial_seed + 2)
            # ensure this works with a differently shaped posterior
            posterior_batched = _get_test_posterior(
                device=self.device, dtype=dtype, batched=True
            )
            samples_batched = sampler(posterior_batched)
            self.assertEqual(samples_batched.shape, torch.Size([4, 3, 3, 2]))
            self.assertEqual(sampler.seed, initial_seed + 3)
            # ensure error is rasied when number of points are < 2
            posterior = _get_test_posterior(device=self.device, n=1, dtype=dtype)
            with self.assertRaises(RuntimeError):
                sampler(posterior)

            # check max_num_comparisons
            sampler = PairwiseSobolQMCNormalSampler(
                num_samples=4, max_num_comparisons=2
            )
            self.assertFalse(sampler.resample)
            self.assertTrue(sampler.collapse_batch_dims)
            # check samples non-batched
            posterior = _get_test_posterior(device=self.device, dtype=dtype)
            samples = sampler(posterior)
            self.assertEqual(samples.shape, torch.Size([4, 2, 2]))

    def test_forward_no_collapse(self):
        for dtype in (torch.float, torch.double):
            # no resample
            sampler = PairwiseSobolQMCNormalSampler(
                num_samples=4, seed=1234, collapse_batch_dims=False
            )
            self.assertFalse(sampler.resample)
            self.assertEqual(sampler.seed, 1234)
            self.assertFalse(sampler.collapse_batch_dims)
            # check samples non-batched
            posterior = _get_test_posterior(device=self.device, dtype=dtype)
            samples = sampler(posterior)
            self.assertEqual(samples.shape, torch.Size([4, 3, 2]))
            self.assertEqual(sampler.seed, 1235)
            # ensure samples are the same
            samples2 = sampler(posterior)
            self.assertTrue(torch.allclose(samples, samples2))
            self.assertEqual(sampler.seed, 1235)
            # ensure this works with a differently shaped posterior
            posterior_batched = _get_test_posterior(
                device=self.device, dtype=dtype, batched=True
            )
            samples_batched = sampler(posterior_batched)
            self.assertEqual(samples_batched.shape, torch.Size([4, 3, 3, 2]))
            self.assertEqual(sampler.seed, 1236)
            # ensure error is rasied when number of points are < 2
            posterior = _get_test_posterior(device=self.device, n=1, dtype=dtype)
            with self.assertRaises(RuntimeError):
                sampler(posterior)

            # resample
            sampler = PairwiseSobolQMCNormalSampler(
                num_samples=4, resample=True, collapse_batch_dims=False
            )
            self.assertTrue(sampler.resample)
            self.assertFalse(sampler.collapse_batch_dims)
            initial_seed = sampler.seed
            # check samples non-batched
            posterior = _get_test_posterior(device=self.device, dtype=dtype)
            samples = sampler(posterior=posterior)
            self.assertEqual(samples.shape, torch.Size([4, 3, 2]))
            self.assertEqual(sampler.seed, initial_seed + 1)
            # ensure samples are not the same
            samples2 = sampler(posterior)
            self.assertFalse(torch.allclose(samples, samples2))
            self.assertEqual(sampler.seed, initial_seed + 2)
            # ensure this works with a differeantly shaped posterior
            posterior_batched = _get_test_posterior(
                device=self.device, dtype=dtype, batched=True
            )
            samples_batched = sampler(posterior_batched)
            self.assertEqual(samples_batched.shape, torch.Size([4, 3, 3, 2]))
            self.assertEqual(sampler.seed, initial_seed + 3)
            # ensure error is rasied when number of points are < 2
            posterior = _get_test_posterior(device=self.device, n=1, dtype=dtype)
            with self.assertRaises(RuntimeError):
                sampler(posterior)

            # check max_num_comparisons
            sampler = PairwiseSobolQMCNormalSampler(
                num_samples=4, max_num_comparisons=2, collapse_batch_dims=False
            )
            self.assertFalse(sampler.resample)
            self.assertFalse(sampler.collapse_batch_dims)
            # check samples non-batched
            posterior = _get_test_posterior(device=self.device, dtype=dtype)
            samples = sampler(posterior)
            self.assertEqual(samples.shape, torch.Size([4, 2, 2]))
