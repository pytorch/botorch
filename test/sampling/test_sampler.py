#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from botorch.exceptions.errors import UnsupportedError
from botorch.posteriors import GPyTorchPosterior
from botorch.sampling.samplers import IIDNormalSampler, MCSampler, SobolQMCNormalSampler
from botorch.utils.testing import BotorchTestCase
from gpytorch.distributions import MultivariateNormal
from gpytorch.lazy import DiagLazyTensor


def _get_test_posterior(device, dtype=torch.float):
    mean = torch.zeros(2, device=device, dtype=dtype)
    cov = torch.eye(2, device=device, dtype=dtype)
    mvn = MultivariateNormal(mean, cov)
    return GPyTorchPosterior(mvn)


def _get_test_posterior_batched(device, dtype=torch.float):
    mean = torch.zeros(3, 2, device=device, dtype=dtype)
    cov = torch.eye(2, device=device, dtype=dtype).repeat(3, 1, 1)
    mvn = MultivariateNormal(mean, cov)
    return GPyTorchPosterior(mvn)


class TestBaseMCSampler(BotorchTestCase):
    def test_MCSampler_abstract_raises(self):
        with self.assertRaises(TypeError):
            MCSampler()


class TestIIDNormalSampler(BotorchTestCase):
    def test_batch_range(self):
        # check batch_range default and can be changed
        sampler = IIDNormalSampler(num_samples=4)
        self.assertEqual(sampler.batch_range, (0, -2))
        sampler.batch_range = (-3, -2)
        self.assertEqual(sampler.batch_range, (-3, -2))
        # check that base samples are cleared after batch_range set
        posterior = _get_test_posterior(self.device)
        _ = sampler(posterior)
        self.assertNotEqual(sampler.base_samples, None)
        sampler.batch_range = (0, -2)
        self.assertEqual(sampler.base_samples, None)

    def test_get_base_sample_shape(self):
        sampler = IIDNormalSampler(num_samples=4)
        self.assertFalse(sampler.resample)
        self.assertEqual(sampler.sample_shape, torch.Size([4]))
        self.assertTrue(sampler.collapse_batch_dims)
        # check sample shape non-batched
        posterior = _get_test_posterior(self.device)
        bss = sampler._get_base_sample_shape(posterior=posterior)
        self.assertEqual(bss, torch.Size([4, 2, 1]))
        # check sample shape batched
        posterior = _get_test_posterior_batched(self.device)
        bss = sampler._get_base_sample_shape(posterior=posterior)
        self.assertEqual(bss, torch.Size([4, 1, 2, 1]))
        # check sample shape with different batch range
        sampler.batch_range = (-3, -1)
        posterior = _get_test_posterior_batched(self.device)
        bss = sampler._get_base_sample_shape(posterior=posterior)
        self.assertEqual(bss, torch.Size([4, 1, 1, 1]))

    def test_get_base_sample_shape_no_collapse(self):
        sampler = IIDNormalSampler(num_samples=4, collapse_batch_dims=False)
        self.assertFalse(sampler.resample)
        self.assertEqual(sampler.sample_shape, torch.Size([4]))
        self.assertFalse(sampler.collapse_batch_dims)
        # check sample shape non-batched
        posterior = _get_test_posterior(self.device)
        bss = sampler._get_base_sample_shape(posterior=posterior)
        self.assertEqual(bss, torch.Size([4, 2, 1]))
        # check sample shape batched
        posterior = _get_test_posterior_batched(self.device)
        bss = sampler._get_base_sample_shape(posterior=posterior)
        self.assertEqual(bss, torch.Size([4, 3, 2, 1]))
        # check sample shape with different batch range
        sampler.batch_range = (-3, -1)
        posterior = _get_test_posterior_batched(self.device)
        bss = sampler._get_base_sample_shape(posterior=posterior)
        self.assertEqual(bss, torch.Size([4, 3, 2, 1]))

    def test_forward(self):
        for dtype in (torch.float, torch.double):

            # no resample
            sampler = IIDNormalSampler(num_samples=4, seed=1234)
            self.assertFalse(sampler.resample)
            self.assertEqual(sampler.seed, 1234)
            self.assertTrue(sampler.collapse_batch_dims)
            # check samples non-batched
            posterior = _get_test_posterior(device=self.device, dtype=dtype)
            samples = sampler(posterior)
            self.assertEqual(samples.shape, torch.Size([4, 2, 1]))
            self.assertEqual(sampler.seed, 1235)
            # ensure samples are the same
            samples2 = sampler(posterior)
            self.assertTrue(torch.allclose(samples, samples2))
            self.assertEqual(sampler.seed, 1235)
            # ensure this works with a differently shaped posterior
            posterior_batched = _get_test_posterior_batched(
                device=self.device, dtype=dtype
            )
            samples_batched = sampler(posterior_batched)
            self.assertEqual(samples_batched.shape, torch.Size([4, 3, 2, 1]))
            self.assertEqual(sampler.seed, 1235)
            # ensure this works when changing the dtype
            new_dtype = torch.float if dtype == torch.double else torch.double
            posterior_batched = _get_test_posterior_batched(
                device=self.device, dtype=new_dtype
            )
            samples_batched = sampler(posterior_batched)
            self.assertEqual(samples_batched.shape, torch.Size([4, 3, 2, 1]))
            self.assertEqual(sampler.seed, 1235)
            # ensure this works with a different batch_range
            # should trigger a resample, so seed goes up
            sampler.batch_range = (-3, -1)
            posterior_batched = _get_test_posterior_batched(
                device=self.device, dtype=dtype
            )
            samples_batched = sampler(posterior_batched)
            self.assertEqual(samples_batched.shape, torch.Size([4, 3, 2, 1]))
            self.assertEqual(sampler.seed, 1236)

            # resample
            sampler = IIDNormalSampler(num_samples=4, resample=True, seed=None)
            self.assertTrue(sampler.resample)
            self.assertTrue(sampler.collapse_batch_dims)
            initial_seed = sampler.seed
            # check samples non-batched
            posterior = _get_test_posterior(device=self.device, dtype=dtype)
            samples = sampler(posterior)
            self.assertEqual(samples.shape, torch.Size([4, 2, 1]))
            self.assertEqual(sampler.seed, initial_seed + 1)
            # ensure samples are different
            samples2 = sampler(posterior)
            self.assertFalse(torch.allclose(samples, samples2))
            self.assertEqual(sampler.seed, initial_seed + 2)
            # ensure this works with a differently shaped posterior
            posterior_batched = _get_test_posterior_batched(
                device=self.device, dtype=dtype
            )
            samples_batched = sampler(posterior_batched)
            self.assertEqual(samples_batched.shape, torch.Size([4, 3, 2, 1]))
            self.assertEqual(sampler.seed, initial_seed + 3)
            # ensure this works with a different batch_range
            sampler.batch_range = (-3, -1)
            posterior_batched = _get_test_posterior_batched(
                device=self.device, dtype=dtype
            )
            samples_batched = sampler(posterior_batched)
            self.assertEqual(samples_batched.shape, torch.Size([4, 3, 2, 1]))
            self.assertEqual(sampler.seed, initial_seed + 4)

    def test_forward_no_collapse(self):
        for dtype in (torch.float, torch.double):

            # no resample
            sampler = IIDNormalSampler(
                num_samples=4, seed=1234, collapse_batch_dims=False
            )
            self.assertFalse(sampler.resample)
            self.assertEqual(sampler.seed, 1234)
            self.assertFalse(sampler.collapse_batch_dims)
            # check samples non-batched
            posterior = _get_test_posterior(device=self.device, dtype=dtype)
            samples = sampler(posterior)
            self.assertEqual(samples.shape, torch.Size([4, 2, 1]))
            self.assertEqual(sampler.seed, 1235)
            # ensure samples are the same
            samples2 = sampler(posterior)
            self.assertTrue(torch.allclose(samples, samples2))
            self.assertEqual(sampler.seed, 1235)
            # ensure this works with a differently shaped posterior
            posterior_batched = _get_test_posterior_batched(
                device=self.device, dtype=dtype
            )
            samples_batched = sampler(posterior_batched)
            self.assertEqual(samples_batched.shape, torch.Size([4, 3, 2, 1]))
            self.assertEqual(sampler.seed, 1236)
            # ensure this works with a different batch_range
            # should trigger a resample, so seed goes up
            sampler.batch_range = (-3, -1)
            posterior_batched = _get_test_posterior_batched(
                device=self.device, dtype=dtype
            )
            samples_batched = sampler(posterior_batched)
            self.assertEqual(samples_batched.shape, torch.Size([4, 3, 2, 1]))
            self.assertEqual(sampler.seed, 1237)

            # resample
            sampler = IIDNormalSampler(
                num_samples=4, resample=True, collapse_batch_dims=False
            )
            self.assertTrue(sampler.resample)
            self.assertFalse(sampler.collapse_batch_dims)
            initial_seed = sampler.seed
            # check samples non-batched
            posterior = _get_test_posterior(device=self.device, dtype=dtype)
            samples = sampler(posterior=posterior)
            self.assertEqual(samples.shape, torch.Size([4, 2, 1]))
            self.assertEqual(sampler.seed, initial_seed + 1)
            # ensure samples are not the same
            samples2 = sampler(posterior)
            self.assertFalse(torch.allclose(samples, samples2))
            self.assertEqual(sampler.seed, initial_seed + 2)
            # ensure this works with a differently shaped posterior
            posterior_batched = _get_test_posterior_batched(
                device=self.device, dtype=dtype
            )
            samples_batched = sampler(posterior_batched)
            self.assertEqual(samples_batched.shape, torch.Size([4, 3, 2, 1]))
            self.assertEqual(sampler.seed, initial_seed + 3)
            # ensure this works with a different batch_range
            sampler.batch_range = (-3, -1)
            posterior_batched = _get_test_posterior_batched(
                device=self.device, dtype=dtype
            )
            samples_batched = sampler(posterior_batched)
            self.assertEqual(samples_batched.shape, torch.Size([4, 3, 2, 1]))
            self.assertEqual(sampler.seed, initial_seed + 4)


class TestSobolQMCNormalSampler(BotorchTestCase):
    def test_batch_range(self):
        # check batch_range default and can be changed
        sampler = SobolQMCNormalSampler(num_samples=4)
        self.assertEqual(sampler.batch_range, (0, -2))
        sampler.batch_range = (-3, -2)
        self.assertEqual(sampler.batch_range, (-3, -2))
        # check that base samples are cleared after batch_range set
        posterior = _get_test_posterior(self.device)
        _ = sampler(posterior)
        self.assertNotEqual(sampler.base_samples, None)
        sampler.batch_range = (0, -2)
        self.assertEqual(sampler.base_samples, None)

    def test_get_base_sample_shape(self):
        sampler = SobolQMCNormalSampler(num_samples=4)
        self.assertFalse(sampler.resample)
        self.assertEqual(sampler.sample_shape, torch.Size([4]))
        self.assertTrue(sampler.collapse_batch_dims)
        # check sample shape non-batched
        posterior = _get_test_posterior(self.device)
        bss = sampler._get_base_sample_shape(posterior=posterior)
        self.assertEqual(bss, torch.Size([4, 2, 1]))
        # check sample shape batched
        posterior = _get_test_posterior_batched(self.device)
        bss = sampler._get_base_sample_shape(posterior=posterior)
        self.assertEqual(bss, torch.Size([4, 1, 2, 1]))
        # check sample shape with different batch range
        sampler.batch_range = (-3, -1)
        posterior = _get_test_posterior_batched(self.device)
        bss = sampler._get_base_sample_shape(posterior=posterior)
        self.assertEqual(bss, torch.Size([4, 1, 1, 1]))

    def test_get_base_sample_shape_no_collapse(self):
        sampler = SobolQMCNormalSampler(num_samples=4, collapse_batch_dims=False)
        self.assertFalse(sampler.resample)
        self.assertEqual(sampler.sample_shape, torch.Size([4]))
        self.assertFalse(sampler.collapse_batch_dims)
        # check sample shape non-batched
        posterior = _get_test_posterior(self.device)
        bss = sampler._get_base_sample_shape(posterior=posterior)
        self.assertEqual(bss, torch.Size([4, 2, 1]))
        # check sample shape batched
        posterior = _get_test_posterior_batched(self.device)
        bss = sampler._get_base_sample_shape(posterior=posterior)
        self.assertEqual(bss, torch.Size([4, 3, 2, 1]))
        # check sample shape with different batch range
        sampler.batch_range = (-3, -1)
        posterior = _get_test_posterior_batched(self.device)
        bss = sampler._get_base_sample_shape(posterior=posterior)
        self.assertEqual(bss, torch.Size([4, 3, 2, 1]))

    def test_forward(self):
        for dtype in (torch.float, torch.double):

            # no resample
            sampler = SobolQMCNormalSampler(num_samples=4, seed=1234)
            self.assertFalse(sampler.resample)
            self.assertEqual(sampler.seed, 1234)
            self.assertTrue(sampler.collapse_batch_dims)
            # check samples non-batched
            posterior = _get_test_posterior(device=self.device, dtype=dtype)
            samples = sampler(posterior)
            self.assertEqual(samples.shape, torch.Size([4, 2, 1]))
            self.assertEqual(sampler.seed, 1235)
            # ensure samples are the same
            samples2 = sampler(posterior)
            self.assertTrue(torch.allclose(samples, samples2))
            self.assertEqual(sampler.seed, 1235)
            # ensure this works with a differently shaped posterior
            posterior_batched = _get_test_posterior_batched(
                device=self.device, dtype=dtype
            )
            samples_batched = sampler(posterior_batched)
            self.assertEqual(samples_batched.shape, torch.Size([4, 3, 2, 1]))
            self.assertEqual(sampler.seed, 1235)
            # ensure this works when changing the dtype
            new_dtype = torch.float if dtype == torch.double else torch.double
            posterior_batched = _get_test_posterior_batched(
                device=self.device, dtype=new_dtype
            )
            samples_batched = sampler(posterior_batched)
            self.assertEqual(samples_batched.shape, torch.Size([4, 3, 2, 1]))
            self.assertEqual(sampler.seed, 1235)
            # ensure this works with a different batch_range
            # should trigger a resample, so seed goes up
            sampler.batch_range = (-3, -1)
            posterior_batched = _get_test_posterior_batched(
                device=self.device, dtype=dtype
            )
            samples_batched = sampler(posterior_batched)
            self.assertEqual(samples_batched.shape, torch.Size([4, 3, 2, 1]))
            self.assertEqual(sampler.seed, 1236)

            # resample
            sampler = SobolQMCNormalSampler(num_samples=4, resample=True, seed=None)
            self.assertTrue(sampler.resample)
            self.assertTrue(sampler.collapse_batch_dims)
            initial_seed = sampler.seed
            # check samples non-batched
            posterior = _get_test_posterior(device=self.device, dtype=dtype)
            samples = sampler(posterior)
            self.assertEqual(samples.shape, torch.Size([4, 2, 1]))
            self.assertEqual(sampler.seed, initial_seed + 1)
            # ensure samples are different
            samples2 = sampler(posterior)
            self.assertFalse(torch.allclose(samples, samples2))
            self.assertEqual(sampler.seed, initial_seed + 2)
            # ensure this works with a differently shaped posterior
            posterior_batched = _get_test_posterior_batched(
                device=self.device, dtype=dtype
            )
            samples_batched = sampler(posterior_batched)
            self.assertEqual(samples_batched.shape, torch.Size([4, 3, 2, 1]))
            self.assertEqual(sampler.seed, initial_seed + 3)
            # ensure this works with a different batch_range
            sampler.batch_range = (-3, -1)
            posterior_batched = _get_test_posterior_batched(
                device=self.device, dtype=dtype
            )
            samples_batched = sampler(posterior_batched)
            self.assertEqual(samples_batched.shape, torch.Size([4, 3, 2, 1]))
            self.assertEqual(sampler.seed, initial_seed + 4)

    def test_forward_no_collapse(self):
        for dtype in (torch.float, torch.double):

            # no resample
            sampler = SobolQMCNormalSampler(
                num_samples=4, seed=1234, collapse_batch_dims=False
            )
            self.assertFalse(sampler.resample)
            self.assertEqual(sampler.seed, 1234)
            self.assertFalse(sampler.collapse_batch_dims)
            # check samples non-batched
            posterior = _get_test_posterior(device=self.device, dtype=dtype)
            samples = sampler(posterior)
            self.assertEqual(samples.shape, torch.Size([4, 2, 1]))
            self.assertEqual(sampler.seed, 1235)
            # ensure samples are the same
            samples2 = sampler(posterior)
            self.assertTrue(torch.allclose(samples, samples2))
            self.assertEqual(sampler.seed, 1235)
            # ensure this works with a differently shaped posterior
            posterior_batched = _get_test_posterior_batched(
                device=self.device, dtype=dtype
            )
            samples_batched = sampler(posterior_batched)
            self.assertEqual(samples_batched.shape, torch.Size([4, 3, 2, 1]))
            self.assertEqual(sampler.seed, 1236)
            # ensure this works with a different batch_range
            # should trigger a resample, so seed goes up
            sampler.batch_range = (-3, -1)
            posterior_batched = _get_test_posterior_batched(
                device=self.device, dtype=dtype
            )
            samples_batched = sampler(posterior_batched)
            self.assertEqual(samples_batched.shape, torch.Size([4, 3, 2, 1]))
            self.assertEqual(sampler.seed, 1237)

            # resample
            sampler = SobolQMCNormalSampler(
                num_samples=4, resample=True, collapse_batch_dims=False
            )
            self.assertTrue(sampler.resample)
            self.assertFalse(sampler.collapse_batch_dims)
            initial_seed = sampler.seed
            # check samples non-batched
            posterior = _get_test_posterior(device=self.device, dtype=dtype)
            samples = sampler(posterior=posterior)
            self.assertEqual(samples.shape, torch.Size([4, 2, 1]))
            self.assertEqual(sampler.seed, initial_seed + 1)
            # ensure samples are not the same
            samples2 = sampler(posterior)
            self.assertFalse(torch.allclose(samples, samples2))
            self.assertEqual(sampler.seed, initial_seed + 2)
            # ensure this works with a differently shaped posterior
            posterior_batched = _get_test_posterior_batched(
                device=self.device, dtype=dtype
            )
            samples_batched = sampler(posterior_batched)
            self.assertEqual(samples_batched.shape, torch.Size([4, 3, 2, 1]))
            self.assertEqual(sampler.seed, initial_seed + 3)
            # ensure this works with a different batch_range
            sampler.batch_range = (-3, -1)
            posterior_batched = _get_test_posterior_batched(
                device=self.device, dtype=dtype
            )
            samples_batched = sampler(posterior_batched)
            self.assertEqual(samples_batched.shape, torch.Size([4, 3, 2, 1]))
            self.assertEqual(sampler.seed, initial_seed + 4)

    def test_unsupported_dimension(self):
        sampler = SobolQMCNormalSampler(num_samples=2)
        maxdim = torch.quasirandom.SobolEngine.MAXDIM + 1
        mean = torch.zeros(maxdim)
        cov = DiagLazyTensor(torch.ones(maxdim))
        mvn = MultivariateNormal(mean, cov)
        posterior = GPyTorchPosterior(mvn)
        with self.assertRaises(UnsupportedError) as e:
            sampler(posterior)
            self.assertIn(f"Requested: {maxdim}", str(e.exception))
