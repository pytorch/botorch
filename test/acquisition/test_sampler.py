#! /usr/bin/env python3

import unittest

import torch
from botorch.acquisition.sampler import (
    IIDNormalSampler,
    MCSampler,
    SobolQMCNormalSampler,
)
from botorch.posteriors import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal


def _get_posterior(cuda=False, dtype=torch.float):
    device = torch.device("cuda") if cuda else torch.device("cpu")
    mean = torch.zeros(2, device=device, dtype=dtype)
    cov = torch.eye(2, device=device, dtype=dtype)
    mvn = MultivariateNormal(mean, cov)
    return GPyTorchPosterior(mvn)


def _get_posterior_batched(cuda=False, dtype=torch.float):
    device = torch.device("cuda") if cuda else torch.device("cpu")
    mean = torch.zeros(3, 2, device=device, dtype=dtype)
    cov = torch.eye(2, device=device, dtype=dtype).repeat(3, 1, 1)
    mvn = MultivariateNormal(mean, cov)
    return GPyTorchPosterior(mvn)


class TestBaseMCSampler(unittest.TestCase):
    def test_MCSampler_abstract_raises(self):
        with self.assertRaises(TypeError):
            MCSampler()


class TestIIDNormalSampler(unittest.TestCase):
    def test_get_base_sample_shape(self):
        sampler = IIDNormalSampler(num_samples=4)
        self.assertEqual(sampler.sample_shape, torch.Size([4]))
        self.assertTrue(sampler.collapse_batch_dims)
        self.assertIsNone(sampler.seed)
        # check sample shape non-batched
        posterior = _get_posterior()
        bss = sampler._get_base_sample_shape(posterior=posterior)
        self.assertEqual(bss, torch.Size([4, 2, 1]))
        # check sample shape batched
        posterior = _get_posterior_batched()
        bss = sampler._get_base_sample_shape(posterior=posterior)
        self.assertEqual(bss, torch.Size([4, 1, 2, 1]))

    def test_get_base_sample_shape_no_collapse(self):
        sampler = IIDNormalSampler(num_samples=4, collapse_batch_dims=False)
        self.assertEqual(sampler.sample_shape, torch.Size([4]))
        self.assertFalse(sampler.collapse_batch_dims)
        self.assertIsNone(sampler.seed)
        # check sample shape non-batched
        posterior = _get_posterior()
        bss = sampler._get_base_sample_shape(posterior=posterior)
        self.assertEqual(bss, torch.Size([4, 2, 1]))
        # check sample shape batched
        posterior = _get_posterior_batched()
        bss = sampler._get_base_sample_shape(posterior=posterior)
        self.assertEqual(bss, torch.Size([4, 3, 2, 1]))

    def test_forward(self, cuda=False):
        for dtype in (torch.float, torch.double):

            # no fixed seed
            sampler = IIDNormalSampler(num_samples=4)
            # check samples non-batched
            posterior = _get_posterior(cuda=cuda, dtype=dtype)
            samples = sampler(posterior)
            self.assertEqual(samples.shape, torch.Size([4, 2, 1]))
            # ensure samples are different
            samples2 = sampler(posterior)
            self.assertFalse(torch.allclose(samples, samples2))
            # ensure this works with a differently shaped posterior
            posterior_batched = _get_posterior_batched(cuda=cuda, dtype=dtype)
            samples_batched = sampler(posterior_batched)
            self.assertEqual(samples_batched.shape, torch.Size([4, 3, 2, 1]))

            # fixed seed
            sampler = IIDNormalSampler(num_samples=4, seed=1234)
            # check samples non-batched
            posterior = _get_posterior(cuda=cuda, dtype=dtype)
            samples = sampler(posterior=posterior)
            self.assertEqual(samples.shape, torch.Size([4, 2, 1]))
            # ensure samples are the same
            samples2 = sampler(posterior)
            self.assertTrue(torch.allclose(samples, samples2))
            # ensure this works with a differently shaped posterior
            posterior_batched = _get_posterior_batched(cuda=cuda, dtype=dtype)
            samples_batched = sampler(posterior_batched)
            self.assertEqual(samples_batched.shape, torch.Size([4, 3, 2, 1]))

    def test_forward_cuda(self):
        if torch.cuda.is_available():
            self.test_forward(cuda=True)

    def test_forward_no_collapse(self, cuda=False):
        for dtype in (torch.float, torch.double):

            # no fixed seed
            sampler = IIDNormalSampler(num_samples=4, collapse_batch_dims=False)
            # check samples non-batched
            posterior = _get_posterior(cuda=cuda, dtype=dtype)
            samples = sampler(posterior)
            self.assertEqual(samples.shape, torch.Size([4, 2, 1]))
            # ensure samples are different
            samples2 = sampler(posterior)
            self.assertFalse(torch.allclose(samples, samples2))
            # ensure this works with a differently shaped posterior
            posterior_batched = _get_posterior_batched(cuda=cuda, dtype=dtype)
            samples_batched = sampler(posterior_batched)
            self.assertEqual(samples_batched.shape, torch.Size([4, 3, 2, 1]))

            # fixed seed
            sampler = IIDNormalSampler(
                num_samples=4, seed=1234, collapse_batch_dims=False
            )
            # check samples non-batched
            posterior = _get_posterior(cuda=cuda, dtype=dtype)
            samples = sampler(posterior=posterior)
            self.assertEqual(samples.shape, torch.Size([4, 2, 1]))
            # ensure samples are the same
            samples2 = sampler(posterior)
            self.assertTrue(torch.allclose(samples, samples2))
            # ensure this works with a differently shaped posterior
            posterior_batched = _get_posterior_batched(cuda=cuda, dtype=dtype)
            samples_batched = sampler(posterior_batched)
            self.assertEqual(samples_batched.shape, torch.Size([4, 3, 2, 1]))

    def test_forward_no_collapse_cuda(self):
        if torch.cuda.is_available():
            self.test_forward_no_collapse(cuda=True)


class TestSobolQMCNormalSampler(unittest.TestCase):
    def test_get_base_sample_shape(self):
        sampler = SobolQMCNormalSampler(num_samples=4)
        self.assertEqual(sampler.sample_shape, torch.Size([4]))
        self.assertTrue(sampler.collapse_batch_dims)
        self.assertIsNone(sampler.seed)
        # check sample shape non-batched
        posterior = _get_posterior()
        bss = sampler._get_base_sample_shape(posterior=posterior)
        self.assertEqual(bss, torch.Size([4, 2, 1]))
        # check sample shape batched
        posterior = _get_posterior_batched()
        bss = sampler._get_base_sample_shape(posterior=posterior)
        self.assertEqual(bss, torch.Size([4, 1, 2, 1]))

    def test_get_base_sample_shape_no_collapse(self):
        sampler = SobolQMCNormalSampler(num_samples=4, collapse_batch_dims=False)
        self.assertEqual(sampler.sample_shape, torch.Size([4]))
        self.assertFalse(sampler.collapse_batch_dims)
        self.assertIsNone(sampler.seed)
        # check sample shape non-batched
        posterior = _get_posterior()
        bss = sampler._get_base_sample_shape(posterior=posterior)
        self.assertEqual(bss, torch.Size([4, 2, 1]))
        # check sample shape batched
        posterior = _get_posterior_batched()
        bss = sampler._get_base_sample_shape(posterior=posterior)
        self.assertEqual(bss, torch.Size([4, 3, 2, 1]))

    def test_forward(self, cuda=False):
        for dtype in (torch.float, torch.double):

            # no fixed seed
            sampler = SobolQMCNormalSampler(num_samples=4)
            # check samples non-batched
            posterior = _get_posterior(cuda=cuda, dtype=dtype)
            samples = sampler(posterior)
            self.assertEqual(samples.shape, torch.Size([4, 2, 1]))
            # ensure samples are different
            samples2 = sampler(posterior)
            self.assertFalse(torch.allclose(samples, samples2))
            # ensure this works with a differently shaped posterior
            posterior_batched = _get_posterior_batched(cuda=cuda, dtype=dtype)
            samples_batched = sampler(posterior_batched)
            self.assertEqual(samples_batched.shape, torch.Size([4, 3, 2, 1]))

            # fixed seed
            sampler = SobolQMCNormalSampler(num_samples=4, seed=1234)
            # check samples non-batched
            posterior = _get_posterior(cuda=cuda, dtype=dtype)
            samples = sampler(posterior=posterior)
            self.assertEqual(samples.shape, torch.Size([4, 2, 1]))
            # ensure samples are the same
            samples2 = sampler(posterior)
            self.assertTrue(torch.allclose(samples, samples2))
            # ensure this works with a differently shaped posterior
            posterior_batched = _get_posterior_batched(cuda=cuda, dtype=dtype)
            samples_batched = sampler(posterior_batched)
            self.assertEqual(samples_batched.shape, torch.Size([4, 3, 2, 1]))

    def test_forward_cuda(self):
        if torch.cuda.is_available():
            self.test_forward(cuda=True)

    def test_forward_no_collapse(self, cuda=False):
        for dtype in (torch.float, torch.double):

            # no fixed seed
            sampler = SobolQMCNormalSampler(num_samples=4, collapse_batch_dims=False)
            # check samples non-batched
            posterior = _get_posterior(cuda=cuda, dtype=dtype)
            samples = sampler(posterior)
            self.assertEqual(samples.shape, torch.Size([4, 2, 1]))
            # ensure samples are different
            samples2 = sampler(posterior)
            self.assertFalse(torch.allclose(samples, samples2))
            # ensure this works with a differently shaped posterior
            posterior_batched = _get_posterior_batched(cuda=cuda, dtype=dtype)
            samples_batched = sampler(posterior_batched)
            self.assertEqual(samples_batched.shape, torch.Size([4, 3, 2, 1]))

            # fixed seed
            sampler = SobolQMCNormalSampler(
                num_samples=4, seed=1234, collapse_batch_dims=False
            )
            # check samples non-batched
            posterior = _get_posterior(cuda=cuda, dtype=dtype)
            samples = sampler(posterior=posterior)
            self.assertEqual(samples.shape, torch.Size([4, 2, 1]))
            # ensure samples are the same
            samples2 = sampler(posterior)
            self.assertTrue(torch.allclose(samples, samples2))
            # ensure this works with a differently shaped posterior
            posterior_batched = _get_posterior_batched(cuda=cuda, dtype=dtype)
            samples_batched = sampler(posterior_batched)
            self.assertEqual(samples_batched.shape, torch.Size([4, 3, 2, 1]))

    def test_forward_no_collapse_cuda(self):
        if torch.cuda.is_available():
            self.test_forward_no_collapse(cuda=True)
