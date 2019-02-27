#!/usr/bin/env python3

import unittest

import torch
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from gpytorch.lazy.non_lazy_tensor import lazify


class TestGPyTorchPosterior(unittest.TestCase):
    def test_GPyTorchPosterior(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            mean = torch.rand(3, dtype=dtype, device=device)
            variance = 1 + torch.rand(3, dtype=dtype, device=device)
            covar = variance.diag()
            mvn = MultivariateNormal(mean, lazify(covar))
            posterior = GPyTorchPosterior(mvn=mvn)
            # basics
            self.assertEqual(posterior.device.type, device.type)
            self.assertTrue(posterior.dtype == dtype)
            self.assertEqual(posterior.event_shape, torch.Size([3, 1]))
            self.assertTrue(torch.equal(posterior.mean, mean.unsqueeze(-1)))
            self.assertTrue(torch.equal(posterior.variance, variance.unsqueeze(-1)))
            # rsample
            samples = posterior.rsample(sample_shape=torch.Size([4]))
            self.assertEqual(samples.shape, torch.Size([4, 3, 1]))
            samples2 = posterior.rsample(sample_shape=torch.Size([4, 2]))
            self.assertEqual(samples2.shape, torch.Size([4, 2, 3, 1]))
            # rsample w/ base samples
            base_samples = posterior.get_base_samples(torch.Size([4]))
            self.assertEqual(base_samples.shape, torch.Size([4, 3, 1]))
            samples_b1 = posterior.rsample(base_samples=base_samples)
            samples_b2 = posterior.rsample(base_samples=base_samples)
            self.assertTrue(torch.allclose(samples_b1, samples_b2))
            base_samples2 = posterior.get_base_samples(torch.Size([4, 2]))
            self.assertEqual(base_samples2.shape, torch.Size([4, 2, 3, 1]))
            samples2_b1 = posterior.rsample(base_samples=base_samples2)
            samples2_b2 = posterior.rsample(base_samples=base_samples2)
            self.assertTrue(torch.allclose(samples2_b1, samples2_b2))

    def test_GPyTorchPosterior_cuda(self):
        if torch.cuda.is_available():
            self.test_GPyTorchPosterior(cuda=True)

    def test_GPyTorchPosterior_Multitask(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            mean = torch.rand(3, 2, dtype=dtype, device=device)
            variance = 1 + torch.rand(3, 2, dtype=dtype, device=device)
            covar = variance.view(-1).diag()
            mvn = MultitaskMultivariateNormal(mean, lazify(covar))
            posterior = GPyTorchPosterior(mvn=mvn)
            # basics
            self.assertEqual(posterior.device.type, device.type)
            self.assertTrue(posterior.dtype == dtype)
            self.assertEqual(posterior.event_shape, torch.Size([3, 2]))
            self.assertTrue(torch.equal(posterior.mean, mean))
            self.assertTrue(torch.equal(posterior.variance, variance))
            # rsample
            samples = posterior.rsample(sample_shape=torch.Size([4]))
            self.assertEqual(samples.shape, torch.Size([4, 3, 2]))
            samples2 = posterior.rsample(sample_shape=torch.Size([4, 2]))
            self.assertEqual(samples2.shape, torch.Size([4, 2, 3, 2]))
            # rsample w/ base samples
            base_samples = posterior.get_base_samples(torch.Size([4]))
            self.assertEqual(base_samples.shape, torch.Size([4, 3, 2]))
            samples_b1 = posterior.rsample(base_samples=base_samples)
            samples_b2 = posterior.rsample(base_samples=base_samples)
            self.assertTrue(torch.allclose(samples_b1, samples_b2))
            base_samples2 = posterior.get_base_samples(torch.Size([4, 2]))
            self.assertEqual(base_samples2.shape, torch.Size([4, 2, 3, 2]))
            samples2_b1 = posterior.rsample(base_samples=base_samples2)
            samples2_b2 = posterior.rsample(base_samples=base_samples2)
            self.assertTrue(torch.allclose(samples2_b1, samples2_b2))

    def test_GPyTorchPosterior_Multitask_cuda(self):
        if torch.cuda.is_available():
            self.test_GPyTorchPosterior_Multitask(cuda=True)

    def test_degenerate_GPyTorchPosterior(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            # singular covariance matrix
            degenerate_covar = torch.tensor(
                [[1, 1, 0], [1, 1, 0], [0, 0, 2]], dtype=dtype, device=device
            )
            mean = torch.rand(3, dtype=dtype, device=device)
            mvn = MultivariateNormal(mean, lazify(degenerate_covar))
            posterior = GPyTorchPosterior(mvn=mvn)
            # basics
            self.assertEqual(posterior.device.type, device.type)
            self.assertTrue(posterior.dtype == dtype)
            self.assertEqual(posterior.event_shape, torch.Size([3, 1]))
            self.assertTrue(torch.equal(posterior.mean, mean.unsqueeze(-1)))
            variance_exp = degenerate_covar.diag().unsqueeze(-1)
            self.assertTrue(torch.equal(posterior.variance, variance_exp))
            # rsample
            samples = posterior.rsample(sample_shape=torch.Size([4]))
            self.assertEqual(samples.shape, torch.Size([4, 3, 1]))
            samples2 = posterior.rsample(sample_shape=torch.Size([4, 2]))
            self.assertEqual(samples2.shape, torch.Size([4, 2, 3, 1]))
            # rsample w/ base samples
            base_samples = posterior.get_base_samples(torch.Size([4]))
            self.assertEqual(base_samples.shape, torch.Size([4, 3, 1]))
            samples_b1 = posterior.rsample(base_samples=base_samples)
            samples_b2 = posterior.rsample(base_samples=base_samples)
            self.assertTrue(torch.allclose(samples_b1, samples_b2))
            base_samples2 = posterior.get_base_samples(torch.Size([4, 2]))
            self.assertEqual(base_samples2.shape, torch.Size([4, 2, 3, 1]))
            samples2_b1 = posterior.rsample(base_samples=base_samples2)
            samples2_b2 = posterior.rsample(base_samples=base_samples2)
            torch.allclose(samples2_b1, samples2_b2)

    def test_degenerate_GPyTorchPosterior_cuda(self):
        if torch.cuda.is_available():
            self.test_degenerate_GPyTorchPosterior(cuda=True)

    def test_degenerate_GPyTorchPosterior_Multitask(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            # singular covariance matrix
            degenerate_covar = torch.tensor(
                [[1, 1, 0], [1, 1, 0], [0, 0, 2]], dtype=dtype, device=device
            )
            mean = torch.rand(3, dtype=dtype, device=device)
            mvn = MultivariateNormal(mean, lazify(degenerate_covar))
            mvn = MultitaskMultivariateNormal.from_independent_mvns([mvn, mvn])
            posterior = GPyTorchPosterior(mvn=mvn)
            # basics
            self.assertEqual(posterior.device.type, device.type)
            self.assertTrue(posterior.dtype == dtype)
            self.assertEqual(posterior.event_shape, torch.Size([3, 2]))
            mean_exp = mean.unsqueeze(-1).repeat(1, 2)
            self.assertTrue(torch.equal(posterior.mean, mean_exp))
            variance_exp = degenerate_covar.diag().unsqueeze(-1).repeat(1, 2)
            self.assertTrue(torch.equal(posterior.variance, variance_exp))
            # rsample
            samples = posterior.rsample(sample_shape=torch.Size([4]))
            self.assertEqual(samples.shape, torch.Size([4, 3, 2]))
            samples2 = posterior.rsample(sample_shape=torch.Size([4, 2]))
            self.assertEqual(samples2.shape, torch.Size([4, 2, 3, 2]))
            # rsample w/ base samples
            base_samples = posterior.get_base_samples(torch.Size([4]))
            self.assertEqual(base_samples.shape, torch.Size([4, 3, 2]))
            samples_b1 = posterior.rsample(base_samples=base_samples)
            samples_b2 = posterior.rsample(base_samples=base_samples)
            self.assertTrue(torch.allclose(samples_b1, samples_b2))
            base_samples2 = posterior.get_base_samples(torch.Size([4, 2]))
            self.assertEqual(base_samples2.shape, torch.Size([4, 2, 3, 2]))
            samples2_b1 = posterior.rsample(base_samples=base_samples2)
            samples2_b2 = posterior.rsample(base_samples=base_samples2)
            torch.allclose(samples2_b1, samples2_b2)

    def test_degenerate_GPyTorchPosterior_Multitask_cuda(self):
        if torch.cuda.is_available():
            self.test_degenerate_GPyTorchPosterior_Multitask(cuda=True)
