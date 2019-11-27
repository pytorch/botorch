#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import warnings

import torch
from botorch.exceptions.errors import UnsupportedError
from botorch.posteriors.gpytorch import GPyTorchPosterior, scalarize_posterior
from botorch.utils.testing import BotorchTestCase
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from gpytorch.lazy.non_lazy_tensor import lazify


def _get_test_posterior(batch_shape, q=1, m=1, **tkwargs):
    mean = torch.rand(*batch_shape, q, m, **tkwargs)
    a = torch.rand(*batch_shape, q * m, q * m, **tkwargs)
    covar = a @ a.transpose(-1, -2)
    diag = torch.diagonal(covar, dim1=-2, dim2=-1)
    diag += torch.rand(*batch_shape, q * m, **tkwargs)  # in-place
    mvn = MultitaskMultivariateNormal(mean, covar)
    return GPyTorchPosterior(mvn)


class TestGPyTorchPosterior(BotorchTestCase):
    def test_GPyTorchPosterior(self):
        for dtype in (torch.float, torch.double):
            mean = torch.rand(3, dtype=dtype, device=self.device)
            variance = 1 + torch.rand(3, dtype=dtype, device=self.device)
            covar = variance.diag()
            mvn = MultivariateNormal(mean, lazify(covar))
            posterior = GPyTorchPosterior(mvn=mvn)
            # basics
            self.assertEqual(posterior.device.type, self.device.type)
            self.assertTrue(posterior.dtype == dtype)
            self.assertEqual(posterior.event_shape, torch.Size([3, 1]))
            self.assertTrue(torch.equal(posterior.mean, mean.unsqueeze(-1)))
            self.assertTrue(torch.equal(posterior.variance, variance.unsqueeze(-1)))
            # rsample
            samples = posterior.rsample()
            self.assertEqual(samples.shape, torch.Size([1, 3, 1]))
            samples = posterior.rsample(sample_shape=torch.Size([4]))
            self.assertEqual(samples.shape, torch.Size([4, 3, 1]))
            samples2 = posterior.rsample(sample_shape=torch.Size([4, 2]))
            self.assertEqual(samples2.shape, torch.Size([4, 2, 3, 1]))
            # rsample w/ base samples
            base_samples = torch.randn(4, 3, 1, device=self.device, dtype=dtype)
            # incompatible shapes
            with self.assertRaises(RuntimeError):
                posterior.rsample(
                    sample_shape=torch.Size([3]), base_samples=base_samples
                )
            samples_b1 = posterior.rsample(
                sample_shape=torch.Size([4]), base_samples=base_samples
            )
            samples_b2 = posterior.rsample(
                sample_shape=torch.Size([4]), base_samples=base_samples
            )
            self.assertTrue(torch.allclose(samples_b1, samples_b2))
            base_samples2 = torch.randn(4, 2, 3, 1, device=self.device, dtype=dtype)
            samples2_b1 = posterior.rsample(
                sample_shape=torch.Size([4, 2]), base_samples=base_samples2
            )
            samples2_b2 = posterior.rsample(
                sample_shape=torch.Size([4, 2]), base_samples=base_samples2
            )
            self.assertTrue(torch.allclose(samples2_b1, samples2_b2))
            # collapse_batch_dims
            b_mean = torch.rand(2, 3, dtype=dtype, device=self.device)
            b_variance = 1 + torch.rand(2, 3, dtype=dtype, device=self.device)
            b_covar = torch.diag_embed(b_variance)
            b_mvn = MultivariateNormal(b_mean, lazify(b_covar))
            b_posterior = GPyTorchPosterior(mvn=b_mvn)
            b_base_samples = torch.randn(4, 1, 3, 1, device=self.device, dtype=dtype)
            b_samples = b_posterior.rsample(
                sample_shape=torch.Size([4]), base_samples=b_base_samples
            )
            self.assertEqual(b_samples.shape, torch.Size([4, 2, 3, 1]))

    def test_GPyTorchPosterior_Multitask(self):
        for dtype in (torch.float, torch.double):
            mean = torch.rand(3, 2, dtype=dtype, device=self.device)
            variance = 1 + torch.rand(3, 2, dtype=dtype, device=self.device)
            covar = variance.view(-1).diag()
            mvn = MultitaskMultivariateNormal(mean, lazify(covar))
            posterior = GPyTorchPosterior(mvn=mvn)
            # basics
            self.assertEqual(posterior.device.type, self.device.type)
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
            base_samples = torch.randn(4, 3, 2, device=self.device, dtype=dtype)
            samples_b1 = posterior.rsample(
                sample_shape=torch.Size([4]), base_samples=base_samples
            )
            samples_b2 = posterior.rsample(
                sample_shape=torch.Size([4]), base_samples=base_samples
            )
            self.assertTrue(torch.allclose(samples_b1, samples_b2))
            base_samples2 = torch.randn(4, 2, 3, 2, device=self.device, dtype=dtype)
            samples2_b1 = posterior.rsample(
                sample_shape=torch.Size([4, 2]), base_samples=base_samples2
            )
            samples2_b2 = posterior.rsample(
                sample_shape=torch.Size([4, 2]), base_samples=base_samples2
            )
            self.assertTrue(torch.allclose(samples2_b1, samples2_b2))
            # collapse_batch_dims
            b_mean = torch.rand(2, 3, 2, dtype=dtype, device=self.device)
            b_variance = 1 + torch.rand(2, 3, 2, dtype=dtype, device=self.device)
            b_covar = torch.diag_embed(b_variance.view(2, 6))
            b_mvn = MultitaskMultivariateNormal(b_mean, lazify(b_covar))
            b_posterior = GPyTorchPosterior(mvn=b_mvn)
            b_base_samples = torch.randn(4, 1, 3, 2, device=self.device, dtype=dtype)
            b_samples = b_posterior.rsample(
                sample_shape=torch.Size([4]), base_samples=b_base_samples
            )
            self.assertEqual(b_samples.shape, torch.Size([4, 2, 3, 2]))

    def test_degenerate_GPyTorchPosterior(self):
        for dtype in (torch.float, torch.double):
            # singular covariance matrix
            degenerate_covar = torch.tensor(
                [[1, 1, 0], [1, 1, 0], [0, 0, 2]], dtype=dtype, device=self.device
            )
            mean = torch.rand(3, dtype=dtype, device=self.device)
            mvn = MultivariateNormal(mean, lazify(degenerate_covar))
            posterior = GPyTorchPosterior(mvn=mvn)
            # basics
            self.assertEqual(posterior.device.type, self.device.type)
            self.assertTrue(posterior.dtype == dtype)
            self.assertEqual(posterior.event_shape, torch.Size([3, 1]))
            self.assertTrue(torch.equal(posterior.mean, mean.unsqueeze(-1)))
            variance_exp = degenerate_covar.diag().unsqueeze(-1)
            self.assertTrue(torch.equal(posterior.variance, variance_exp))

            # rsample
            with warnings.catch_warnings(record=True) as w:
                # we check that the p.d. warning is emitted - this only
                # happens once per posterior, so we need to check only once
                samples = posterior.rsample(sample_shape=torch.Size([4]))
                self.assertEqual(len(w), 1)
                self.assertTrue(issubclass(w[-1].category, RuntimeWarning))
                self.assertTrue("not p.d." in str(w[-1].message))
            self.assertEqual(samples.shape, torch.Size([4, 3, 1]))
            samples2 = posterior.rsample(sample_shape=torch.Size([4, 2]))
            self.assertEqual(samples2.shape, torch.Size([4, 2, 3, 1]))
            # rsample w/ base samples
            base_samples = torch.randn(4, 3, 1, device=self.device, dtype=dtype)
            samples_b1 = posterior.rsample(
                sample_shape=torch.Size([4]), base_samples=base_samples
            )
            samples_b2 = posterior.rsample(
                sample_shape=torch.Size([4]), base_samples=base_samples
            )
            self.assertTrue(torch.allclose(samples_b1, samples_b2))
            base_samples2 = torch.randn(4, 2, 3, 1, device=self.device, dtype=dtype)
            samples2_b1 = posterior.rsample(
                sample_shape=torch.Size([4, 2]), base_samples=base_samples2
            )
            samples2_b2 = posterior.rsample(
                sample_shape=torch.Size([4, 2]), base_samples=base_samples2
            )
            self.assertTrue(torch.allclose(samples2_b1, samples2_b2))
            # collapse_batch_dims
            b_mean = torch.rand(2, 3, dtype=dtype, device=self.device)
            b_degenerate_covar = degenerate_covar.expand(2, *degenerate_covar.shape)
            b_mvn = MultivariateNormal(b_mean, lazify(b_degenerate_covar))
            b_posterior = GPyTorchPosterior(mvn=b_mvn)
            b_base_samples = torch.randn(4, 2, 3, 1, device=self.device, dtype=dtype)
            with warnings.catch_warnings(record=True) as w:
                b_samples = b_posterior.rsample(
                    sample_shape=torch.Size([4]), base_samples=b_base_samples
                )
                self.assertEqual(len(w), 1)
                self.assertTrue(issubclass(w[-1].category, RuntimeWarning))
                self.assertTrue("not p.d." in str(w[-1].message))
            self.assertEqual(b_samples.shape, torch.Size([4, 2, 3, 1]))

    def test_degenerate_GPyTorchPosterior_Multitask(self):
        for dtype in (torch.float, torch.double):
            # singular covariance matrix
            degenerate_covar = torch.tensor(
                [[1, 1, 0], [1, 1, 0], [0, 0, 2]], dtype=dtype, device=self.device
            )
            mean = torch.rand(3, dtype=dtype, device=self.device)
            mvn = MultivariateNormal(mean, lazify(degenerate_covar))
            mvn = MultitaskMultivariateNormal.from_independent_mvns([mvn, mvn])
            posterior = GPyTorchPosterior(mvn=mvn)
            # basics
            self.assertEqual(posterior.device.type, self.device.type)
            self.assertTrue(posterior.dtype == dtype)
            self.assertEqual(posterior.event_shape, torch.Size([3, 2]))
            mean_exp = mean.unsqueeze(-1).repeat(1, 2)
            self.assertTrue(torch.equal(posterior.mean, mean_exp))
            variance_exp = degenerate_covar.diag().unsqueeze(-1).repeat(1, 2)
            self.assertTrue(torch.equal(posterior.variance, variance_exp))
            # rsample
            with warnings.catch_warnings(record=True) as w:
                # we check that the p.d. warning is emitted - this only
                # happens once per posterior, so we need to check only once
                samples = posterior.rsample(sample_shape=torch.Size([4]))
                self.assertEqual(len(w), 1)
                self.assertTrue(issubclass(w[-1].category, RuntimeWarning))
                self.assertTrue("not p.d." in str(w[-1].message))
            self.assertEqual(samples.shape, torch.Size([4, 3, 2]))
            samples2 = posterior.rsample(sample_shape=torch.Size([4, 2]))
            self.assertEqual(samples2.shape, torch.Size([4, 2, 3, 2]))
            # rsample w/ base samples
            base_samples = torch.randn(4, 3, 2, device=self.device, dtype=dtype)
            samples_b1 = posterior.rsample(
                sample_shape=torch.Size([4]), base_samples=base_samples
            )
            samples_b2 = posterior.rsample(
                sample_shape=torch.Size([4]), base_samples=base_samples
            )
            self.assertTrue(torch.allclose(samples_b1, samples_b2))
            base_samples2 = torch.randn(4, 2, 3, 2, device=self.device, dtype=dtype)
            samples2_b1 = posterior.rsample(
                sample_shape=torch.Size([4, 2]), base_samples=base_samples2
            )
            samples2_b2 = posterior.rsample(
                sample_shape=torch.Size([4, 2]), base_samples=base_samples2
            )
            self.assertTrue(torch.allclose(samples2_b1, samples2_b2))
            # collapse_batch_dims
            b_mean = torch.rand(2, 3, dtype=dtype, device=self.device)
            b_degenerate_covar = degenerate_covar.expand(2, *degenerate_covar.shape)
            b_mvn = MultivariateNormal(b_mean, lazify(b_degenerate_covar))
            b_mvn = MultitaskMultivariateNormal.from_independent_mvns([b_mvn, b_mvn])
            b_posterior = GPyTorchPosterior(mvn=b_mvn)
            b_base_samples = torch.randn(4, 1, 3, 2, device=self.device, dtype=dtype)
            with warnings.catch_warnings(record=True) as w:
                b_samples = b_posterior.rsample(
                    sample_shape=torch.Size([4]), base_samples=b_base_samples
                )
                self.assertEqual(len(w), 1)
                self.assertTrue(issubclass(w[-1].category, RuntimeWarning))
                self.assertTrue("not p.d." in str(w[-1].message))
            self.assertEqual(b_samples.shape, torch.Size([4, 2, 3, 2]))

    def test_scalarize_posterior(self):
        for batch_shape, m, dtype in itertools.product(
            ([], [3]), (1, 2), (torch.float, torch.double)
        ):
            tkwargs = {"device": self.device, "dtype": dtype}
            offset = torch.rand(1).item()
            weights = torch.randn(m, **tkwargs)
            posterior = _get_test_posterior(batch_shape, m=m, **tkwargs)
            mean, covar = posterior.mvn.mean, posterior.mvn.covariance_matrix
            new_posterior = scalarize_posterior(posterior, weights, offset)
            exp_size = torch.Size(batch_shape + [1, 1])
            self.assertEqual(new_posterior.mean.shape, exp_size)
            new_mean_exp = offset + mean @ weights
            self.assertTrue(torch.allclose(new_posterior.mean[..., -1], new_mean_exp))
            self.assertEqual(new_posterior.variance.shape, exp_size)
            new_covar_exp = ((covar @ weights) @ weights).unsqueeze(-1)
            self.assertTrue(
                torch.allclose(new_posterior.variance[..., -1], new_covar_exp)
            )
            # test errors
            with self.assertRaises(RuntimeError):
                scalarize_posterior(posterior, weights[:-1], offset)
            posterior2 = _get_test_posterior(batch_shape, q=2, m=m, **tkwargs)
            with self.assertRaises(UnsupportedError):
                scalarize_posterior(posterior2, weights, offset)
