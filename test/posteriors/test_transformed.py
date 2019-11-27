#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.posteriors.transformed import TransformedPosterior
from botorch.utils.testing import BotorchTestCase
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from gpytorch.lazy.non_lazy_tensor import lazify


class TestTransformedPosterior(BotorchTestCase):
    def test_transformed_posterior(self):
        for dtype in (torch.float, torch.double):
            for m in (1, 2):
                shape = torch.Size([3, m])
                mean = torch.rand(shape, dtype=dtype, device=self.device)
                variance = 1 + torch.rand(shape, dtype=dtype, device=self.device)
                if m == 1:
                    covar = torch.diag_embed(variance.squeeze(-1))
                    mvn = MultivariateNormal(mean.squeeze(-1), lazify(covar))
                else:
                    covar = torch.diag_embed(variance.view(*variance.shape[:-2], -1))
                    mvn = MultitaskMultivariateNormal(mean, lazify(covar))
                p_base = GPyTorchPosterior(mvn=mvn)
                p_tf = TransformedPosterior(  # dummy transforms
                    posterior=p_base,
                    sample_transform=lambda s: s + 2,
                    mean_transform=lambda m, v: 2 * m + v,
                    variance_transform=lambda m, v: m + 2 * v,
                )
                # mean, variance
                self.assertEqual(p_tf.device.type, self.device.type)
                self.assertTrue(p_tf.dtype == dtype)
                self.assertEqual(p_tf.event_shape, shape)
                self.assertTrue(torch.equal(p_tf.mean, 2 * mean + variance))
                self.assertTrue(torch.equal(p_tf.variance, mean + 2 * variance))
                # rsample
                samples = p_tf.rsample()
                self.assertEqual(samples.shape, torch.Size([1]) + shape)
                samples = p_tf.rsample(sample_shape=torch.Size([4]))
                self.assertEqual(samples.shape, torch.Size([4]) + shape)
                samples2 = p_tf.rsample(sample_shape=torch.Size([4, 2]))
                self.assertEqual(samples2.shape, torch.Size([4, 2]) + shape)
                # rsample w/ base samples
                base_samples = torch.randn(4, *shape, device=self.device, dtype=dtype)
                # incompatible shapes
                with self.assertRaises(RuntimeError):
                    p_tf.rsample(
                        sample_shape=torch.Size([3]), base_samples=base_samples
                    )
                # make sure sample transform is applied correctly
                samples_base = p_base.rsample(
                    sample_shape=torch.Size([4]), base_samples=base_samples
                )
                samples_tf = p_tf.rsample(
                    sample_shape=torch.Size([4]), base_samples=base_samples
                )
                self.assertTrue(torch.equal(samples_tf, samples_base + 2))
                # check error handling
                p_tf_2 = TransformedPosterior(
                    posterior=p_base, sample_transform=lambda s: s + 2
                )
                with self.assertRaises(NotImplementedError):
                    p_tf_2.mean
                with self.assertRaises(NotImplementedError):
                    p_tf_2.variance
