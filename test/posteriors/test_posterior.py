#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from unittest.mock import Mock

import torch
from botorch.posteriors import GPyTorchPosterior, Posterior, PosteriorList
from botorch.utils.testing import BotorchTestCase
from gpytorch.distributions import MultivariateNormal
from gpytorch.lazy.non_lazy_tensor import lazify


class NotSoAbstractPosterior(Posterior):
    @property
    def device(self):
        pass

    @property
    def dtype(self):
        pass

    @property
    def event_shape(self):
        pass

    def rsample(self, *args):
        pass


class TestPosterior(BotorchTestCase):
    def test_abstract_base_posterior(self):
        with self.assertRaises(TypeError):
            Posterior()

    def test_mean_var_notimplemented_error(self):
        posterior = NotSoAbstractPosterior()
        with self.assertRaisesRegex(NotImplementedError, "NotSoAbstractPosterior"):
            posterior.mean
        with self.assertRaisesRegex(NotImplementedError, "NotSoAbstractPosterior"):
            posterior.variance


class TestPosteriorList(BotorchTestCase):
    def _make_gpytorch_posterior(self, shape, dtype):
        mean = torch.rand(*shape, dtype=dtype, device=self.device)
        variance = 1 + torch.rand(*shape, dtype=dtype, device=self.device)
        covar = torch.diag_embed(variance)
        mvn = MultivariateNormal(mean, lazify(covar))
        return GPyTorchPosterior(mvn=mvn)

    def test_posterior_list(self):
        for dtype in (torch.float, torch.double):
            shape = torch.Size([3])
            p_1 = self._make_gpytorch_posterior(shape, dtype)
            p_2 = self._make_gpytorch_posterior(shape, dtype)
            p = PosteriorList(p_1, p_2)

            self.assertEqual(p.base_sample_shape, shape + torch.Size([2]))
            self.assertEqual(p.event_shape, shape + torch.Size([1, 1]))
            self.assertEqual(p.device.type, self.device.type)
            self.assertEqual(p.dtype, dtype)
            self.assertTrue(
                torch.equal(p.mean, torch.cat([p_1.mean, p_2.mean], dim=-1))
            )
            self.assertTrue(
                torch.equal(p.variance, torch.cat([p_1.variance, p_2.variance], dim=-1))
            )
            # test sampling w/o base samples
            sample_shape = torch.Size([4])
            samples = p.sample(sample_shape=sample_shape)
            self.assertEqual(samples.shape, torch.Size([4, 3, 2]))
            # test sampling w/ base samples
            base_samples = torch.randn(
                sample_shape + p.base_sample_shape, device=self.device, dtype=dtype
            )
            samples = p.sample(sample_shape=sample_shape, base_samples=base_samples)
            bs_1, bs_2 = torch.split(base_samples, 1, dim=-1)
            samples_1 = p_1.sample(sample_shape=sample_shape, base_samples=bs_1)
            samples_2 = p_2.sample(sample_shape=sample_shape, base_samples=bs_2)
            samples_expected = torch.cat([samples_1, samples_2], dim=-1)
            self.assertTrue(torch.equal(samples, samples_expected))

    def test_posterior_list_errors(self):
        shape_1 = torch.Size([3, 2])
        shape_2 = torch.Size([4, 1])
        p_1 = self._make_gpytorch_posterior(shape_1, torch.float)
        p_2 = self._make_gpytorch_posterior(shape_2, torch.double)
        p = PosteriorList(p_1, p_2)

        bs_err_msg = (
            "`PosteriorList` only supported if the constituent posteriors "
            "all have the same `batch_shape`."
        )
        with self.assertRaisesRegex(NotImplementedError, bs_err_msg):
            p.base_sample_shape
        with self.assertRaisesRegex(NotImplementedError, bs_err_msg):
            p.event_shape
        dtype_err_msg = "Multi-dtype posteriors are currently not supported."
        with self.assertRaisesRegex(NotImplementedError, dtype_err_msg):
            p.dtype
        device_err_msg = "Multi-device posteriors are currently not supported."
        p_2.mvn.loc = Mock()
        with self.assertRaisesRegex(NotImplementedError, device_err_msg):
            p.device
