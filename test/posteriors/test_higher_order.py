#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from botorch.exceptions.errors import BotorchTensorDimensionError
from botorch.models.higher_order_gp import HigherOrderGP
from botorch.posteriors.higher_order import HigherOrderGPPosterior
from botorch.posteriors.transformed import TransformedPosterior
from botorch.sampling.normal import IIDNormalSampler
from botorch.utils.testing import BotorchTestCase


class TestHigherOrderGPPosterior(BotorchTestCase):
    def setUp(self):
        super().setUp()
        torch.random.manual_seed(0)

        train_x = torch.rand(2, 10, 1, device=self.device)
        train_y = torch.randn(2, 10, 3, 5, device=self.device)

        m1 = HigherOrderGP(train_x, train_y)
        m2 = HigherOrderGP(train_x[0], train_y[0], outcome_transform=None)

        torch.random.manual_seed(0)
        test_x = torch.rand(2, 5, 1, device=self.device)

        posterior1 = m1.posterior(test_x)
        posterior2 = m2.posterior(test_x[0])
        posterior3 = m2.posterior(test_x)

        self.post_list = [
            [m1, test_x, posterior1, TransformedPosterior],
            [m2, test_x[0], posterior2, HigherOrderGPPosterior],
            [m2, test_x, posterior3, HigherOrderGPPosterior],
        ]

    def test_HigherOrderGPPosterior(self):
        sample_shaping = torch.Size([5, 3, 5])

        for post_collection in self.post_list:
            model, test_x, posterior, posterior_class = post_collection

            self.assertIsInstance(posterior, posterior_class)

            batch_shape = test_x.shape[:-2]
            expected_extended_shape = batch_shape + sample_shaping

            self.assertEqual(posterior._extended_shape(), expected_extended_shape)

            # test providing no base samples
            samples_0 = posterior.rsample()
            self.assertEqual(samples_0.shape, torch.Size((1, *expected_extended_shape)))

            # test that providing all base samples produces non-torch.random results
            if len(batch_shape) > 0:
                base_sample_shape = (8, 2, (5 + 10 + 10) * 3 * 5)
            else:
                base_sample_shape = (8, (5 + 10 + 10) * 3 * 5)
            base_samples = torch.randn(*base_sample_shape, device=self.device)

            samples_1 = posterior.rsample_from_base_samples(
                base_samples=base_samples, sample_shape=torch.Size((8,))
            )
            samples_2 = posterior.rsample_from_base_samples(
                base_samples=base_samples, sample_shape=torch.Size((8,))
            )
            self.assertAllClose(samples_1, samples_2)

            # test that botorch.sampler picks up the correct shapes
            sampler = IIDNormalSampler(sample_shape=torch.Size([5]))
            samples_det_shape = sampler(posterior).shape
            self.assertEqual(
                samples_det_shape, torch.Size([5, *expected_extended_shape])
            )

            # test that providing only some base samples is okay
            base_samples = torch.randn(
                8, np.prod(expected_extended_shape), device=self.device
            )
            samples_3 = posterior.rsample_from_base_samples(
                base_samples=base_samples, sample_shape=torch.Size((8,))
            )
            self.assertEqual(samples_3.shape, torch.Size([8, *expected_extended_shape]))

            # test that providing the wrong number base samples errors out
            base_samples = torch.randn(8, 50 * 2 * 3 * 5, device=self.device)
            with self.assertRaises(BotorchTensorDimensionError):
                posterior.rsample_from_base_samples(
                    base_samples=base_samples, sample_shape=torch.Size((8,))
                )

            # test that providing the wrong shapes of base samples fails
            base_samples = torch.randn(8, 5 * 2 * 3 * 5, device=self.device)
            with self.assertRaises(RuntimeError):
                posterior.rsample_from_base_samples(
                    base_samples=base_samples, sample_shape=torch.Size((4,))
                )

            # finally we check the quality of the variances and the samples
            # test that the posterior variances are the same as the evaluation variance
            posterior_variance = posterior.variance

            model.eval()
            eval_mode_variance = model(test_x).variance.reshape_as(posterior_variance)
            if hasattr(model, "outcome_transform"):
                eval_mode_variance = model.outcome_transform.untransform(
                    eval_mode_variance, eval_mode_variance
                )[1]
            self.assertLess(
                (posterior_variance - eval_mode_variance).norm()
                / eval_mode_variance.norm(),
                4e-2,
            )

            # and finally test that sampling with no base samples is okay
            samples_3 = posterior.rsample(sample_shape=torch.Size((5000,)))
            sampled_variance = samples_3.var(dim=0).view(-1)
            posterior_variance = posterior_variance.view(-1)
            self.assertLess(
                (posterior_variance - sampled_variance).norm()
                / posterior_variance.norm(),
                5e-2,
            )
