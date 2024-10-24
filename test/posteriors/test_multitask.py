#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from botorch.exceptions.errors import BotorchTensorDimensionError
from botorch.models.multitask import KroneckerMultiTaskGP
from botorch.posteriors.multitask import _permute_solve, MultitaskGPPosterior
from botorch.sampling.normal import IIDNormalSampler
from botorch.utils.testing import BotorchTestCase
from linear_operator.operators import to_linear_operator


def get_posterior_test_cases(
    device: torch.device, dtype: torch.dtype
) -> list[tuple[KroneckerMultiTaskGP, torch.Tensor, MultitaskGPPosterior]]:
    torch.random.manual_seed(0)

    train_x = torch.rand(10, 1, device=device, dtype=dtype)
    train_y = torch.randn(10, 3, device=device, dtype=dtype)

    m2 = KroneckerMultiTaskGP(train_x, train_y)

    torch.random.manual_seed(0)
    test_x = torch.rand(2, 5, 1, device=device, dtype=dtype)

    posterior0 = m2.posterior(test_x[0])
    posterior1 = m2.posterior(test_x)
    posterior2 = m2.posterior(test_x[0], observation_noise=True)
    posterior3 = m2.posterior(test_x, observation_noise=True)

    post_list = [
        (m2, test_x[0], posterior0),
        (m2, test_x, posterior1),
        (m2, test_x[0], posterior2),
        (m2, test_x, posterior3),
    ]
    return post_list


class TestMultitaskGPPosterior(BotorchTestCase):
    def _test_MultitaskGPPosterior(self, dtype: torch.dtype) -> None:
        post_list = get_posterior_test_cases(device=self.device, dtype=dtype)
        sample_shaping = torch.Size([5, 3])

        for post_collection in post_list:
            model, test_x, posterior = post_collection

            self.assertIsInstance(posterior, MultitaskGPPosterior)

            batch_shape = test_x.shape[:-2]
            expected_extended_shape = batch_shape + sample_shaping

            self.assertEqual(posterior._extended_shape(), expected_extended_shape)

            # test providing no base samples
            samples_0 = posterior.rsample()
            self.assertEqual(samples_0.shape, torch.Size((1, *expected_extended_shape)))

            # test that providing all base samples produces non-torch.random results
            scale = 2 if posterior.observation_noise else 1
            base_sample_shaping = torch.Size(
                [
                    2 * model.train_targets.numel()
                    + scale * sample_shaping[0] * sample_shaping[1]
                ]
            )

            expected_base_sample_shape = batch_shape + base_sample_shaping
            self.assertEqual(
                posterior.base_sample_shape,
                expected_base_sample_shape,
            )
            base_samples = torch.randn(
                8,
                *expected_base_sample_shape,
                device=self.device,
                dtype=dtype,
            )

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
                8,
                np.prod(expected_extended_shape),
                device=self.device,
                dtype=dtype,
            )
            samples_3 = posterior.rsample_from_base_samples(
                base_samples=base_samples, sample_shape=torch.Size((8,))
            )
            self.assertEqual(samples_3.shape, torch.Size([8, *expected_extended_shape]))

            # test that providing the wrong number base samples errors out
            base_samples = torch.randn(8, 50 * 2 * 3, device=self.device)
            with self.assertRaises(BotorchTensorDimensionError):
                posterior.rsample_from_base_samples(
                    base_samples=base_samples, sample_shape=torch.Size((8,))
                )

            # test that providing the wrong shapes of base samples fails
            base_samples = torch.randn(8, 5 * 2 * 3, device=self.device)
            with self.assertRaises(RuntimeError):
                posterior.rsample_from_base_samples(
                    base_samples=base_samples, sample_shape=torch.Size((4,))
                )

            # finally we check the quality of the variances and the samples
            # test that the posterior variances are the same as the evaluation variance
            posterior_variance = posterior.variance
            posterior_mean = posterior.mean.view(-1)

            model.eval()
            if not posterior.observation_noise:
                eval_mode_variance = model(test_x).variance
            else:
                eval_mode_variance = model.likelihood(model(test_x)).variance
            eval_mode_variance = eval_mode_variance.reshape_as(posterior_variance)

            self.assertLess(
                (posterior_variance - eval_mode_variance).norm()
                / eval_mode_variance.norm(),
                4e-2,
            )

            # and finally test that sampling with no base samples is okay
            samples_3 = posterior.rsample(sample_shape=torch.Size((10000,)))
            sampled_variance = samples_3.var(dim=0).view(-1)
            sampled_mean = samples_3.mean(dim=0).view(-1)

            posterior_variance = posterior_variance.view(-1)

            # slightly higher tolerance here because of the potential for low norms
            # Note: This check appears to yield different results with CUDA
            # depending on what machine it is run on, so you may see better
            # convergence.
            self.assertLess(
                (posterior_mean - sampled_mean).norm() / posterior_mean.norm(),
                0.13,
            )
            self.assertLess(
                (posterior_variance - sampled_variance).norm()
                / posterior_variance.norm(),
                5e-2,
            )

    def test_MultitaskGPPosterior(self) -> None:
        for dtype in (torch.float, torch.double):
            with self.subTest(dtype=dtype):
                self._test_MultitaskGPPosterior(dtype=dtype)

    def test_draw_from_base_covar(self):
        # grab a posterior
        posterior = get_posterior_test_cases(device=self.device, dtype=torch.float32)[
            0
        ][2]

        base_samples = torch.randn(4, 30, 1, device=self.device)
        base_mat = torch.randn(30, 30, device=self.device)
        sym_mat = base_mat.matmul(base_mat.t())

        # base, non-lt case
        res = posterior._draw_from_base_covar(sym_mat, base_samples)
        self.assertIsInstance(res, torch.Tensor)

        # too many samples works
        base_samples = torch.randn(4, 50, 1, device=self.device)
        res = posterior._draw_from_base_covar(sym_mat, base_samples)
        self.assertIsInstance(res, torch.Tensor)

        # too few samples fails
        base_samples = torch.randn(4, 10, 1, device=self.device)
        with self.assertRaises(RuntimeError):
            res = posterior._draw_from_base_covar(sym_mat, base_samples)


class TestPermuteSolve(BotorchTestCase):
    def test_permute_solve_tensor(self):
        # Random PSD matrix
        a = torch.randn(32, 32, device=self.device, dtype=torch.float64)
        A = torch.mm(a, a.t())

        # Random batched column vector
        b = torch.randn(4, 1, 32, 1, device=self.device, dtype=torch.float64)

        # Compare results of permuted and standard solve
        x_1 = _permute_solve(to_linear_operator(A), b)
        x_2 = torch.linalg.solve(A, b)
        self.assertAllClose(x_1, x_2)

        # Ensure also works if b is not batched
        x_1 = _permute_solve(to_linear_operator(A), b[0, 0, :, :])
        x_2 = torch.linalg.solve(A, b[0, 0, :, :])
        self.assertAllClose(x_1, x_2)
