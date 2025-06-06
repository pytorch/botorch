#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from itertools import product

import torch
from botorch.models.transforms.utils import (
    expand_and_copy_tensor,
    lognorm_to_norm,
    norm_to_lognorm,
    norm_to_lognorm_mean,
    norm_to_lognorm_variance,
)
from botorch.utils.testing import BotorchTestCase


class TestTransformUtils(BotorchTestCase):
    def test_lognorm_to_norm(self):
        for dtype in (torch.float, torch.double):
            # independent case
            mu = torch.tensor([0.25, 0.5, 1.0], device=self.device, dtype=dtype)
            diag = torch.tensor([0.5, 2.0, 1.0], device=self.device, dtype=dtype)
            Cov = torch.diag_embed((math.exp(1) - 1) * diag)
            mu_n, Cov_n = lognorm_to_norm(mu, Cov)
            mu_n_expected = torch.tensor(
                [-2.73179, -2.03864, -0.5], device=self.device, dtype=dtype
            )
            diag_expected = torch.tensor(
                [2.69099, 2.69099, 1.0], device=self.device, dtype=dtype
            )
            self.assertAllClose(mu_n, mu_n_expected)
            self.assertAllClose(Cov_n, torch.diag_embed(diag_expected))

            # correlated case
            Z = torch.zeros(3, 3, device=self.device, dtype=dtype)
            Z[0, 2] = math.sqrt(math.exp(1)) - 1
            Z[2, 0] = math.sqrt(math.exp(1)) - 1
            mu = torch.ones(3, device=self.device, dtype=dtype)
            Cov = torch.diag_embed(mu * (math.exp(1) - 1)) + Z
            mu_n, Cov_n = lognorm_to_norm(mu, Cov)
            mu_n_expected = -0.5 * torch.ones(3, device=self.device, dtype=dtype)
            Cov_n_expected = torch.tensor(
                [[1.0, 0.0, 0.5], [0.0, 1.0, 0.0], [0.5, 0.0, 1.0]],
                device=self.device,
                dtype=dtype,
            )
            self.assertAllClose(mu_n, mu_n_expected, atol=1e-4)
            self.assertAllClose(Cov_n, Cov_n_expected, atol=1e-4)

    def test_norm_to_lognorm(self):
        for dtype in (torch.float, torch.double):
            # Test joint, independent
            expmu = torch.tensor([1.0, 2.0, 3.0], device=self.device, dtype=dtype)
            expdiag = torch.tensor([1.5, 2.0, 3], device=self.device, dtype=dtype)
            mu = torch.log(expmu)
            diag = torch.log(expdiag)
            Cov = torch.diag_embed(diag)
            mu_ln, Cov_ln = norm_to_lognorm(mu, Cov)
            mu_ln_expected = expmu * torch.exp(0.5 * diag)
            diag_ln_expected = torch.tensor(
                [0.75, 8.0, 54.0], device=self.device, dtype=dtype
            )
            Cov_ln_expected = torch.diag_embed(diag_ln_expected)
            self.assertAllClose(Cov_ln, Cov_ln_expected)
            self.assertAllClose(mu_ln, mu_ln_expected)

            # Test joint, correlated
            Cov[0, 2] = 0.1
            Cov[2, 0] = 0.1
            mu_ln, Cov_ln = norm_to_lognorm(mu, Cov)
            Cov_ln_expected[0, 2] = 0.669304
            Cov_ln_expected[2, 0] = 0.669304
            self.assertAllClose(Cov_ln, Cov_ln_expected)
            self.assertAllClose(mu_ln, mu_ln_expected)

            # Test marginal
            mu = torch.tensor([-1.0, 0.0, 1.0], device=self.device, dtype=dtype)
            v = torch.tensor([1.0, 2.0, 3.0], device=self.device, dtype=dtype)
            var = 2 * (torch.log(v) - mu)
            mu_ln = norm_to_lognorm_mean(mu, var)
            var_ln = norm_to_lognorm_variance(mu, var)
            mu_ln_expected = torch.tensor(
                [1.0, 2.0, 3.0], device=self.device, dtype=dtype
            )
            var_ln_expected = torch.special.expm1(var) * mu_ln_expected**2
            self.assertAllClose(mu_ln, mu_ln_expected)
            self.assertAllClose(var_ln, var_ln_expected)

    def test_round_trip(self):
        for dtype, batch_shape in product((torch.float, torch.double), ([], [2])):
            with self.subTest(dtype=dtype, batch_shape=batch_shape):
                mu = 5 + torch.rand(*batch_shape, 4, device=self.device, dtype=dtype)
                a = 0.2 * torch.randn(
                    *batch_shape, 4, 4, device=self.device, dtype=dtype
                )
                diag = 3.0 + 2 * torch.rand(
                    *batch_shape, 4, device=self.device, dtype=dtype
                )
                Cov = a @ a.transpose(-1, -2) + torch.diag_embed(diag)
                mu_n, Cov_n = lognorm_to_norm(mu, Cov)
                mu_rt, Cov_rt = norm_to_lognorm(mu_n, Cov_n)
                self.assertAllClose(mu_rt, mu, atol=1e-4)
                self.assertAllClose(Cov_rt, Cov, atol=1e-4)

    def test_expand_and_copy_tensor(self):
        for input_batch_shape, batch_shape in product(
            (torch.Size([4, 1]), torch.Size([2, 3, 1])),
            (torch.Size([5]), torch.Size([])),
        ):
            with self.subTest(
                input_batch_shape=input_batch_shape, batch_shape=batch_shape
            ):
                if len(batch_shape) == 0:
                    input_batch_shape = input_batch_shape[:-1]
                X = torch.rand(input_batch_shape + torch.Size([2, 1]))
                expand_shape = (
                    torch.broadcast_shapes(input_batch_shape, batch_shape)
                    + X.shape[-2:]
                )
                X_tf = expand_and_copy_tensor(X, batch_shape=batch_shape)
                self.assertEqual(X_tf.shape, expand_shape)
                self.assertFalse(X_tf is X.expand(expand_shape))
        with self.assertRaisesRegex(RuntimeError, "are not broadcastable"):
            expand_and_copy_tensor(X=torch.rand(2, 2, 3), batch_shape=torch.Size([3]))
