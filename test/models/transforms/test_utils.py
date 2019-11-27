#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
from botorch.models.transforms.utils import (
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
            self.assertTrue(torch.allclose(mu_n, mu_n_expected))
            self.assertTrue(torch.allclose(Cov_n, torch.diag_embed(diag_expected)))

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
            self.assertTrue(torch.allclose(mu_n, mu_n_expected, atol=1e-4))
            self.assertTrue(torch.allclose(Cov_n, Cov_n_expected, atol=1e-4))

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
            self.assertTrue(torch.allclose(Cov_ln, Cov_ln_expected))
            self.assertTrue(torch.allclose(mu_ln, mu_ln_expected))

            # Test joint, correlated
            Cov[0, 2] = 0.1
            Cov[2, 0] = 0.1
            mu_ln, Cov_ln = norm_to_lognorm(mu, Cov)
            Cov_ln_expected[0, 2] = 0.669304
            Cov_ln_expected[2, 0] = 0.669304
            self.assertTrue(torch.allclose(Cov_ln, Cov_ln_expected))
            self.assertTrue(torch.allclose(mu_ln, mu_ln_expected))

            # Test marginal
            mu = torch.tensor([-1.0, 0.0, 1.0], device=self.device, dtype=dtype)
            v = torch.tensor([1.0, 2.0, 3.0], device=self.device, dtype=dtype)
            var = 2 * (torch.log(v) - mu)
            mu_ln = norm_to_lognorm_mean(mu, var)
            var_ln = norm_to_lognorm_variance(mu, var)
            mu_ln_expected = torch.tensor(
                [1.0, 2.0, 3.0], device=self.device, dtype=dtype
            )
            var_ln_expected = (torch.exp(var) - 1) * mu_ln_expected ** 2
            self.assertTrue(torch.allclose(mu_ln, mu_ln_expected))
            self.assertTrue(torch.allclose(var_ln, var_ln_expected))

    def test_round_trip(self):
        for dtype in (torch.float, torch.double):
            for batch_shape in ([], [2]):
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
                self.assertTrue(torch.allclose(mu_rt, mu, atol=1e-4))
                self.assertTrue(torch.allclose(Cov_rt, Cov, atol=1e-4))
