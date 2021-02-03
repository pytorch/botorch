#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import math

import numpy as np
import torch
from botorch.sampling.qmc import MultivariateNormalQMCEngine, NormalQMCEngine
from botorch.utils.testing import BotorchTestCase
from scipy.stats import shapiro


class NormalQMCTests(BotorchTestCase):
    def test_NormalQMCEngine(self):
        for d in (1, 2):
            engine = NormalQMCEngine(d=d)
            samples = engine.draw()
            self.assertEqual(samples.dtype, torch.float)
            self.assertEqual(samples.shape, torch.Size([1, d]))
            samples = engine.draw(n=5)
            self.assertEqual(samples.shape, torch.Size([5, d]))
            # test double dtype
            samples = engine.draw(dtype=torch.double)
            self.assertEqual(samples.dtype, torch.double)
            self.assertEqual(samples.shape, torch.Size([1, d]))

    def test_NormalQMCEngineInvTransform(self):
        for d in (1, 2):
            engine = NormalQMCEngine(d=d, inv_transform=True)
            samples = engine.draw()
            self.assertEqual(samples.dtype, torch.float)
            self.assertEqual(samples.shape, torch.Size([1, d]))
            samples = engine.draw(n=5)
            self.assertEqual(samples.shape, torch.Size([5, d]))
            # test double dtype
            samples = engine.draw(dtype=torch.double)
            self.assertEqual(samples.dtype, torch.double)
            self.assertEqual(samples.shape, torch.Size([1, d]))

    def test_NormalQMCEngineSeeded(self):
        # test even dimension
        engine = NormalQMCEngine(d=2, seed=12345)
        samples = engine.draw(n=2)
        self.assertEqual(samples.dtype, torch.float)
        self.assertEqual(samples.shape, torch.Size([2, 2]))
        # test odd dimension
        engine = NormalQMCEngine(d=3, seed=12345)
        samples = engine.draw(n=2)
        self.assertEqual(samples.shape, torch.Size([2, 3]))

    def test_NormalQMCEngineSeededOut(self):
        # test even dimension
        engine = NormalQMCEngine(d=2, seed=12345)
        out = torch.zeros(2, 2)
        self.assertIsNone(engine.draw(n=2, out=out))
        self.assertTrue(torch.all(out != 0))
        # test odd dimension
        engine = NormalQMCEngine(d=3, seed=12345)
        out = torch.empty(2, 3)
        self.assertIsNone(engine.draw(n=2, out=out))
        self.assertTrue(torch.all(out != 0))

    def test_NormalQMCEngineSeededInvTransform(self):
        # test even dimension
        engine = NormalQMCEngine(d=2, seed=12345, inv_transform=True)
        samples = engine.draw(n=2)
        self.assertEqual(samples.dtype, torch.float)
        self.assertEqual(samples.shape, torch.Size([2, 2]))
        # test odd dimension
        engine = NormalQMCEngine(d=3, seed=12345, inv_transform=True)
        samples = engine.draw(n=2)
        self.assertEqual(samples.shape, torch.Size([2, 3]))

    def test_NormalQMCEngineShapiro(self):
        engine = NormalQMCEngine(d=2, seed=12345)
        samples = engine.draw(n=256)
        self.assertEqual(samples.dtype, torch.float)
        self.assertTrue(torch.all(torch.abs(samples.mean(dim=0)) < 1e-2))
        self.assertTrue(torch.all(torch.abs(samples.std(dim=0) - 1) < 1e-2))
        # perform Shapiro-Wilk test for normality
        for i in (0, 1):
            _, pval = shapiro(samples[:, i])
            self.assertGreater(pval, 0.9)
        # make sure samples are uncorrelated
        cov = np.cov(samples.numpy().transpose())
        self.assertLess(np.abs(cov[0, 1]), 1e-2)

    def test_NormalQMCEngineShapiroInvTransform(self):
        engine = NormalQMCEngine(d=2, seed=12345, inv_transform=True)
        samples = engine.draw(n=256)
        self.assertEqual(samples.dtype, torch.float)
        self.assertTrue(torch.all(torch.abs(samples.mean(dim=0)) < 1e-2))
        self.assertTrue(torch.all(torch.abs(samples.std(dim=0) - 1) < 1e-2))
        # perform Shapiro-Wilk test for normality
        for i in (0, 1):
            _, pval = shapiro(samples[:, i])
            self.assertGreater(pval, 0.9)
        # make sure samples are uncorrelated
        cov = np.cov(samples.numpy().transpose())
        self.assertLess(np.abs(cov[0, 1]), 1e-2)


class MultivariateNormalQMCTests(BotorchTestCase):
    def test_MultivariateNormalQMCEngineShapeErrors(self):
        with self.assertRaises(ValueError):
            MultivariateNormalQMCEngine(mean=torch.zeros(2), cov=torch.zeros(2, 1))
        with self.assertRaises(ValueError):
            MultivariateNormalQMCEngine(mean=torch.zeros(1), cov=torch.eye(2))

    def test_MultivariateNormalQMCEngineNonPSD(self):
        for dtype in (torch.float, torch.double):
            # try with non-psd, non-pd cov and expect an assertion error
            mean = torch.zeros(2, device=self.device, dtype=dtype)
            cov = torch.tensor([[1, 2], [2, 1]], device=self.device, dtype=dtype)
            with self.assertRaises(ValueError):
                MultivariateNormalQMCEngine(mean=mean, cov=cov)

    def test_MultivariateNormalQMCEngineNonPD(self):
        for dtype in (torch.float, torch.double):
            mean = torch.zeros(3, device=self.device, dtype=dtype)
            cov = torch.tensor(
                [[1, 0, 1], [0, 1, 1], [1, 1, 2]], device=self.device, dtype=dtype
            )
            # try with non-pd but psd cov; should work
            engine = MultivariateNormalQMCEngine(mean=mean, cov=cov)
            self.assertTrue(engine._corr_matrix is not None)

    def test_MultivariateNormalQMCEngineSymmetric(self):
        for dtype in (torch.float, torch.double):
            # try with non-symmetric cov and expect an error
            mean = torch.zeros(2, device=self.device, dtype=dtype)
            cov = torch.tensor([[1, 0], [2, 1]], device=self.device, dtype=dtype)
            with self.assertRaises(ValueError):
                MultivariateNormalQMCEngine(mean=mean, cov=cov)

    def test_MultivariateNormalQMCEngine(self):
        for d, dtype in itertools.product((1, 2, 3), (torch.float, torch.double)):
            mean = torch.rand(d, device=self.device, dtype=dtype)
            cov = torch.eye(d, device=self.device, dtype=dtype)
            engine = MultivariateNormalQMCEngine(mean=mean, cov=cov)
            samples = engine.draw()
            self.assertEqual(samples.dtype, dtype)
            self.assertEqual(samples.device.type, self.device.type)
            self.assertEqual(samples.shape, torch.Size([1, d]))
            samples = engine.draw(n=5)
            self.assertEqual(samples.shape, torch.Size([5, d]))

    def test_MultivariateNormalQMCEngineInvTransform(self):
        for d, dtype in itertools.product((1, 2, 3), (torch.float, torch.double)):
            mean = torch.rand(d, device=self.device, dtype=dtype)
            cov = torch.eye(d, device=self.device, dtype=dtype)
            engine = MultivariateNormalQMCEngine(mean=mean, cov=cov, inv_transform=True)
            samples = engine.draw()
            self.assertEqual(samples.dtype, dtype)
            self.assertEqual(samples.device.type, self.device.type)
            self.assertEqual(samples.shape, torch.Size([1, d]))
            samples = engine.draw(n=5)
            self.assertEqual(samples.shape, torch.Size([5, d]))

    def test_MultivariateNormalQMCEngineSeeded(self):
        for dtype in (torch.float, torch.double):

            # test even dimension
            a = torch.randn(2, 2)
            cov = a @ a.transpose(-1, -2) + torch.rand(2).diag()
            mean = torch.zeros(2, device=self.device, dtype=dtype)
            cov = cov.to(device=self.device, dtype=dtype)
            engine = MultivariateNormalQMCEngine(mean=mean, cov=cov, seed=12345)
            samples = engine.draw(n=2)
            self.assertEqual(samples.dtype, dtype)
            self.assertEqual(samples.device.type, self.device.type)

            # test odd dimension
            a = torch.randn(3, 3)
            cov = a @ a.transpose(-1, -2) + torch.rand(3).diag()
            mean = torch.zeros(3, device=self.device, dtype=dtype)
            cov = cov.to(device=self.device, dtype=dtype)
            engine = MultivariateNormalQMCEngine(mean, cov, seed=12345)
            samples = engine.draw(n=2)
            self.assertEqual(samples.dtype, dtype)
            self.assertEqual(samples.device.type, self.device.type)

    def test_MultivariateNormalQMCEngineSeededOut(self):
        for dtype in (torch.float, torch.double):
            # test even dimension
            a = torch.randn(2, 2)
            cov = a @ a.transpose(-1, -2) + torch.rand(2).diag()
            mean = torch.zeros(2, device=self.device, dtype=dtype)
            cov = cov.to(device=self.device, dtype=dtype)
            engine = MultivariateNormalQMCEngine(mean=mean, cov=cov, seed=12345)
            out = torch.zeros(2, 2, device=self.device, dtype=dtype)
            self.assertIsNone(engine.draw(n=2, out=out))
            self.assertTrue(torch.all(out != 0))
            # test odd dimension
            a = torch.randn(3, 3)
            cov = a @ a.transpose(-1, -2) + torch.rand(3).diag()
            mean = torch.zeros(3, device=self.device, dtype=dtype)
            cov = cov.to(device=self.device, dtype=dtype)
            engine = MultivariateNormalQMCEngine(mean, cov, seed=12345)
            out = torch.zeros(2, 3, device=self.device, dtype=dtype)
            self.assertIsNone(engine.draw(n=2, out=out))
            self.assertTrue(torch.all(out != 0))

    def test_MultivariateNormalQMCEngineSeededInvTransform(self):
        for dtype in (torch.float, torch.double):
            # test even dimension
            a = torch.randn(2, 2)
            cov = a @ a.transpose(-1, -2) + torch.rand(2).diag()
            mean = torch.zeros(2, device=self.device, dtype=dtype)
            cov = cov.to(device=self.device, dtype=dtype)
            engine = MultivariateNormalQMCEngine(
                mean=mean, cov=cov, seed=12345, inv_transform=True
            )
            samples = engine.draw(n=2)
            self.assertEqual(samples.dtype, dtype)
            self.assertEqual(samples.device.type, self.device.type)
            # test odd dimension
            a = torch.randn(3, 3)
            cov = a @ a.transpose(-1, -2) + torch.rand(3).diag()
            mean = torch.zeros(3, device=self.device, dtype=dtype)
            cov = cov.to(device=self.device, dtype=dtype)
            engine = MultivariateNormalQMCEngine(
                mean=mean, cov=cov, seed=12345, inv_transform=True
            )
            samples = engine.draw(n=2)
            self.assertEqual(samples.dtype, dtype)
            self.assertEqual(samples.device.type, self.device.type)

    def test_MultivariateNormalQMCEngineShapiro(self):
        for dtype in (torch.float, torch.double):
            # test the standard case
            mean = torch.zeros(2, device=self.device, dtype=dtype)
            cov = torch.eye(2, device=self.device, dtype=dtype)
            engine = MultivariateNormalQMCEngine(mean=mean, cov=cov, seed=12345)
            samples = engine.draw(n=256)
            self.assertEqual(samples.dtype, dtype)
            self.assertEqual(samples.device.type, self.device.type)
            self.assertTrue(torch.all(torch.abs(samples.mean(dim=0)) < 1e-2))
            self.assertTrue(torch.all(torch.abs(samples.std(dim=0) - 1) < 1e-2))
            # perform Shapiro-Wilk test for normality
            samples = samples.cpu().numpy()
            for i in (0, 1):
                _, pval = shapiro(samples[:, i])
                self.assertGreater(pval, 0.9)
            # make sure samples are uncorrelated
            cov = np.cov(samples.transpose())
            self.assertLess(np.abs(cov[0, 1]), 1e-2)

            # test the correlated, non-zero mean case
            mean = torch.tensor([1.0, 2.0], device=self.device, dtype=dtype)
            cov = torch.tensor(
                [[1.5, 0.5], [0.5, 1.5]], device=self.device, dtype=dtype
            )
            engine = MultivariateNormalQMCEngine(mean=mean, cov=cov, seed=12345)
            samples = engine.draw(n=256)
            self.assertEqual(samples.dtype, dtype)
            self.assertEqual(samples.device.type, self.device.type)
            self.assertTrue(torch.all(torch.abs(samples.mean(dim=0) - mean) < 1e-2))
            self.assertTrue(
                torch.all(torch.abs(samples.std(dim=0) - math.sqrt(1.5)) < 1e-2)
            )
            # perform Shapiro-Wilk test for normality
            samples = samples.cpu().numpy()
            for i in (0, 1):
                _, pval = shapiro(samples[:, i])
                self.assertGreater(pval, 0.9)
            # check covariance
            cov = np.cov(samples.transpose())
            self.assertLess(np.abs(cov[0, 1] - 0.5), 1e-2)

    def test_MultivariateNormalQMCEngineShapiroInvTransform(self):
        for dtype in (torch.float, torch.double):
            # test the standard case
            mean = torch.zeros(2, device=self.device, dtype=dtype)
            cov = torch.eye(2, device=self.device, dtype=dtype)
            engine = MultivariateNormalQMCEngine(
                mean=mean, cov=cov, seed=12345, inv_transform=True
            )
            samples = engine.draw(n=256)
            self.assertEqual(samples.dtype, dtype)
            self.assertEqual(samples.device.type, self.device.type)
            self.assertTrue(torch.all(torch.abs(samples.mean(dim=0)) < 1e-2))
            self.assertTrue(torch.all(torch.abs(samples.std(dim=0) - 1) < 1e-2))
            # perform Shapiro-Wilk test for normality
            samples = samples.cpu().numpy()
            for i in (0, 1):
                _, pval = shapiro(samples[:, i])
                self.assertGreater(pval, 0.9)
            # make sure samples are uncorrelated
            cov = np.cov(samples.transpose())
            self.assertLess(np.abs(cov[0, 1]), 1e-2)

            # test the correlated, non-zero mean case
            mean = torch.tensor([1.0, 2.0], device=self.device, dtype=dtype)
            cov = torch.tensor(
                [[1.5, 0.5], [0.5, 1.5]], device=self.device, dtype=dtype
            )
            engine = MultivariateNormalQMCEngine(
                mean=mean, cov=cov, seed=12345, inv_transform=True
            )
            samples = engine.draw(n=256)
            self.assertEqual(samples.dtype, dtype)
            self.assertEqual(samples.device.type, self.device.type)
            self.assertTrue(torch.all(torch.abs(samples.mean(dim=0) - mean) < 1e-2))
            self.assertTrue(
                torch.all(torch.abs(samples.std(dim=0) - math.sqrt(1.5)) < 1e-2)
            )
            # perform Shapiro-Wilk test for normality
            samples = samples.cpu().numpy()
            for i in (0, 1):
                _, pval = shapiro(samples[:, i])
                self.assertGreater(pval, 0.9)
            # check covariance
            cov = np.cov(samples.transpose())
            self.assertLess(np.abs(cov[0, 1] - 0.5), 1e-2)

    def test_MultivariateNormalQMCEngineDegenerate(self):
        for dtype in (torch.float, torch.double):
            # X, Y iid standard Normal and Z = X + Y, random vector (X, Y, Z)
            mean = torch.zeros(3, device=self.device, dtype=dtype)
            cov = torch.tensor(
                [[1, 0, 1], [0, 1, 1], [1, 1, 2]], device=self.device, dtype=dtype
            )
            engine = MultivariateNormalQMCEngine(mean=mean, cov=cov, seed=12345)
            samples = engine.draw(n=4096)
            self.assertEqual(samples.dtype, dtype)
            self.assertEqual(samples.device.type, self.device.type)
            self.assertTrue(torch.all(torch.abs(samples.mean(dim=0)) < 1e-2))
            self.assertTrue(torch.abs(torch.std(samples[:, 0]) - 1) < 1e-2)
            self.assertTrue(torch.abs(torch.std(samples[:, 1]) - 1) < 1e-2)
            self.assertTrue(torch.abs(torch.std(samples[:, 2]) - math.sqrt(2)) < 1e-2)
            for i in (0, 1, 2):
                _, pval = shapiro(samples[:, i].cpu().numpy())
                self.assertGreater(pval, 0.9)
            cov = np.cov(samples.cpu().numpy().transpose())
            self.assertLess(np.abs(cov[0, 1]), 1e-2)
            self.assertLess(np.abs(cov[0, 2] - 1), 1e-2)
            # check to see if X + Y = Z almost exactly
            self.assertTrue(
                torch.all(
                    torch.abs(samples[:, 0] + samples[:, 1] - samples[:, 2]) < 1e-5
                )
            )
