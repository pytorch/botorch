#!/usr/bin/env python3

import unittest

import numpy as np
from botorch.qmc import MultivariateNormalQMCEngine, NormalQMCEngine
from scipy.stats import shapiro


class NormalQMCTests(unittest.TestCase):
    def test_NormalQMCEngine(self):
        # d = 1
        engine = NormalQMCEngine(d=1)
        samples = engine.draw()
        self.assertEqual(samples.shape, (1, 1))
        samples = engine.draw(n=5)
        self.assertEqual(samples.shape, (5, 1))
        # d = 2
        engine = NormalQMCEngine(d=2)
        samples = engine.draw()
        self.assertEqual(samples.shape, (1, 2))
        samples = engine.draw(n=5)
        self.assertEqual(samples.shape, (5, 2))

    def test_NormalQMCEngineInvTransform(self):
        # d = 1
        engine = NormalQMCEngine(d=1, inv_transform=True)
        samples = engine.draw()
        self.assertEqual(samples.shape, (1, 1))
        samples = engine.draw(n=5)
        self.assertEqual(samples.shape, (5, 1))
        # d = 2
        engine = NormalQMCEngine(d=2, inv_transform=True)
        samples = engine.draw()
        self.assertEqual(samples.shape, (1, 2))
        samples = engine.draw(n=5)
        self.assertEqual(samples.shape, (5, 2))

    def test_NormalQMCEngineSeeded(self):
        # test even dimension
        engine = NormalQMCEngine(d=2, seed=12345)
        samples = engine.draw(n=2)
        samples_expected = np.array(
            [[-0.63099602, -1.32950772], [0.29625805, 1.86425618]]
        )
        self.assertTrue(np.allclose(samples, samples_expected))
        # test odd dimension
        engine = NormalQMCEngine(d=3, seed=12345)
        samples = engine.draw(n=2)
        samples_expected = np.array(
            [
                [1.83169884, -1.40473647, 0.24334828],
                [0.36596099, 1.2987395, -1.47556275],
            ]
        )
        self.assertTrue(np.allclose(samples, samples_expected))

    def test_NormalQMCEngineSeededInvTransform(self):
        # test even dimension
        engine = NormalQMCEngine(d=2, seed=12345, inv_transform=True)
        samples = engine.draw(n=2)
        samples_expected = np.array(
            [[-0.41622922, 0.46622792], [-0.96063897, -0.75568963]]
        )
        self.assertTrue(np.allclose(samples, samples_expected))
        # test odd dimension
        engine = NormalQMCEngine(d=3, seed=12345, inv_transform=True)
        samples = engine.draw(n=2)
        samples_expected = np.array(
            [
                [-1.40525266, 1.37652443, -0.8519666],
                [-0.166497, -2.3153681, -0.15975676],
            ]
        )
        self.assertTrue(np.allclose(samples, samples_expected))

    def test_NormalQMCEngineShapiro(self):
        engine = NormalQMCEngine(d=2, seed=12345)
        samples = engine.draw(n=250)
        self.assertTrue(all(np.abs(samples.mean(axis=0)) < 1e-2))
        self.assertTrue(all(np.abs(samples.std(axis=0) - 1) < 1e-2))
        # perform Shapiro-Wilk test for normality
        for i in (0, 1):
            _, pval = shapiro(samples[:, i])
            self.assertGreater(pval, 0.9)
        # make sure samples are uncorrelated
        cov = np.cov(samples.transpose())
        self.assertLess(np.abs(cov[0, 1]), 1e-2)

    def test_NormalQMCEngineShapiroInvTransform(self):
        engine = NormalQMCEngine(d=2, seed=12345, inv_transform=True)
        samples = engine.draw(n=250)
        self.assertTrue(all(np.abs(samples.mean(axis=0)) < 1e-2))
        self.assertTrue(all(np.abs(samples.std(axis=0) - 1) < 1e-2))
        # perform Shapiro-Wilk test for normality
        for i in (0, 1):
            _, pval = shapiro(samples[:, i])
            self.assertGreater(pval, 0.9)
        # make sure samples are uncorrelated
        cov = np.cov(samples.transpose())
        self.assertLess(np.abs(cov[0, 1]), 1e-2)


class MultivariateNormalQMCTests(unittest.TestCase):
    def test_MultivariateNormalQMCEngineNonPSD(self):
        # try with non-psd, non-pd cov and expect an assertion error
        self.assertRaises(
            ValueError, MultivariateNormalQMCEngine, [0, 0], [[1, 2], [2, 1]]
        )

    def test_MultivariateNormalQMCEngineNonPD(self):
        # try with non-pd but psd cov; should work
        engine = MultivariateNormalQMCEngine(
            [0, 0, 0], [[1, 0, 1], [0, 1, 1], [1, 1, 2]]
        )
        self.assertTrue(engine._corr_matrix is not None)

    def test_MultivariateNormalQMCEngineSymmetric(self):
        # try with non-symmetric cov and expect an error
        self.assertRaises(
            ValueError, MultivariateNormalQMCEngine, [0, 0], [[1, 0], [2, 1]]
        )

    def test_MultivariateNormalQMCEngine(self):
        # d = 1 scalar
        engine = MultivariateNormalQMCEngine(mean=0, cov=5)
        samples = engine.draw()
        self.assertEqual(samples.shape, (1, 1))
        samples = engine.draw(n=5)
        self.assertEqual(samples.shape, (5, 1))

        # d = 2 list
        engine = MultivariateNormalQMCEngine(mean=[0, 1], cov=[[1, 0], [0, 1]])
        samples = engine.draw()
        self.assertEqual(samples.shape, (1, 2))
        samples = engine.draw(n=5)
        self.assertEqual(samples.shape, (5, 2))

        # d = 3 np.array
        mean = np.array([0, 1, 2])
        cov = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        engine = MultivariateNormalQMCEngine(mean, cov)
        samples = engine.draw()
        self.assertEqual(samples.shape, (1, 3))
        samples = engine.draw(n=5)
        self.assertEqual(samples.shape, (5, 3))

    def test_MultivariateNormalQMCEngineInvTransform(self):
        # d = 1 scalar
        engine = MultivariateNormalQMCEngine(mean=0, cov=5, inv_transform=True)
        samples = engine.draw()
        self.assertEqual(samples.shape, (1, 1))
        samples = engine.draw(n=5)
        self.assertEqual(samples.shape, (5, 1))

        # d = 2 list
        engine = MultivariateNormalQMCEngine(
            mean=[0, 1], cov=[[1, 0], [0, 1]], inv_transform=True
        )
        samples = engine.draw()
        self.assertEqual(samples.shape, (1, 2))
        samples = engine.draw(n=5)
        self.assertEqual(samples.shape, (5, 2))

        # d = 3 np.array
        mean = np.array([0, 1, 2])
        cov = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        engine = MultivariateNormalQMCEngine(mean, cov, inv_transform=True)
        samples = engine.draw()
        self.assertEqual(samples.shape, (1, 3))
        samples = engine.draw(n=5)
        self.assertEqual(samples.shape, (5, 3))

    def test_MultivariateNormalQMCEngineSeeded(self):
        # test even dimension
        np.random.seed(54321)
        a = np.random.randn(2, 2)
        A = a @ a.transpose() + np.diag(np.random.rand(2))
        engine = MultivariateNormalQMCEngine(np.array([0, 0]), A, seed=12345)
        samples = engine.draw(n=2)
        samples_expected = np.array(
            [[-0.67595995, -2.27437872], [0.317369, 2.66203577]]
        )
        self.assertTrue(np.allclose(samples, samples_expected))

        # test odd dimension
        np.random.seed(54321)
        a = np.random.randn(3, 3)
        A = a @ a.transpose() + np.diag(np.random.rand(3))
        engine = MultivariateNormalQMCEngine(np.array([0, 0, 0]), A, seed=12345)
        samples = engine.draw(n=2)
        samples_expected = np.array(
            [
                [2.05178452, -6.35744194, 0.67944512],
                [0.40993262, 2.60517697, -1.69415825],
            ]
        )
        self.assertTrue(np.allclose(samples, samples_expected))

    def test_MultivariateNormalQMCEngineSeededInvTransform(self):
        # test even dimension
        np.random.seed(54321)
        a = np.random.randn(2, 2)
        A = a @ a.transpose() + np.diag(np.random.rand(2))
        engine = MultivariateNormalQMCEngine(
            np.array([0, 0]), A, seed=12345, inv_transform=True
        )
        samples = engine.draw(n=2)
        samples_expected = np.array(
            [[-0.44588916, 0.22657776], [-1.02909281, -1.83193033]]
        )
        self.assertTrue(np.allclose(samples, samples_expected))

        # test odd dimension
        np.random.seed(54321)
        a = np.random.randn(3, 3)
        A = a @ a.transpose() + np.diag(np.random.rand(3))
        engine = MultivariateNormalQMCEngine(
            np.array([0, 0, 0]), A, seed=12345, inv_transform=True
        )
        samples = engine.draw(n=2)
        samples_expected = np.array(
            [
                [-1.5740992, 5.61057598, -1.28218525],
                [-0.18650226, -5.41662685, 0.023199],
            ]
        )
        self.assertTrue(np.allclose(samples, samples_expected))

    def test_MultivariateNormalQMCEngineShapiro(self):
        # test the standard case
        engine = MultivariateNormalQMCEngine(
            mean=[0, 0], cov=[[1, 0], [0, 1]], seed=12345
        )
        samples = engine.draw(n=250)
        self.assertTrue(all(np.abs(samples.mean(axis=0)) < 1e-2))
        self.assertTrue(all(np.abs(samples.std(axis=0) - 1) < 1e-2))
        # perform Shapiro-Wilk test for normality
        for i in (0, 1):
            _, pval = shapiro(samples[:, i])
            self.assertGreater(pval, 0.9)
        # make sure samples are uncorrelated
        cov = np.cov(samples.transpose())
        self.assertLess(np.abs(cov[0, 1]), 1e-2)

        # test the correlated, non-zero mean case
        engine = MultivariateNormalQMCEngine(
            mean=[1.0, 2.0], cov=[[1.5, 0.5], [0.5, 1.5]], seed=12345
        )
        samples = engine.draw(n=250)
        self.assertTrue(all(np.abs(samples.mean(axis=0) - [1, 2]) < 1e-2))
        self.assertTrue(all(np.abs(samples.std(axis=0) - np.sqrt(1.5)) < 1e-2))
        # perform Shapiro-Wilk test for normality
        for i in (0, 1):
            _, pval = shapiro(samples[:, i])
            self.assertGreater(pval, 0.9)
        # check covariance
        cov = np.cov(samples.transpose())
        self.assertLess(np.abs(cov[0, 1] - 0.5), 1e-2)

    def test_MultivariateNormalQMCEngineShapiroInvTransform(self):
        # test the standard case
        engine = MultivariateNormalQMCEngine(
            mean=[0, 0], cov=[[1, 0], [0, 1]], seed=12345, inv_transform=True
        )
        samples = engine.draw(n=250)
        self.assertTrue(all(np.abs(samples.mean(axis=0)) < 1e-2))
        self.assertTrue(all(np.abs(samples.std(axis=0) - 1) < 1e-2))
        # perform Shapiro-Wilk test for normality
        for i in (0, 1):
            _, pval = shapiro(samples[:, i])
            self.assertGreater(pval, 0.9)
        # make sure samples are uncorrelated
        cov = np.cov(samples.transpose())
        self.assertLess(np.abs(cov[0, 1]), 1e-2)

        # test the correlated, non-zero mean case
        engine = MultivariateNormalQMCEngine(
            mean=[1.0, 2.0],
            cov=[[1.5, 0.5], [0.5, 1.5]],
            seed=12345,
            inv_transform=True,
        )
        samples = engine.draw(n=250)
        self.assertTrue(all(np.abs(samples.mean(axis=0) - [1, 2]) < 1e-2))
        self.assertTrue(all(np.abs(samples.std(axis=0) - np.sqrt(1.5)) < 1e-2))
        # perform Shapiro-Wilk test for normality
        for i in (0, 1):
            _, pval = shapiro(samples[:, i])
            self.assertGreater(pval, 0.9)
        # check covariance
        cov = np.cov(samples.transpose())
        self.assertLess(np.abs(cov[0, 1] - 0.5), 1e-2)

    def test_MultivariateNormalQMCEngineDegenerate(self):
        # X, Y iid standard Normal and Z = X + Y, random vector (X, Y, Z)
        engine = MultivariateNormalQMCEngine(
            mean=[0.0, 0.0, 0.0],
            cov=[[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 2.0]],
            seed=12345,
        )
        samples = engine.draw(n=2000)
        self.assertTrue(all(np.abs(samples.mean(axis=0)) < 1e-2))
        self.assertTrue(np.abs(np.std(samples[:, 0]) - 1) < 1e-2)
        self.assertTrue(np.abs(np.std(samples[:, 1]) - 1) < 1e-2)
        self.assertTrue(np.abs(np.std(samples[:, 2]) - np.sqrt(2)) < 1e-2)
        for i in (0, 1, 2):
            _, pval = shapiro(samples[:, i])
            self.assertGreater(pval, 0.9)
        cov = np.cov(samples.transpose())
        self.assertLess(np.abs(cov[0, 1]), 1e-2)
        self.assertLess(np.abs(cov[0, 2] - 1), 1e-2)
        # check to see if X + Y = Z almost exactly
        self.assertTrue(
            all(np.abs(samples[:, 0] + samples[:, 1] - samples[:, 2]) < 1e-5)
        )
