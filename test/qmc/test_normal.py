#!/usr/bin/env python3

import unittest

import numpy as np
from botorch.qmc import NormalQMCEngine
from scipy.stats import shapiro


class NormalQMCTests(unittest.TestCase):
    def testNormalQMCEngine(self):
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

    def testNormalQMCEngineInvTransform(self):
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

    def testNormalQMCEngineSeeded(self):
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

    def testNormalQMCEngineSeededInvTransform(self):
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

    def testNormalQMCEngineShapiro(self):
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

    def testNormalQMCEngineShapiroInvTransform(self):
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


if __name__ == "__main__":
    unittest.main()
