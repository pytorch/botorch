#!/usr/bin/env python3

import unittest

import numpy as np
from botorch.qmc import NormalQMCEngine


class NormalQMCTests(unittest.TestCase):
    def testNormalQMCEngine(self):
        engine = NormalQMCEngine()
        samples = engine.draw()
        self.assertEqual(len(samples), 1)
        samples = engine.draw(n=5)
        self.assertEqual(len(samples), 5)

    def testNormalQMCEngineSeeded(self):
        engine = NormalQMCEngine(seed=1234)
        samples = engine.draw(n=2)
        samples_expected = np.array([-0.73137909, -0.64682878])
        self.assertTrue(np.allclose(np.sort(samples), np.sort(samples_expected)))

    def testNormalQMCEngineShapiro(self):
        engine = NormalQMCEngine(seed=1234)
        samples = engine.draw(n=250)
        # check mean and standard deviation
        self.assertEqual(len(samples), 250)
        self.assertLess(np.abs(samples.mean()), 1e-2)
        self.assertLess(np.abs(samples.std() - 1), 1e-2)
        # if scipy is available, perform Shapiro-Wilk test for normality
        try:
            from scipy.stats import shapiro

            _, pval = shapiro(samples)
            self.assertGreater(pval, 0.99)
        except ModuleNotFoundError:
            pass


if __name__ == "__main__":
    unittest.main()
