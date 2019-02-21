#!/usr/bin/env python3

import unittest
from collections import Counter

import numpy as np
from botorch.qmc.sobol import SobolEngine, _test_find_index, multinomial_qmc


class SobolTests(unittest.TestCase):
    # set maxDiff to None to show all differences when tests fail
    maxDiff = None

    def setUp(self):
        engine_unscrambled_1d = SobolEngine(1)
        self.draws_unscrambled_1d = engine_unscrambled_1d.draw(10)
        engine_unscrambled_3d = SobolEngine(3)
        self.draws_unscrambled_3d = engine_unscrambled_3d.draw(10)
        engine_scrambled_1d = SobolEngine(1, scramble=True, seed=12345)
        self.draws_scrambled_1d = engine_scrambled_1d.draw(10)
        engine_scrambled_3d = SobolEngine(3, scramble=True, seed=12345)
        self.draws_scrambled_3d = engine_scrambled_3d.draw(10)

    def testUnscrambled1DSobol(self):
        expected = [0.5, 0.75, 0.25, 0.375, 0.875, 0.625, 0.125, 0.1875, 0.6875, 0.9375]
        self.assertEqual(self.draws_unscrambled_1d.shape[0], 10)
        self.assertEqual(self.draws_unscrambled_1d.shape[1], 1)
        self.assertTrue(
            np.array_equal(self.draws_unscrambled_1d.flatten(), np.array(expected))
        )

    def testUnscrambled3DSobol(self):
        expected_dim3 = [
            0.5,
            0.75,
            0.25,
            0.625,
            0.125,
            0.375,
            0.875,
            0.3125,
            0.8125,
            0.5625,
        ]
        self.assertEqual(self.draws_unscrambled_3d.shape[0], 10)
        self.assertEqual(self.draws_unscrambled_3d.shape[1], 3)
        self.assertTrue(
            np.array_equal(self.draws_unscrambled_3d[:, 2], np.array(expected_dim3))
        )
        self.assertTrue(
            np.array_equal(
                self.draws_unscrambled_3d[:, 0], self.draws_unscrambled_1d.flatten()
            )
        )

    def testUnscrambled3DAsyncSobol(self):
        engine_unscrambled_3d = SobolEngine(3)
        draws = np.vstack([engine_unscrambled_3d.draw() for i in range(10)])
        self.assertTrue(np.array_equal(self.draws_unscrambled_3d, draws))

    def testUnscrambledFastForwardAndResetSobol(self):
        engine_unscrambled_3d = SobolEngine(3).fast_forward(5)
        draws = engine_unscrambled_3d.draw(5)
        self.assertTrue(np.array_equal(self.draws_unscrambled_3d[5:10, :], draws))

        engine_unscrambled_3d.reset()
        even_draws = []
        for i in range(10):
            if i % 2 == 0:
                even_draws.append(engine_unscrambled_3d.draw())
            else:
                engine_unscrambled_3d.fast_forward(1)
        self.assertTrue(
            np.array_equal(
                self.draws_unscrambled_3d[[i for i in range(10) if i % 2 == 0],],
                np.vstack(even_draws),
            )
        )

    def testUnscrambledHighDimSobol(self):
        engine = SobolEngine(1111)
        count1 = Counter(engine.draw().flatten().tolist())
        count2 = Counter(engine.draw().flatten().tolist())
        count3 = Counter(engine.draw().flatten().tolist())
        self.assertEqual(count1, Counter({0.5: 1111}))
        self.assertEqual(count2, Counter({0.25: 580, 0.75: 531}))
        self.assertEqual(count3, Counter({0.25: 531, 0.75: 580}))

    def testUnscrambledSobolBounds(self):
        engine = SobolEngine(1111)
        draws = engine.draw(1000)
        self.assertTrue(np.all(draws >= 0))
        self.assertTrue(np.all(draws <= 1))

    def testUnscrambledDistributionSobol(self):
        engine = SobolEngine(1111)
        draws = engine.draw(1000)
        self.assertTrue(
            np.allclose(np.mean(draws, axis=0), np.repeat(0.5, 1111), atol=0.01)
        )
        self.assertTrue(
            np.allclose(
                np.percentile(draws, 25, axis=0), np.repeat(0.25, 1111), atol=0.01
            )
        )
        self.assertTrue(
            np.allclose(
                np.percentile(draws, 75, axis=0), np.repeat(0.75, 1111), atol=0.01
            )
        )

    def testScrambled1DSobol(self):
        expected = [
            0.46784395,
            0.03562005,
            0.91319746,
            0.86014303,
            0.23796839,
            0.25856809,
            0.63636296,
            0.69455189,
            0.316758,
            0.18673652,
        ]
        print(self.draws_scrambled_1d.flatten())
        self.assertEqual(self.draws_scrambled_1d.shape[0], 10)
        self.assertEqual(self.draws_scrambled_1d.shape[1], 1)
        self.assertTrue(
            np.allclose(self.draws_scrambled_1d.flatten(), np.array(expected))
        )

    def testScrambled3DSobol(self):
        expected_dim3 = [
            0.19711632,
            0.43653634,
            0.79965184,
            0.08670237,
            0.70811484,
            0.90994149,
            0.29499525,
            0.83833538,
            0.46057166,
            0.15769824,
        ]
        self.assertEqual(self.draws_scrambled_3d.shape[0], 10)
        self.assertEqual(self.draws_scrambled_3d.shape[1], 3)
        self.assertTrue(
            np.allclose(
                self.draws_scrambled_3d[:, 2], np.array(expected_dim3), atol=1e-5
            )
        )

    def testScrambled3DAsyncSobol(self):
        engine_unscrambled_3d = SobolEngine(3)
        draws = np.vstack([engine_unscrambled_3d.draw() for i in range(10)])
        self.assertTrue(np.array_equal(self.draws_unscrambled_3d, draws))

    def testScrambledSobolBounds(self):
        engine = SobolEngine(100, scramble=True)
        draws = engine.draw(1000)
        self.assertTrue(np.all(draws >= 0))
        self.assertTrue(np.all(draws <= 1))

    def testScrambledFastForwardAndResetSobol(self):
        engine_scrambled_3d = SobolEngine(3, scramble=True, seed=12345).fast_forward(5)
        draws = engine_scrambled_3d.draw(5)
        self.assertTrue(np.array_equal(self.draws_scrambled_3d[5:10,], draws))

        engine_scrambled_3d.reset()
        even_draws = []
        for i in range(10):
            if i % 2 == 0:
                even_draws.append(engine_scrambled_3d.draw())
            else:
                engine_scrambled_3d.fast_forward(1)
        self.assertTrue(
            np.array_equal(
                self.draws_scrambled_3d[[i for i in range(10) if i % 2 == 0],],
                np.vstack(even_draws),
            )
        )

    def testScrambledDistributionSobol(self):
        engine = SobolEngine(10, scramble=True, seed=12345)
        draws = engine.draw(1000)
        self.assertTrue(
            np.allclose(np.mean(draws, axis=0), np.repeat(0.5, 10), atol=0.01)
        )
        self.assertTrue(
            np.allclose(
                np.percentile(draws, 25, axis=0), np.repeat(0.25, 10), atol=0.01
            )
        )
        self.assertTrue(
            np.allclose(
                np.percentile(draws, 75, axis=0), np.repeat(0.75, 10), atol=0.01
            )
        )

    def test0Dim(self):
        engine = SobolEngine(0)
        draws = engine.draw(5)
        self.assertTrue(np.array_equal(np.empty((5, 0)), draws))


class MultinomialQMCTests(unittest.TestCase):
    def testMultinomialNegativePs(self):
        p = np.array([0.12, 0.26, -0.05, 0.35, 0.22])
        self.assertRaises(ValueError, multinomial_qmc, 10, p)

    def testMultinomialSumOfPTooLarge(self):
        p = np.array([0.12, 0.26, 0.1, 0.35, 0.22])
        self.assertRaises(ValueError, multinomial_qmc, 10, p)

    def testMultinomialBasicDraw(self):
        p = np.array([0.12, 0.26, 0.05, 0.35, 0.22])
        expected = np.array([12, 25, 6, 34, 23])
        self.assertTrue(np.array_equal(multinomial_qmc(100, p, seed=12345), expected))

    def testMultinomialDistribution(self):
        p = np.array([0.12, 0.26, 0.05, 0.35, 0.22])
        draws = multinomial_qmc(10000, p, seed=12345)
        np.testing.assert_almost_equal(draws / np.sum(draws), p, decimal=4)

    def testFindIndex(self):
        p_cumulative = np.array([0.1, 0.4, 0.45, 0.6, 0.75, 0.9, 0.99, 1.0])
        size = len(p_cumulative)
        self.assertEqual(_test_find_index(p_cumulative, size, 0.0), 0)
        self.assertEqual(_test_find_index(p_cumulative, size, 0.4), 2)
        self.assertEqual(_test_find_index(p_cumulative, size, 0.44999), 2)
        self.assertEqual(_test_find_index(p_cumulative, size, 0.45001), 3)
        self.assertEqual(_test_find_index(p_cumulative, size, 1.0), size - 1)
