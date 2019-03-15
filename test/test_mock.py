#!/usr/bin/env python3

import unittest

import torch

from .mock import MockModel, MockPosterior


class TestMock(unittest.TestCase):
    def test_MockPosterior(self):
        mean = torch.rand(2)
        variance = torch.eye(2)
        samples = torch.rand(1, 2)
        mp = MockPosterior(mean=mean, variance=variance, samples=samples)
        self.assertTrue(torch.equal(mp.mean, mean))
        self.assertTrue(torch.equal(mp.variance, variance))
        self.assertTrue(torch.all(mp.sample() == samples.unsqueeze(0)))
        self.assertTrue(
            torch.all(mp.sample(torch.Size([2])) == samples.repeat(2, 1, 1))
        )

    def test_MockModel(self):
        mp = MockPosterior()
        mm = MockModel(mp)
        X = torch.empty(0)
        self.assertEqual(mm.posterior(X), mp)
