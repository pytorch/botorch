#!/usr/bin/env python3

import unittest

import torch

from .mock import MockLikelihood, MockModel


class TestMock(unittest.TestCase):
    def test_mockmodel(self):
        o = object()
        mm = MockModel(o)
        mm.eval()
        self.assertEqual(mm(None), o)

    def test_mocklikelihood(self):
        mean = torch.rand(2)
        covariance = torch.eye(2)
        samples = torch.rand(1, 2)
        ml = MockLikelihood(mean=mean, covariance=covariance, samples=samples)
        self.assertTrue(torch.equal(ml.mean, mean))
        self.assertTrue(torch.equal(ml.covariance_matrix, covariance))
        self.assertTrue(torch.all(ml.rsample() == samples.unsqueeze(0)))
        self.assertTrue(
            torch.all(ml.rsample(torch.Size([2])) == samples.repeat(2, 1, 1))
        )
