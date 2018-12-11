#! /usr/bin/env python3

import unittest
from test.utils.mock import MockModel, MockPosterior

import torch
from botorch.acquisition.functional.thompson_sampling_utils import (
    discrete_thompson_sample,
)


class TestThompsonSamplingUtils(unittest.TestCase):
    def test_discrete_thompson_sample(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            # Test with batch
            samples = torch.zeros([1, 3, 1], device=device, dtype=dtype)
            res = discrete_thompson_sample(
                X=torch.zeros(1, device=device, dtype=dtype),  # dummy for type checking
                model=MockModel(MockPosterior(samples=samples)),
                objective=lambda Y: Y,
                constraints=[lambda Y: torch.zeros_like(Y, device=device, dtype=dtype)],
                mc_samples=2,
            )
            self.assertEqual(res[0][0], 1)
            self.assertEqual(res[1][0], 1)
            self.assertEqual(res[2][0], 1)
            self.assertEqual(res.shape, (3, 1))

            # Test without a batch
            samples = torch.zeros([1, 1, 1], device=device, dtype=dtype)
            res = discrete_thompson_sample(
                X=torch.zeros(1, device=device, dtype=dtype),  # dummy for type checking
                model=MockModel(MockPosterior(samples=samples)),
                objective=lambda Y: Y,
                constraints=[lambda Y: torch.zeros_like(Y, device=device, dtype=dtype)],
                mc_samples=2,
            )
            self.assertEqual(res.item(), 1)

            # Test with two different samples
            samples = torch.zeros([1, 1, 2], device=device, dtype=dtype)
            samples[:, :, 1] = samples[:, :, 1] + 1
            res = discrete_thompson_sample(
                X=torch.zeros(1, device=device, dtype=dtype),  # dummy for type checking
                model=MockModel(MockPosterior(samples=samples)),
                objective=lambda Y: Y,
                constraints=[lambda Y: torch.zeros_like(Y, device=device, dtype=dtype)],
                mc_samples=2,
            )
            # Note due to batch mode transform decorator this is not (1, 2)
            self.assertEqual(res.shape, (2,))
            self.assertEqual(res[0], 0)
            self.assertEqual(res[1], 1)

            # Test with two different samples, two batches
            samples = torch.zeros([1, 2, 2], device=device, dtype=dtype)
            samples[:, 1, 1] = samples[:, 1, 1] + 1
            samples[:, 0, 0] = samples[:, 0, 0] + 1
            res = discrete_thompson_sample(
                X=torch.zeros(1, device=device, dtype=dtype),  # dummy for type checking
                model=MockModel(MockPosterior(samples=samples)),
                objective=lambda Y: Y,
                constraints=[lambda Y: torch.zeros_like(Y, device=device, dtype=dtype)],
                mc_samples=2,
            )
            self.assertEqual(res.shape, (2, 2))
            self.assertEqual(res[0][0], 1)
            self.assertEqual(res[0][1], 0)
            self.assertEqual(res[1][0], 0)
            self.assertEqual(res[1][1], 1)

    def test_discrete_thompson_sample_cuda(self):
        if torch.cuda.is_available():
            self.test_discrete_thompson_sample(cuda=True)


if __name__ == "__main__":
    unittest.main()
