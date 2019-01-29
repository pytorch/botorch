#!/usr/bin/env python3
import unittest

import torch
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from gpytorch.lazy.non_lazy_tensor import lazify


class TestGPyTorchPosterior(unittest.TestCase):
    def test_rsample_degenerate(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            degenerate_covar = torch.tensor(
                [[0.1, 0.0, 2.0], [0.1, 0.1, 1.0], [0.0, 0.0, 2.0]],
                dtype=dtype,
                device=device,
            )
            mean = torch.zeros(3, dtype=dtype, device=device)
            mvn = MultivariateNormal(mean, lazify(degenerate_covar))
            posterior = GPyTorchPosterior(mvn=mvn)
            samples = posterior.rsample(sample_shape=torch.Size([4]))
            self.assertEqual(list(samples.shape), [4, 3])

            # test multitask
            degenerate_covar2 = torch.zeros((6, 6), device=device, dtype=dtype)
            degenerate_covar2[:3, :3] = degenerate_covar
            degenerate_covar2[3:, 3:] = degenerate_covar
            mean2 = torch.zeros((3, 2), dtype=dtype, device=device)
            mvn2 = MultitaskMultivariateNormal(mean2, lazify(degenerate_covar2))
            posterior2 = GPyTorchPosterior(mvn=mvn2)
            samples2 = posterior2.rsample(sample_shape=torch.Size([4]))
            self.assertEqual(list(samples2.shape), [4, 3, 2])

    def test_rsample_degenerate_cuda(self):
        if torch.cuda.is_available():
            self.test_rsample_degenerate(cuda=True)
