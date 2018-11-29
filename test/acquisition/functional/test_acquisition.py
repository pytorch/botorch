#! /usr/bin/env python3

import unittest

import torch
from botorch.acquisition.functional.acquisition import (
    expected_improvement,
    posterior_mean,
    probability_of_improvement,
    upper_confidence_bound,
)

from ...utils.mock import MockModel, MockPosterior
from ...utils.utils import approx_equal


class TestFunctionalAcquisition(unittest.TestCase):
    def test_expected_improvement(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            mean = torch.tensor([0.0, 0.5], device=device, dtype=dtype)
            variance = torch.ones(2, device=device, dtype=dtype)
            mm = MockModel(MockPosterior(mean=mean, variance=variance))
            X = torch.empty(2, 1, device=device, dtype=dtype)  # dummy Tensor for shape
            ei = expected_improvement(X=X, model=mm, best_f=0.0)
            ei_expected = torch.tensor([0.39894, 0.69780], device=device, dtype=dtype)
            self.assertTrue(approx_equal(ei, ei_expected))

    def test_expected_improvement_cuda(self, cuda=False):
        if torch.cuda.is_available():
            self.test_expected_improvement(cuda=True)

    def test_expected_improvement_batch(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            mean = torch.tensor([[0.0, 0.5], [1.0, -0.5]], device=device, dtype=dtype)
            variance = torch.ones_like(mean)
            mm = MockModel(MockPosterior(mean=mean, variance=variance))
            # dummy Tensor for shape
            X = torch.empty(2, 2, 1, device=device, dtype=dtype)
            best_f = torch.zeros(2, device=device, dtype=dtype)
            ei = expected_improvement(X=X, model=mm, best_f=best_f)
            ei_expected = torch.tensor(
                [[0.39894, 0.69780], [1.08332, 0.39894]], device=device, dtype=dtype
            )
            self.assertTrue(approx_equal(ei, ei_expected))

    def test_expected_improvement_batch_cuda(self, cuda=False):
        if torch.cuda.is_available():
            self.test_expected_improvement_batch(cuda=True)

    def test_posterior_mean(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            mean = torch.tensor([0.0, 0.5], device=device, dtype=dtype)
            mm = MockModel(MockPosterior(mean=mean))
            pm = posterior_mean(
                X=torch.tensor(0.0, device=device, dtype=dtype), model=mm
            )
            self.assertTrue(torch.equal(pm, mean))

    def test_posterior_mean_cuda(self, cuda=False):
        if torch.cuda.is_available():
            self.test_posterior_mean(cuda=True)

    def test_posterior_mean_batch(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            mean = torch.tensor([[0.0, 0.5], [1.0, -0.5]], device=device, dtype=dtype)
            mm = MockModel(MockPosterior(mean=mean))
            pm = posterior_mean(
                X=torch.tensor(0.0, device=device, dtype=dtype), model=mm
            )
            self.assertTrue(torch.equal(pm, mean))

    def test_posterior_mean_batch_cuda(self, cuda=False):
        if torch.cuda.is_available():
            self.test_posterior_mean_batch(cuda=True)

    def test_probability_of_improvement(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            mean = torch.tensor([0.0, 0.67449], device=device, dtype=dtype)
            variance = torch.ones(2, device=device, dtype=dtype)
            mm = MockModel(MockPosterior(mean=mean, variance=variance))
            X = torch.empty(2, 1, device=device, dtype=dtype)  # dummy Tensor for shape
            pi = probability_of_improvement(X=X, model=mm, best_f=0.0)
            pi_expected = torch.tensor([0.5, 0.75], device=device, dtype=dtype)
            self.assertTrue(approx_equal(pi, pi_expected))

    def test_probability_of_improvement_cuda(self, cuda=False):
        if torch.cuda.is_available():
            self.test_probability_of_improvement(cuda=True)

    def test_probability_of_improvement_batch(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            mean = torch.tensor(
                [[0.0, 0.67449], [-0.67449, 1.28155]], device=device, dtype=dtype
            )
            variance = torch.ones_like(mean)
            mm = MockModel(MockPosterior(mean=mean, variance=variance))
            # dummy Tensor for shape
            X = torch.empty(2, 2, 1, device=device, dtype=dtype)
            best_f = torch.zeros(2, device=device, dtype=dtype)
            pi = probability_of_improvement(X=X, model=mm, best_f=best_f)
            pi_expected = torch.tensor(
                [[0.5, 0.75], [0.25, 0.9]], device=device, dtype=dtype
            )
            self.assertTrue(approx_equal(pi, pi_expected))

    def test_probability_of_improvement_batch_cuda(self, cuda=False):
        if torch.cuda.is_available():
            self.test_probability_of_improvement_batch(cuda=True)

    def test_upper_confidence_bound(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            mean = torch.tensor([0.0, 0.5], device=device, dtype=dtype)
            variance = torch.tensor([1.0, 4.0], device=device, dtype=dtype)
            mm = MockModel(MockPosterior(mean=mean, variance=variance))
            X = torch.empty(2, 1, device=device, dtype=dtype)  # dummy Tensor for shape
            ucb = upper_confidence_bound(X=X, model=mm, beta=1.0)
            ucb_expected = torch.tensor([1.0, 2.5], device=device, dtype=dtype)
            self.assertTrue(approx_equal(ucb, ucb_expected))

    def test_upper_confidence_bound_cuda(self, cuda=False):
        if torch.cuda.is_available():
            self.test_upper_confidence_bound(cuda=True)

    def test_upper_confidence_bound_batch(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            mean = torch.tensor([0.0, 0.5], device=device, dtype=dtype).repeat(2, 1)
            variance = torch.tensor([1.0, 4.0], device=device, dtype=dtype).repeat(2, 1)
            mm = MockModel(MockPosterior(mean=mean, variance=variance))
            # dummy Tensor for shape
            X = torch.empty(2, 2, 1, device=device, dtype=dtype)
            beta = torch.tensor([1.0, 2.0], device=device, dtype=dtype)
            ucb = upper_confidence_bound(X=X, model=mm, beta=beta)
            ucb_expected = torch.tensor(
                [[1.0, 2.5], [1.41421, 3.32842]], device=device, dtype=dtype
            )
            self.assertTrue(approx_equal(ucb, ucb_expected))

    def test_upper_confidence_bound_batch_cuda(self, cuda=False):
        if torch.cuda.is_available():
            self.test_upper_confidence_bound_batch(cuda=True)


if __name__ == "__main__":
    unittest.main()
