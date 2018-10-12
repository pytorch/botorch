#! /usr/bin/env python3

import unittest

import torch
from botorch.acquisition.modules import (
    AcquisitionFunction,
    ExpectedImprovement,
    PosteriorMean,
    ProbabilityOfImprovement,
    UpperConfidenceBound,
)

from ..utils.mock import MockLikelihood, MockModel
from ..utils.utils import approx_equal


class TestAcquisitionModules(unittest.TestCase):
    def test_acquisition_function_abstract_module(self):
        with self.assertRaises(TypeError):
            AcquisitionFunction()

    def test_expected_improvement_module(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        mean = torch.tensor([0.0, 0.5], device=device)
        cov = torch.eye(2, device=device)
        mm = MockModel(MockLikelihood(mean=mean, covariance=cov))
        module = ExpectedImprovement(model=mm, best_f=0.0)
        X = torch.empty(2, 1, device=device)  # dummy Tensor for shape
        ei = module(X)
        ei_expected = torch.tensor([0.39894, 0.69780], device=device)
        self.assertTrue(approx_equal(ei, ei_expected))

    def test_expected_improvement_module_cuda(self, cuda=False):
        if torch.cuda.is_available():
            self.test_expected_improvement_module(cuda=True)

    def test_expected_improvement_module_batch(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        mean = torch.tensor([[0.0, 0.5], [1.0, -0.5]], device=device)
        cov = torch.eye(2, device=device).repeat(2, 1, 1)
        mm = MockModel(MockLikelihood(mean=mean, covariance=cov))
        best_f = torch.zeros(2, device=device)
        module = ExpectedImprovement(model=mm, best_f=best_f)
        X = torch.empty(2, 2, 1, device=device)  # dummy Tensor for shape
        ei = module(X)
        ei_expected = torch.tensor(
            [[0.39894, 0.69780], [1.08332, 0.39894]], device=device
        )
        self.assertTrue(approx_equal(ei, ei_expected))

    def test_expected_improvement_module_batch_cuda(self, cuda=False):
        if torch.cuda.is_available():
            self.test_expected_improvement_module_batch(cuda=True)

    def test_posterior_mean_module(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        mean = torch.tensor([0.0, 0.5], device=device)
        mm = MockModel(MockLikelihood(mean=mean))
        module = PosteriorMean(model=mm)
        pm = module(torch.tensor(0.0, device=device))
        self.assertTrue(torch.equal(pm, mean))

    def test_posterior_mean_module_cuda(self, cuda=False):
        if torch.cuda.is_available():
            self.test_posterior_mean_module(cuda=True)

    def test_posterior_mean_module_batch(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        mean = torch.tensor([[0.0, 0.5], [1.0, -0.5]], device=device)
        mm = MockModel(MockLikelihood(mean=mean))
        module = PosteriorMean(model=mm)
        pm = module(torch.tensor(0.0, device=device))
        self.assertTrue(torch.equal(pm, mean))

    def test_posterior_mean_module_batch_cuda(self, cuda=False):
        if torch.cuda.is_available():
            self.test_posterior_mean_module_batch(cuda=True)

    def test_probability_of_improvement_module(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        mean = torch.tensor([0.0, 0.67449], device=device)
        cov = torch.eye(2, device=device)
        mm = MockModel(MockLikelihood(mean=mean, covariance=cov))
        module = ProbabilityOfImprovement(model=mm, best_f=0.0)
        X = torch.empty(2, 1, device=device)  # dummy Tensor for shape
        pi = module(X)
        pi_expected = torch.tensor([0.5, 0.75], device=device)
        self.assertTrue(approx_equal(pi, pi_expected))

    def test_probability_of_improvement_module_cuda(self, cuda=False):
        if torch.cuda.is_available():
            self.test_probability_of_improvement_module(cuda=True)

    def test_probability_of_improvement_module_batch(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        mean = torch.tensor([[0.0, 0.67449], [-0.67449, 1.28155]], device=device)
        cov = torch.eye(2, device=device).repeat(2, 1, 1)
        mm = MockModel(MockLikelihood(mean=mean, covariance=cov))
        best_f = torch.zeros(2, device=device)
        module = ProbabilityOfImprovement(model=mm, best_f=best_f)
        X = torch.empty(2, 2, 1, device=device)  # dummy Tensor for shape
        pi = module(X)
        pi_expected = torch.tensor([[0.5, 0.75], [0.25, 0.9]], device=device)
        self.assertTrue(approx_equal(pi, pi_expected))

    def test_probability_of_improvement_module_batch_cuda(self, cuda=False):
        if torch.cuda.is_available():
            self.test_probability_of_improvement_module_batch(cuda=True)

    def test_upper_confidence_bound_module(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        mean = torch.tensor([0.0, 0.5], device=device)
        cov = torch.tensor([[1.0, 0.0], [0.0, 4.0]], device=device)
        mm = MockModel(MockLikelihood(mean=mean, covariance=cov))
        module = UpperConfidenceBound(model=mm, beta=1.0)
        X = torch.empty(2, 1, device=device)  # dummy Tensor for shape
        ucb = module(X)
        ucb_expected = torch.tensor([1.0, 2.5], device=device)
        self.assertTrue(approx_equal(ucb, ucb_expected))

    def test_upper_confidence_bound_module_cuda(self, cuda=False):
        if torch.cuda.is_available():
            self.test_upper_confidence_bound_module(cuda=True)

    def test_upper_confidence_bound_module_batch(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        mean = torch.tensor([0.0, 0.5], device=device).repeat(2, 1)
        cov = torch.tensor([[1.0, 0.0], [0.0, 4.0]], device=device).repeat(2, 1, 1)
        mm = MockModel(MockLikelihood(mean=mean, covariance=cov))
        beta = torch.tensor([1.0, 2.0], device=device)
        module = UpperConfidenceBound(model=mm, beta=beta)
        X = torch.empty(2, 2, 1, device=device)  # dummy Tensor for shape
        ucb = module(X)
        ucb_expected = torch.tensor([[1.0, 2.5], [1.41421, 3.32842]], device=device)
        self.assertTrue(approx_equal(ucb, ucb_expected))

    def test_upper_confidence_bound_module_batch_cuda(self, cuda=False):
        if torch.cuda.is_available():
            self.test_upper_confidence_bound_module_batch(cuda=True)


if __name__ == "__main__":
    unittest.main()
