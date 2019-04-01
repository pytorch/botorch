#! /usr/bin/env python3

import unittest

import torch
from botorch.acquisition.analytic import (
    AnalyticAcquisitionFunction,
    ConstrainedExpectedImprovement,
    ExpectedImprovement,
    PosteriorMean,
    ProbabilityOfImprovement,
    UpperConfidenceBound,
)
from botorch.exceptions import UnsupportedError

from ..mock import MockModel, MockPosterior


class TestAnalyticAcquisitionFunction(unittest.TestCase):
    def test_abstract_raises(self):
        with self.assertRaises(TypeError):
            AnalyticAcquisitionFunction()


class TestExpectedImprovement(unittest.TestCase):
    def test_expected_improvement(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            mean = torch.tensor([[-0.5]], device=device, dtype=dtype)
            variance = torch.ones(1, 1, device=device, dtype=dtype)
            mm = MockModel(MockPosterior(mean=mean, variance=variance))
            module = ExpectedImprovement(model=mm, best_f=0.0)
            X = torch.empty(1, 1, device=device, dtype=dtype)  # dummy
            ei = module(X)
            ei_expected = torch.tensor(0.19780, device=device, dtype=dtype)
            self.assertTrue(torch.allclose(ei, ei_expected, atol=1e-4))

    def test_expected_improvement_cuda(self, cuda=False):
        if torch.cuda.is_available():
            self.test_expected_improvement(cuda=True)

    def test_expected_improvement_batch(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            mean = torch.tensor([[-0.5], [0.0], [0.5]], device=device, dtype=dtype)
            variance = torch.ones(3, 1, device=device, dtype=dtype)
            mm = MockModel(MockPosterior(mean=mean, variance=variance))
            module = ExpectedImprovement(model=mm, best_f=0.0)
            X = torch.empty(3, 1, device=device, dtype=dtype)  # dummy
            ei = module(X)
            ei_expected = torch.tensor(
                [0.19780, 0.39894, 0.69780], device=device, dtype=dtype
            )
            self.assertTrue(torch.allclose(ei, ei_expected, atol=1e-4))
            # check for proper error if multi-output model
            mean2 = torch.rand(3, 2, device=device, dtype=dtype)
            variance2 = torch.rand(3, 2, device=device, dtype=dtype)
            mm2 = MockModel(MockPosterior(mean=mean2, variance=variance2))
            module2 = ExpectedImprovement(model=mm2, best_f=0.0)
            with self.assertRaises(UnsupportedError):
                module2(X)

    def test_expected_improvement_batch_cuda(self, cuda=False):
        if torch.cuda.is_available():
            self.test_expected_improvement_batch(cuda=True)


class TestPosteriorMean(unittest.TestCase):
    def test_posterior_mean(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            mean = torch.tensor([[0.25]], device=device, dtype=dtype)
            mm = MockModel(MockPosterior(mean=mean))
            module = PosteriorMean(model=mm)
            X = torch.empty(1, device=device, dtype=dtype)
            pm = module(X)
            self.assertTrue(torch.equal(pm, mean[0, 0]))
            # check for proper error if multi-output model
            mean2 = torch.rand(1, 2, device=device, dtype=dtype)
            mm2 = MockModel(MockPosterior(mean=mean2))
            module2 = PosteriorMean(model=mm2)
            with self.assertRaises(UnsupportedError):
                module2(X)

    def test_posterior_mean_cuda(self, cuda=False):
        if torch.cuda.is_available():
            self.test_posterior_mean(cuda=True)

    def test_posterior_mean_batch(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            mean = torch.tensor([[-0.5], [0.0], [0.5]], device=device, dtype=dtype)
            mm = MockModel(MockPosterior(mean=mean))
            module = PosteriorMean(model=mm)
            X = torch.empty(3, 1, device=device, dtype=dtype)
            pm = module(X)
            self.assertTrue(torch.equal(pm, mean.view(-1)))
            # check for proper error if multi-output model
            mean2 = torch.rand(3, 2, device=device, dtype=dtype)
            mm2 = MockModel(MockPosterior(mean=mean2))
            module2 = PosteriorMean(model=mm2)
            with self.assertRaises(UnsupportedError):
                module2(X)

    def test_posterior_mean_batch_cuda(self, cuda=False):
        if torch.cuda.is_available():
            self.test_posterior_mean_batch(cuda=True)


class TestProbabilityOfImprovement(unittest.TestCase):
    def test_probability_of_improvement(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            mean = torch.tensor([[0.0]], device=device, dtype=dtype)
            variance = torch.ones(1, 1, device=device, dtype=dtype)
            mm = MockModel(MockPosterior(mean=mean, variance=variance))
            module = ProbabilityOfImprovement(model=mm, best_f=0.0)
            X = torch.zeros(1, device=device, dtype=dtype)
            pi = module(X)
            pi_expected = torch.tensor(0.5, device=device, dtype=dtype)
            self.assertTrue(torch.allclose(pi, pi_expected, atol=1e-4))
            # check for proper error if multi-output model
            mean2 = torch.rand(1, 2, device=device, dtype=dtype)
            variance2 = torch.ones_like(mean2)
            mm2 = MockModel(MockPosterior(mean=mean2, variance=variance2))
            module2 = ProbabilityOfImprovement(model=mm2, best_f=0.0)
            with self.assertRaises(UnsupportedError):
                module2(X)

    def test_probability_of_improvement_cuda(self, cuda=False):
        if torch.cuda.is_available():
            self.test_probability_of_improvement(cuda=True)

    def test_probability_of_improvement_batch(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            mean = torch.tensor([[0.0], [0.67449]], device=device, dtype=dtype)
            variance = torch.ones_like(mean)
            mm = MockModel(MockPosterior(mean=mean, variance=variance))
            module = ProbabilityOfImprovement(model=mm, best_f=0.0)
            X = torch.zeros(2, 1, device=device, dtype=dtype)
            pi = module(X)
            pi_expected = torch.tensor([0.5, 0.75], device=device, dtype=dtype)
            self.assertTrue(torch.allclose(pi, pi_expected, atol=1e-4))
            # check for proper error if multi-output model
            mean2 = torch.rand(3, 2, device=device, dtype=dtype)
            variance2 = torch.ones_like(mean2)
            mm2 = MockModel(MockPosterior(mean=mean2, variance=variance2))
            module2 = ProbabilityOfImprovement(model=mm2, best_f=0.0)
            with self.assertRaises(UnsupportedError):
                module2(X)

    def test_probability_of_improvement_batch_cuda(self, cuda=False):
        if torch.cuda.is_available():
            self.test_probability_of_improvement_batch(cuda=True)


class TestUpperConfidenceBound(unittest.TestCase):
    def test_upper_confidence_bound(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            mean = torch.tensor([[0.0]], device=device, dtype=dtype)
            variance = torch.tensor([[1.0]], device=device, dtype=dtype)
            mm = MockModel(MockPosterior(mean=mean, variance=variance))
            module = UpperConfidenceBound(model=mm, beta=1.0)
            X = torch.zeros(1, 1, device=device, dtype=dtype)
            ucb = module(X)
            ucb_expected = torch.tensor([1.0], device=device, dtype=dtype)
            self.assertTrue(torch.allclose(ucb, ucb_expected, atol=1e-4))
            # check for proper error if multi-output model
            mean2 = torch.rand(1, 2, device=device, dtype=dtype)
            variance2 = torch.rand(1, 2, device=device, dtype=dtype)
            mm2 = MockModel(MockPosterior(mean=mean2, variance=variance2))
            module2 = UpperConfidenceBound(model=mm2, beta=1.0)
            with self.assertRaises(UnsupportedError):
                module2(X)

    def test_upper_confidence_bound_cuda(self, cuda=False):
        if torch.cuda.is_available():
            self.test_upper_confidence_bound(cuda=True)

    def test_upper_confidence_bound_batch(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            mean = torch.tensor([[0.0], [0.5]], device=device, dtype=dtype)
            variance = torch.tensor([[1.0], [4.0]], device=device, dtype=dtype)
            mm = MockModel(MockPosterior(mean=mean, variance=variance))
            module = UpperConfidenceBound(model=mm, beta=1.0)
            X = torch.zeros(2, 1, device=device, dtype=dtype)
            ucb = module(X)
            ucb_expected = torch.tensor([1.0, 2.5], device=device, dtype=dtype)
            self.assertTrue(torch.allclose(ucb, ucb_expected, atol=1e-4))
            # check for proper error if multi-output model
            mean2 = torch.rand(3, 2, device=device, dtype=dtype)
            variance2 = torch.rand(3, 2, device=device, dtype=dtype)
            mm2 = MockModel(MockPosterior(mean=mean2, variance=variance2))
            module2 = UpperConfidenceBound(model=mm2, beta=1.0)
            with self.assertRaises(UnsupportedError):
                module2(X)

    def test_upper_confidence_bound_batch_cuda(self, cuda=False):
        if torch.cuda.is_available():
            self.test_upper_confidence_bound_batch(cuda=True)


class TestConstrainedExpectedImprovement(unittest.TestCase):
    def test_constrained_expected_improvement(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            # one constraint
            mean = torch.tensor([[-0.5, 0.0]], device=device, dtype=dtype).unsqueeze(
                dim=-2
            )
            variance = torch.ones(1, 2, device=device, dtype=dtype).unsqueeze(dim=-2)
            mm = MockModel(MockPosterior(mean=mean, variance=variance))
            module = ConstrainedExpectedImprovement(
                model=mm, best_f=0.0, objective_index=0, constraints={1: [None, 0]}
            )
            X = torch.empty(1, 1, device=device, dtype=dtype)  # dummy
            ei = module(X)
            ei_expected_unconstrained = torch.tensor(
                0.19780, device=device, dtype=dtype
            )
            ei_expected = ei_expected_unconstrained * 0.5
            self.assertTrue(torch.allclose(ei, ei_expected, atol=1e-4))

            # check that error raised if no constraints
            with self.assertRaises(ValueError):
                module = ConstrainedExpectedImprovement(
                    model=mm, best_f=0.0, objective_index=0, constraints={}
                )

            # check that error raised if objective is a constraint
            with self.assertRaises(ValueError):
                module = ConstrainedExpectedImprovement(
                    model=mm, best_f=0.0, objective_index=0, constraints={0: [None, 0]}
                )

            # three constraints
            N = torch.distributions.Normal(loc=0.0, scale=1.0)
            a = N.icdf(torch.tensor(0.75))  # get a so that P(-a <= N <= a) = 0.5
            mean = torch.tensor(
                [[-0.5, 0.0, 5.0, 0.0]], device=device, dtype=dtype
            ).unsqueeze(dim=-2)
            variance = torch.ones(1, 4, device=device, dtype=dtype).unsqueeze(dim=-2)
            mm = MockModel(MockPosterior(mean=mean, variance=variance))
            module = ConstrainedExpectedImprovement(
                model=mm,
                best_f=0.0,
                objective_index=0,
                constraints={1: [None, 0], 2: [5.0, None], 3: [-a, a]},
            )
            X = torch.empty(1, 1, device=device, dtype=dtype)  # dummy
            ei = module(X)
            ei_expected_unconstrained = torch.tensor(
                0.19780, device=device, dtype=dtype
            )
            ei_expected = ei_expected_unconstrained * 0.5 * 0.5 * 0.5
            self.assertTrue(torch.allclose(ei, ei_expected, atol=1e-4))

    def test_constrained_expected_improvement_cuda(self, cuda=False):
        if torch.cuda.is_available():
            self.test_constrained_expected_improvement(cuda=True)

    def test_constrained_expected_improvement_batch(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            mean = torch.tensor(
                [[-0.5, 0.0, 5.0, 0.0], [0.0, 0.0, 5.0, 0.0], [0.5, 0.0, 5.0, 0.0]],
                device=device,
                dtype=dtype,
            ).unsqueeze(dim=-2)
            variance = torch.ones(3, 4, device=device, dtype=dtype).unsqueeze(dim=-2)
            N = torch.distributions.Normal(loc=0.0, scale=1.0)
            a = N.icdf(torch.tensor(0.75))  # get a so that P(-a <= N <= a) = 0.5
            mm = MockModel(MockPosterior(mean=mean, variance=variance))
            module = ConstrainedExpectedImprovement(
                model=mm,
                best_f=0.0,
                objective_index=0,
                constraints={1: [None, 0], 2: [5.0, None], 3: [-a, a]},
            )
            X = torch.empty(3, 1, device=device, dtype=dtype)  # dummy
            ei = module(X)
            ei_expected_unconstrained = torch.tensor(
                [0.19780, 0.39894, 0.69780], device=device, dtype=dtype
            )
            ei_expected = ei_expected_unconstrained * 0.5 * 0.5 * 0.5
            self.assertTrue(torch.allclose(ei, ei_expected, atol=1e-4))

    def test_constrained_expected_improvement_batch_cuda(self, cuda=False):
        if torch.cuda.is_available():
            self.test_constrained_expected_improvement_batch(cuda=True)
