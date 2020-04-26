#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
from botorch.acquisition.analytic import (
    AnalyticAcquisitionFunction,
    ConstrainedExpectedImprovement,
    ExpectedImprovement,
    NoisyExpectedImprovement,
    PosteriorMean,
    ProbabilityOfImprovement,
    UpperConfidenceBound,
)
from botorch.acquisition.objective import IdentityMCObjective, ScalarizedObjective
from botorch.exceptions import UnsupportedError
from botorch.models import FixedNoiseGP, SingleTaskGP
from botorch.posteriors import GPyTorchPosterior
from botorch.utils.testing import BotorchTestCase, MockModel, MockPosterior
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal


NEI_NOISE = [
    [-0.099],
    [-0.004],
    [0.227],
    [-0.182],
    [0.018],
    [0.334],
    [-0.270],
    [0.156],
    [-0.237],
    [0.052],
]


class DummyAnalyticAcquisitionFunction(AnalyticAcquisitionFunction):
    def forward(self, X):
        pass


class TestAnalyticAcquisitionFunction(BotorchTestCase):
    def test_abstract_raises(self):
        with self.assertRaises(TypeError):
            AnalyticAcquisitionFunction()
        # raise if model is multi-output, but no objective is given
        mean = torch.zeros(1, 2)
        variance = torch.ones(1, 2)
        mm = MockModel(MockPosterior(mean=mean, variance=variance))
        with self.assertRaises(UnsupportedError):
            DummyAnalyticAcquisitionFunction(model=mm)


class TestExpectedImprovement(BotorchTestCase):
    def test_expected_improvement(self):
        for dtype in (torch.float, torch.double):
            mean = torch.tensor([[-0.5]], device=self.device, dtype=dtype)
            variance = torch.ones(1, 1, device=self.device, dtype=dtype)
            mm = MockModel(MockPosterior(mean=mean, variance=variance))

            # basic test
            module = ExpectedImprovement(model=mm, best_f=0.0)
            X = torch.empty(1, 1, device=self.device, dtype=dtype)  # dummy
            ei = module(X)
            ei_expected = torch.tensor(0.19780, device=self.device, dtype=dtype)
            self.assertTrue(torch.allclose(ei, ei_expected, atol=1e-4))

            # test maximize
            module = ExpectedImprovement(model=mm, best_f=0.0, maximize=False)
            X = torch.empty(1, 1, device=self.device, dtype=dtype)  # dummy
            ei = module(X)
            ei_expected = torch.tensor(0.6978, device=self.device, dtype=dtype)
            self.assertTrue(torch.allclose(ei, ei_expected, atol=1e-4))
            with self.assertRaises(UnsupportedError):
                module.set_X_pending(None)

            # test objective (single-output)
            mean = torch.tensor([0.5], device=self.device, dtype=dtype)
            covar = torch.tensor([[0.16]], device=self.device, dtype=dtype)
            mvn = MultivariateNormal(mean, covar)
            p = GPyTorchPosterior(mvn)
            mm = MockModel(p)
            weights = torch.tensor([0.5], device=self.device, dtype=dtype)
            obj = ScalarizedObjective(weights)
            ei = ExpectedImprovement(model=mm, best_f=0.0, objective=obj)
            X = torch.rand(1, 2, device=self.device, dtype=dtype)
            ei_expected = torch.tensor(0.2601, device=self.device, dtype=dtype)
            torch.allclose(ei(X), ei_expected, atol=1e-4)

            # test objective (multi-output)
            mean = torch.tensor([[-0.25, 0.5]], device=self.device, dtype=dtype)
            covar = torch.tensor(
                [[[0.5, 0.125], [0.125, 0.5]]], device=self.device, dtype=dtype
            )
            mvn = MultitaskMultivariateNormal(mean, covar)
            p = GPyTorchPosterior(mvn)
            mm = MockModel(p)
            weights = torch.tensor([2.0, 1.0], device=self.device, dtype=dtype)
            obj = ScalarizedObjective(weights)
            ei = ExpectedImprovement(model=mm, best_f=0.0, objective=obj)
            X = torch.rand(1, 2, device=self.device, dtype=dtype)
            ei_expected = torch.tensor(0.6910, device=self.device, dtype=dtype)
            torch.allclose(ei(X), ei_expected, atol=1e-4)

    def test_expected_improvement_batch(self):
        for dtype in (torch.float, torch.double):
            mean = torch.tensor([-0.5, 0.0, 0.5], device=self.device, dtype=dtype).view(
                3, 1, 1
            )
            variance = torch.ones(3, 1, 1, device=self.device, dtype=dtype)
            mm = MockModel(MockPosterior(mean=mean, variance=variance))
            module = ExpectedImprovement(model=mm, best_f=0.0)
            X = torch.empty(3, 1, 1, device=self.device, dtype=dtype)  # dummy
            ei = module(X)
            ei_expected = torch.tensor(
                [0.19780, 0.39894, 0.69780], device=self.device, dtype=dtype
            )
            self.assertTrue(torch.allclose(ei, ei_expected, atol=1e-4))
            # check for proper error if multi-output model
            mean2 = torch.rand(3, 1, 2, device=self.device, dtype=dtype)
            variance2 = torch.rand(3, 1, 2, device=self.device, dtype=dtype)
            mm2 = MockModel(MockPosterior(mean=mean2, variance=variance2))
            with self.assertRaises(UnsupportedError):
                ExpectedImprovement(model=mm2, best_f=0.0)

            # test objective (single-output)
            mean = torch.tensor([[[0.5]], [[0.25]]], device=self.device, dtype=dtype)
            covar = torch.tensor(
                [[[[0.16]]], [[[0.125]]]], device=self.device, dtype=dtype
            )
            mvn = MultivariateNormal(mean, covar)
            p = GPyTorchPosterior(mvn)
            mm = MockModel(p)
            weights = torch.tensor([0.5], device=self.device, dtype=dtype)
            obj = ScalarizedObjective(weights)
            ei = ExpectedImprovement(model=mm, best_f=0.0, objective=obj)
            X = torch.rand(2, 1, 2, device=self.device, dtype=dtype)
            ei_expected = torch.tensor(
                [[0.2601], [0.1500]], device=self.device, dtype=dtype
            )
            torch.allclose(ei(X), ei_expected, atol=1e-4)

            # test objective (multi-output)
            mean = torch.tensor(
                [[[-0.25, 0.5]], [[0.2, -0.1]]], device=self.device, dtype=dtype
            )
            covar = torch.tensor(
                [[[0.5, 0.125], [0.125, 0.5]], [[0.25, -0.1], [-0.1, 0.25]]],
                device=self.device,
                dtype=dtype,
            )
            mvn = MultitaskMultivariateNormal(mean, covar)
            p = GPyTorchPosterior(mvn)
            mm = MockModel(p)
            weights = torch.tensor([2.0, 1.0], device=self.device, dtype=dtype)
            obj = ScalarizedObjective(weights)
            ei = ExpectedImprovement(model=mm, best_f=0.0, objective=obj)
            X = torch.rand(2, 1, 2, device=self.device, dtype=dtype)
            ei_expected = torch.tensor(
                [0.6910, 0.5371], device=self.device, dtype=dtype
            )
            torch.allclose(ei(X), ei_expected, atol=1e-4)

        # test bad objective class
        with self.assertRaises(UnsupportedError):
            ExpectedImprovement(model=mm, best_f=0.0, objective=IdentityMCObjective())


class TestPosteriorMean(BotorchTestCase):
    def test_posterior_mean(self):
        for dtype in (torch.float, torch.double):
            mean = torch.tensor([[0.25]], device=self.device, dtype=dtype)
            mm = MockModel(MockPosterior(mean=mean))
            module = PosteriorMean(model=mm)
            X = torch.empty(1, 1, device=self.device, dtype=dtype)
            pm = module(X)
            self.assertTrue(torch.equal(pm, mean.view(-1)))
            # check for proper error if multi-output model
            mean2 = torch.rand(1, 2, device=self.device, dtype=dtype)
            mm2 = MockModel(MockPosterior(mean=mean2))
            with self.assertRaises(UnsupportedError):
                PosteriorMean(model=mm2)

    def test_posterior_mean_batch(self):
        for dtype in (torch.float, torch.double):
            mean = torch.tensor([-0.5, 0.0, 0.5], device=self.device, dtype=dtype).view(
                3, 1, 1
            )
            mm = MockModel(MockPosterior(mean=mean))
            module = PosteriorMean(model=mm)
            X = torch.empty(3, 1, 1, device=self.device, dtype=dtype)
            pm = module(X)
            self.assertTrue(torch.equal(pm, mean.view(-1)))
            # check for proper error if multi-output model
            mean2 = torch.rand(3, 1, 2, device=self.device, dtype=dtype)
            mm2 = MockModel(MockPosterior(mean=mean2))
            with self.assertRaises(UnsupportedError):
                PosteriorMean(model=mm2)


class TestProbabilityOfImprovement(BotorchTestCase):
    def test_probability_of_improvement(self):
        for dtype in (torch.float, torch.double):
            mean = torch.tensor([0.0], device=self.device, dtype=dtype).view(1, 1)
            variance = torch.ones(1, 1, device=self.device, dtype=dtype)
            mm = MockModel(MockPosterior(mean=mean, variance=variance))

            module = ProbabilityOfImprovement(model=mm, best_f=1.96)
            X = torch.zeros(1, 1, device=self.device, dtype=dtype)
            pi = module(X)
            pi_expected = torch.tensor(0.0250, device=self.device, dtype=dtype)
            self.assertTrue(torch.allclose(pi, pi_expected, atol=1e-4))

            module = ProbabilityOfImprovement(model=mm, best_f=1.96, maximize=False)
            X = torch.zeros(1, 1, device=self.device, dtype=dtype)
            pi = module(X)
            pi_expected = torch.tensor(0.9750, device=self.device, dtype=dtype)
            self.assertTrue(torch.allclose(pi, pi_expected, atol=1e-4))

            # check for proper error if multi-output model
            mean2 = torch.rand(1, 2, device=self.device, dtype=dtype)
            variance2 = torch.ones_like(mean2)
            mm2 = MockModel(MockPosterior(mean=mean2, variance=variance2))
            with self.assertRaises(UnsupportedError):
                ProbabilityOfImprovement(model=mm2, best_f=0.0)

    def test_probability_of_improvement_batch(self):
        for dtype in (torch.float, torch.double):
            mean = torch.tensor([0.0, 0.67449], device=self.device, dtype=dtype).view(
                2, 1, 1
            )
            variance = torch.ones_like(mean)
            mm = MockModel(MockPosterior(mean=mean, variance=variance))
            module = ProbabilityOfImprovement(model=mm, best_f=0.0)
            X = torch.zeros(2, 1, 1, device=self.device, dtype=dtype)
            pi = module(X)
            pi_expected = torch.tensor([0.5, 0.75], device=self.device, dtype=dtype)
            self.assertTrue(torch.allclose(pi, pi_expected, atol=1e-4))
            # check for proper error if multi-output model
            mean2 = torch.rand(3, 1, 2, device=self.device, dtype=dtype)
            variance2 = torch.ones_like(mean2)
            mm2 = MockModel(MockPosterior(mean=mean2, variance=variance2))
            with self.assertRaises(UnsupportedError):
                ProbabilityOfImprovement(model=mm2, best_f=0.0)


class TestUpperConfidenceBound(BotorchTestCase):
    def test_upper_confidence_bound(self):
        for dtype in (torch.float, torch.double):
            mean = torch.tensor([[0.0]], device=self.device, dtype=dtype)
            variance = torch.tensor([[1.0]], device=self.device, dtype=dtype)
            mm = MockModel(MockPosterior(mean=mean, variance=variance))

            module = UpperConfidenceBound(model=mm, beta=1.0)
            X = torch.zeros(1, 1, device=self.device, dtype=dtype)
            ucb = module(X)
            ucb_expected = torch.tensor([1.0], device=self.device, dtype=dtype)
            self.assertTrue(torch.allclose(ucb, ucb_expected, atol=1e-4))

            module = UpperConfidenceBound(model=mm, beta=1.0, maximize=False)
            X = torch.zeros(1, 1, device=self.device, dtype=dtype)
            ucb = module(X)
            ucb_expected = torch.tensor([-1.0], device=self.device, dtype=dtype)
            self.assertTrue(torch.allclose(ucb, ucb_expected, atol=1e-4))

            # check for proper error if multi-output model
            mean2 = torch.rand(1, 2, device=self.device, dtype=dtype)
            variance2 = torch.rand(1, 2, device=self.device, dtype=dtype)
            mm2 = MockModel(MockPosterior(mean=mean2, variance=variance2))
            with self.assertRaises(UnsupportedError):
                UpperConfidenceBound(model=mm2, beta=1.0)

    def test_upper_confidence_bound_batch(self):
        for dtype in (torch.float, torch.double):
            mean = torch.tensor([0.0, 0.5], device=self.device, dtype=dtype).view(
                2, 1, 1
            )
            variance = torch.tensor([1.0, 4.0], device=self.device, dtype=dtype).view(
                2, 1, 1
            )
            mm = MockModel(MockPosterior(mean=mean, variance=variance))
            module = UpperConfidenceBound(model=mm, beta=1.0)
            X = torch.zeros(2, 1, 1, device=self.device, dtype=dtype)
            ucb = module(X)
            ucb_expected = torch.tensor([1.0, 2.5], device=self.device, dtype=dtype)
            self.assertTrue(torch.allclose(ucb, ucb_expected, atol=1e-4))
            # check for proper error if multi-output model
            mean2 = torch.rand(3, 1, 2, device=self.device, dtype=dtype)
            variance2 = torch.rand(3, 1, 2, device=self.device, dtype=dtype)
            mm2 = MockModel(MockPosterior(mean=mean2, variance=variance2))
            with self.assertRaises(UnsupportedError):
                UpperConfidenceBound(model=mm2, beta=1.0)


class TestConstrainedExpectedImprovement(BotorchTestCase):
    def test_constrained_expected_improvement(self):
        for dtype in (torch.float, torch.double):
            # one constraint
            mean = torch.tensor(
                [[-0.5, 0.0]], device=self.device, dtype=dtype
            ).unsqueeze(dim=-2)
            variance = torch.ones(1, 2, device=self.device, dtype=dtype).unsqueeze(
                dim=-2
            )
            mm = MockModel(MockPosterior(mean=mean, variance=variance))
            module = ConstrainedExpectedImprovement(
                model=mm, best_f=0.0, objective_index=0, constraints={1: [None, 0]}
            )
            X = torch.empty(1, 1, device=self.device, dtype=dtype)  # dummy
            ei = module(X)
            ei_expected_unconstrained = torch.tensor(
                0.19780, device=self.device, dtype=dtype
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

            # check that error raised if constraint lower > upper
            with self.assertRaises(ValueError):
                module = ConstrainedExpectedImprovement(
                    model=mm, best_f=0.0, objective_index=0, constraints={0: [1, 0]}
                )

            # three constraints
            N = torch.distributions.Normal(loc=0.0, scale=1.0)
            a = N.icdf(torch.tensor(0.75))  # get a so that P(-a <= N <= a) = 0.5
            mean = torch.tensor(
                [[-0.5, 0.0, 5.0, 0.0]], device=self.device, dtype=dtype
            ).unsqueeze(dim=-2)
            variance = torch.ones(1, 4, device=self.device, dtype=dtype).unsqueeze(
                dim=-2
            )
            mm = MockModel(MockPosterior(mean=mean, variance=variance))
            module = ConstrainedExpectedImprovement(
                model=mm,
                best_f=0.0,
                objective_index=0,
                constraints={1: [None, 0], 2: [5.0, None], 3: [-a, a]},
            )
            X = torch.empty(1, 1, device=self.device, dtype=dtype)  # dummy
            ei = module(X)
            ei_expected_unconstrained = torch.tensor(
                0.19780, device=self.device, dtype=dtype
            )
            ei_expected = ei_expected_unconstrained * 0.5 * 0.5 * 0.5
            self.assertTrue(torch.allclose(ei, ei_expected, atol=1e-4))
            # test maximize
            module_min = ConstrainedExpectedImprovement(
                model=mm,
                best_f=0.0,
                objective_index=0,
                constraints={1: [None, 0]},
                maximize=False,
            )
            ei_min = module_min(X)
            ei_expected_unconstrained_min = torch.tensor(
                0.6978, device=self.device, dtype=dtype
            )
            ei_expected_min = ei_expected_unconstrained_min * 0.5
            self.assertTrue(torch.allclose(ei_min, ei_expected_min, atol=1e-4))
            # test invalid onstraints
            with self.assertRaises(ValueError):
                ConstrainedExpectedImprovement(
                    model=mm,
                    best_f=0.0,
                    objective_index=0,
                    constraints={1: [1.0, -1.0]},
                )

    def test_constrained_expected_improvement_batch(self):
        for dtype in (torch.float, torch.double):
            mean = torch.tensor(
                [[-0.5, 0.0, 5.0, 0.0], [0.0, 0.0, 5.0, 0.0], [0.5, 0.0, 5.0, 0.0]],
                device=self.device,
                dtype=dtype,
            ).unsqueeze(dim=-2)
            variance = torch.ones(3, 4, device=self.device, dtype=dtype).unsqueeze(
                dim=-2
            )
            N = torch.distributions.Normal(loc=0.0, scale=1.0)
            a = N.icdf(torch.tensor(0.75))  # get a so that P(-a <= N <= a) = 0.5
            mm = MockModel(MockPosterior(mean=mean, variance=variance))
            module = ConstrainedExpectedImprovement(
                model=mm,
                best_f=0.0,
                objective_index=0,
                constraints={1: [None, 0], 2: [5.0, None], 3: [-a, a]},
            )
            X = torch.empty(3, 1, 1, device=self.device, dtype=dtype)  # dummy
            ei = module(X)
            self.assertTrue(ei.shape == torch.Size([3]))
            ei_expected_unconstrained = torch.tensor(
                [0.19780, 0.39894, 0.69780], device=self.device, dtype=dtype
            )
            ei_expected = ei_expected_unconstrained * 0.5 * 0.5 * 0.5
            self.assertTrue(torch.allclose(ei, ei_expected, atol=1e-4))


class TestNoisyExpectedImprovement(BotorchTestCase):
    def _get_model(self, dtype=torch.float):
        state_dict = {
            "mean_module.constant": torch.tensor([-0.0066]),
            "covar_module.raw_outputscale": torch.tensor(1.0143),
            "covar_module.base_kernel.raw_lengthscale": torch.tensor([[-0.99]]),
            "covar_module.base_kernel.lengthscale_prior.concentration": torch.tensor(
                3.0
            ),
            "covar_module.base_kernel.lengthscale_prior.rate": torch.tensor(6.0),
            "covar_module.outputscale_prior.concentration": torch.tensor(2.0),
            "covar_module.outputscale_prior.rate": torch.tensor(0.1500),
        }
        train_x = torch.linspace(0, 1, 10, device=self.device, dtype=dtype).unsqueeze(
            -1
        )
        train_y = torch.sin(train_x * (2 * math.pi))
        noise = torch.tensor(NEI_NOISE, device=self.device, dtype=dtype)
        train_y += noise
        train_yvar = torch.full_like(train_y, 0.25 ** 2)
        model = FixedNoiseGP(train_X=train_x, train_Y=train_y, train_Yvar=train_yvar)
        model.load_state_dict(state_dict)
        model.to(train_x)
        model.eval()
        return model

    def test_noisy_expected_improvement(self):
        for dtype in (torch.float, torch.double):
            model = self._get_model(dtype=dtype)
            X_observed = model.train_inputs[0]
            nEI = NoisyExpectedImprovement(model, X_observed, num_fantasies=5)
            X_test = torch.tensor(
                [[[0.25]], [[0.75]]],
                device=X_observed.device,
                dtype=dtype,
                requires_grad=True,
            )
            val = nEI(X_test)
            # test basics
            self.assertEqual(val.dtype, dtype)
            self.assertEqual(val.device.type, X_observed.device.type)
            self.assertEqual(val.shape, torch.Size([2]))
            # test values
            self.assertGreater(val[0].item(), 1e-4)
            self.assertLess(val[1].item(), 1e-6)
            # test gradient
            val.sum().backward()
            self.assertGreater(X_test.grad[0].abs().item(), 1e-5)
            # test without gradient
            with torch.no_grad():
                nEI(X_test)
            # test non-FixedNoiseGP model
            other_model = SingleTaskGP(X_observed, model.train_targets.unsqueeze(-1))
            with self.assertRaises(UnsupportedError):
                NoisyExpectedImprovement(other_model, X_observed, num_fantasies=5)
            # Test with minimize
            nEI = NoisyExpectedImprovement(
                model, X_observed, num_fantasies=5, maximize=False
            )
