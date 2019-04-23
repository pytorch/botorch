#!/usr/bin/env python3

import unittest
from unittest import mock

import torch
from botorch.acquisition import monte_carlo, utils
from botorch.acquisition.objective import MCAcquisitionObjective
from botorch.acquisition.sampler import IIDNormalSampler, SobolQMCNormalSampler
from torch import Tensor

from ..mock import MockModel, MockPosterior


class DummyMCObjective(MCAcquisitionObjective):
    def forward(self, samples: Tensor) -> Tensor:
        return samples.sum(-1)


class TestGetAcquisitionFunction(unittest.TestCase):
    def setUp(self):
        self.model = mock.MagicMock()
        self.objective = DummyMCObjective()
        self.X_observed = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
        self.X_pending = torch.tensor([[1.0, 3.0, 4.0]])
        self.mc_samples = 250
        self.qmc = True
        self.seed = 1

    @mock.patch(f"{monte_carlo.__name__}.qExpectedImprovement")
    def test_GetQEI(self, mock_acqf):
        acqf = utils.get_acquisition_function(
            acquisition_function_name="qEI",
            model=self.model,
            objective=self.objective,
            X_observed=self.X_observed,
            X_pending=self.X_pending,
            mc_samples=self.mc_samples,
            seed=self.seed,
        )
        self.assertTrue(acqf == mock_acqf.return_value)
        best_f = self.objective(self.model.posterior(self.X_observed).mean).max().item()
        mock_acqf.assert_called_once_with(
            model=self.model, best_f=best_f, sampler=mock.ANY, objective=self.objective
        )
        args, kwargs = mock_acqf.call_args
        self.assertEqual(args, ())
        sampler = kwargs["sampler"]
        self.assertIsInstance(sampler, SobolQMCNormalSampler)
        self.assertEqual(sampler.sample_shape, torch.Size([self.mc_samples]))
        self.assertEqual(sampler.seed, 1)

    @mock.patch(f"{monte_carlo.__name__}.qProbabilityOfImprovement")
    def test_GetQPI(self, mock_acqf):
        # basic test
        acqf = utils.get_acquisition_function(
            acquisition_function_name="qPI",
            model=self.model,
            objective=self.objective,
            X_observed=self.X_observed,
            X_pending=self.X_pending,
            mc_samples=self.mc_samples,
            seed=self.seed,
        )
        self.assertTrue(acqf == mock_acqf.return_value)
        best_f = self.objective(self.model.posterior(self.X_observed).mean).max().item()
        mock_acqf.assert_called_once_with(
            model=self.model,
            best_f=best_f,
            sampler=mock.ANY,
            objective=self.objective,
            tau=1e-3,
        )
        args, kwargs = mock_acqf.call_args
        self.assertEqual(args, ())
        sampler = kwargs["sampler"]
        self.assertIsInstance(sampler, SobolQMCNormalSampler)
        self.assertEqual(sampler.sample_shape, torch.Size([self.mc_samples]))
        self.assertEqual(sampler.seed, 1)
        # test with different tau, non-qmc
        acqf = utils.get_acquisition_function(
            acquisition_function_name="qPI",
            model=self.model,
            objective=self.objective,
            X_observed=self.X_observed,
            X_pending=self.X_pending,
            mc_samples=self.mc_samples,
            qmc=False,
            seed=2,
            tau=1.0,
        )
        self.assertTrue(mock_acqf.call_count, 2)
        args, kwargs = mock_acqf.call_args
        self.assertEqual(args, ())
        self.assertEqual(kwargs["tau"], 1.0)
        sampler = kwargs["sampler"]
        self.assertIsInstance(sampler, IIDNormalSampler)
        self.assertEqual(sampler.sample_shape, torch.Size([self.mc_samples]))
        self.assertEqual(sampler.seed, 2)

    @mock.patch(f"{monte_carlo.__name__}.qNoisyExpectedImprovement")
    def test_GetQNEI(self, mock_acqf):
        # basic test
        acqf = utils.get_acquisition_function(
            acquisition_function_name="qNEI",
            model=self.model,
            objective=self.objective,
            X_observed=self.X_observed,
            X_pending=self.X_pending,
            mc_samples=self.mc_samples,
            seed=self.seed,
        )
        self.assertTrue(acqf == mock_acqf.return_value)
        self.assertTrue(mock_acqf.call_count, 1)
        args, kwargs = mock_acqf.call_args
        self.assertEqual(args, ())
        X_baseline_exp = torch.cat([self.X_observed, self.X_pending], dim=-2)
        self.assertTrue(torch.equal(kwargs["X_baseline"], X_baseline_exp))
        sampler = kwargs["sampler"]
        self.assertIsInstance(sampler, SobolQMCNormalSampler)
        self.assertEqual(sampler.sample_shape, torch.Size([self.mc_samples]))
        self.assertEqual(sampler.seed, 1)
        # test with non-qmc, no X_pending
        acqf = utils.get_acquisition_function(
            acquisition_function_name="qNEI",
            model=self.model,
            objective=self.objective,
            X_observed=self.X_observed,
            X_pending=None,
            mc_samples=self.mc_samples,
            qmc=False,
            seed=2,
        )
        self.assertTrue(mock_acqf.call_count, 2)
        args, kwargs = mock_acqf.call_args
        self.assertEqual(args, ())
        self.assertTrue(torch.equal(kwargs["X_baseline"], self.X_observed))
        sampler = kwargs["sampler"]
        self.assertIsInstance(sampler, IIDNormalSampler)
        self.assertEqual(sampler.sample_shape, torch.Size([self.mc_samples]))
        self.assertEqual(sampler.seed, 2)
        self.assertTrue(torch.equal(kwargs["X_baseline"], self.X_observed))

    @mock.patch(f"{monte_carlo.__name__}.qSimpleRegret")
    def test_GetQSR(self, mock_acqf):
        # basic test
        acqf = utils.get_acquisition_function(
            acquisition_function_name="qSR",
            model=self.model,
            objective=self.objective,
            X_observed=self.X_observed,
            X_pending=self.X_pending,
            mc_samples=self.mc_samples,
            seed=self.seed,
        )
        self.assertTrue(acqf == mock_acqf.return_value)
        mock_acqf.assert_called_once_with(
            model=self.model, sampler=mock.ANY, objective=self.objective
        )
        args, kwargs = mock_acqf.call_args
        self.assertEqual(args, ())
        sampler = kwargs["sampler"]
        self.assertIsInstance(sampler, SobolQMCNormalSampler)
        self.assertEqual(sampler.sample_shape, torch.Size([self.mc_samples]))
        self.assertEqual(sampler.seed, 1)
        # test with non-qmc
        acqf = utils.get_acquisition_function(
            acquisition_function_name="qSR",
            model=self.model,
            objective=self.objective,
            X_observed=self.X_observed,
            X_pending=self.X_pending,
            mc_samples=self.mc_samples,
            qmc=False,
            seed=2,
        )
        self.assertTrue(mock_acqf.call_count, 2)
        args, kwargs = mock_acqf.call_args
        self.assertEqual(args, ())
        sampler = kwargs["sampler"]
        self.assertIsInstance(sampler, IIDNormalSampler)
        self.assertEqual(sampler.sample_shape, torch.Size([self.mc_samples]))
        self.assertEqual(sampler.seed, 2)

    @mock.patch(f"{monte_carlo.__name__}.qUpperConfidenceBound")
    def test_GetQUCB(self, mock_acqf):
        # make sure beta is specified
        with self.assertRaises(ValueError):
            acqf = utils.get_acquisition_function(
                acquisition_function_name="qUCB",
                model=self.model,
                objective=self.objective,
                X_observed=self.X_observed,
                X_pending=self.X_pending,
                mc_samples=self.mc_samples,
                seed=self.seed,
            )
        acqf = utils.get_acquisition_function(
            acquisition_function_name="qUCB",
            model=self.model,
            objective=self.objective,
            X_observed=self.X_observed,
            X_pending=self.X_pending,
            mc_samples=self.mc_samples,
            seed=self.seed,
            beta=0.3,
        )
        self.assertTrue(acqf == mock_acqf.return_value)
        mock_acqf.assert_called_once_with(
            model=self.model, beta=0.3, sampler=mock.ANY, objective=self.objective
        )
        args, kwargs = mock_acqf.call_args
        self.assertEqual(args, ())
        sampler = kwargs["sampler"]
        self.assertIsInstance(sampler, SobolQMCNormalSampler)
        self.assertEqual(sampler.sample_shape, torch.Size([self.mc_samples]))
        self.assertEqual(sampler.seed, 1)
        # test with different tau, non-qmc
        acqf = utils.get_acquisition_function(
            acquisition_function_name="qUCB",
            model=self.model,
            objective=self.objective,
            X_observed=self.X_observed,
            X_pending=self.X_pending,
            mc_samples=self.mc_samples,
            qmc=False,
            seed=2,
            beta=0.2,
        )
        self.assertTrue(mock_acqf.call_count, 2)
        args, kwargs = mock_acqf.call_args
        self.assertEqual(args, ())
        self.assertEqual(kwargs["beta"], 0.2)
        sampler = kwargs["sampler"]
        self.assertIsInstance(sampler, IIDNormalSampler)
        self.assertEqual(sampler.sample_shape, torch.Size([self.mc_samples]))
        self.assertEqual(sampler.seed, 2)

    def test_GetUnknownAcquisitionFunction(self):
        with self.assertRaises(NotImplementedError):
            utils.get_acquisition_function(
                acquisition_function_name="foo",
                model=self.model,
                objective=self.objective,
                X_observed=self.X_observed,
                X_pending=self.X_pending,
                mc_samples=self.mc_samples,
                seed=self.seed,
            )


class TestGetInfeasibleCost(unittest.TestCase):
    def test_get_infeasible_cost(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            X = torch.zeros(5, 1, device=device, dtype=dtype)
            means = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=device, dtype=dtype)
            variances = torch.tensor(
                [0.09, 0.25, 0.36, 0.25, 0.09], device=device, dtype=dtype
            )
            mm = MockModel(MockPosterior(mean=means, variance=variances))
            # means - 6 * std = [-0.8, -1, -0.6, 1, 3.2]. After applying the
            # objective, the minimum becomes -6.0, so 6.0 should be returned.
            M = utils.get_infeasible_cost(
                X=X, model=mm, objective=lambda Y: Y.squeeze(-1) - 5.0
            )
            self.assertEqual(M, 6.0)

    def test_get_infeasible_cost_cuda(self):
        if torch.cuda.is_available():
            self.test_get_infeasible_cost(cuda=True)
