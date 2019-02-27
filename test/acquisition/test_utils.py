#!/usr/bin/env python3

import unittest
from unittest import mock

import torch
from botorch.acquisition import utils


def dummy_objective(Y):
    return Y


def dummy_constraint(Y):
    return Y


class TestGetAcquisitionFunction(unittest.TestCase):
    def setUp(self):
        self.model = mock.MagicMock()
        self.X_observed = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
        self.X_pending = torch.tensor([[1.0, 3.0, 4.0]])
        self.seed = 1
        self.constraints = [dummy_constraint]
        self.infeasible_cost = 0.0

    @mock.patch(f"{utils.__name__}.qExpectedImprovement")
    def testGetQEI(self, mock_acquisition):
        acquisition_function = utils.get_acquisition_function(
            acquisition_function_name="qEI",
            model=self.model,
            X_observed=self.X_observed,
            objective=dummy_objective,
            constraints=self.constraints,
            infeasible_cost=self.infeasible_cost,
            X_pending=self.X_pending,
            seed=self.seed,
            acquisition_function_args=None,
        )
        self.assertTrue(acquisition_function == mock_acquisition.return_value)
        mock_acquisition.assert_called_once_with(
            model=self.model,
            best_f=self.model.posterior(self.X_observed).mean.max().item(),
            objective=dummy_objective,
            constraints=self.constraints,
            infeasible_cost=self.infeasible_cost,
            X_pending=self.X_pending,
            seed=self.seed,
        )

    @mock.patch(f"{utils.__name__}.qProbabilityOfImprovement")
    def testGetQPI(self, mock_acquisition):
        acquisition_function = utils.get_acquisition_function(
            acquisition_function_name="qPI",
            model=self.model,
            X_observed=self.X_observed,
            objective=dummy_objective,
            constraints=self.constraints,
            X_pending=self.X_pending,
            seed=self.seed,
            acquisition_function_args=None,
        )
        self.assertTrue(acquisition_function == mock_acquisition.return_value)
        mock_acquisition.assert_called_once_with(
            model=self.model,
            best_f=self.model.posterior(self.X_observed).mean.max().item(),
            objective=dummy_objective,
            constraints=self.constraints,
            X_pending=self.X_pending,
            seed=self.seed,
        )

    @mock.patch(f"{utils.__name__}.qNoisyExpectedImprovement")
    def testGetQNEI(self, mock_acquisition):
        acquisition_function = utils.get_acquisition_function(
            acquisition_function_name="qNEI",
            model=self.model,
            X_observed=self.X_observed,
            objective=dummy_objective,
            constraints=self.constraints,
            infeasible_cost=self.infeasible_cost,
            X_pending=self.X_pending,
            seed=self.seed,
            acquisition_function_args=None,
        )
        self.assertTrue(acquisition_function == mock_acquisition.return_value)
        mock_acquisition.assert_called_once_with(
            model=self.model,
            X_observed=self.X_observed,
            objective=dummy_objective,
            constraints=self.constraints,
            infeasible_cost=self.infeasible_cost,
            X_pending=self.X_pending,
            seed=self.seed,
        )

    @mock.patch(f"{utils.__name__}.qUpperConfidenceBound")
    def testGetQUCB(self, mock_acquisition):

        acquisition_function = utils.get_acquisition_function(
            acquisition_function_name="qUCB",
            model=self.model,
            X_observed=self.X_observed,
            objective=dummy_objective,
            constraints=self.constraints,
            X_pending=self.X_pending,
            seed=self.seed,
            acquisition_function_args={"beta": 2.0},
        )
        self.assertTrue(acquisition_function == mock_acquisition.return_value)
        mock_acquisition.assert_called_once_with(
            model=self.model, beta=2.0, X_pending=self.X_pending, seed=self.seed
        )

    @mock.patch(f"{utils.__name__}.qKnowledgeGradient")
    def testGetQKG(self, mock_acquisition):
        acquisition_function = utils.get_acquisition_function(
            acquisition_function_name="qKG",
            model=self.model,
            X_observed=self.X_observed,
            objective=dummy_objective,
            constraints=self.constraints,
            X_pending=self.X_pending,
            seed=self.seed,
        )
        self.assertTrue(acquisition_function == mock_acquisition.return_value)
        mock_acquisition.assert_called_once_with(
            model=self.model,
            X_observed=self.X_observed,
            objective=dummy_objective,
            constraints=self.constraints,
            X_pending=self.X_pending,
            seed=self.seed,
        )

    @mock.patch(f"{utils.__name__}.qKnowledgeGradientNoDiscretization")
    def testGetQKGNoDiscretization(self, mock_acquisition):
        acquisition_function = utils.get_acquisition_function(
            acquisition_function_name="qKGNoDiscretization",
            model=self.model,
            X_observed=self.X_observed,
            objective=dummy_objective,
            constraints=self.constraints,
            X_pending=self.X_pending,
            seed=self.seed,
        )
        self.assertTrue(acquisition_function == mock_acquisition.return_value)
        mock_acquisition.assert_called_once_with(
            model=self.model,
            objective=dummy_objective,
            constraints=self.constraints,
            X_pending=self.X_pending,
            seed=self.seed,
        )

    def testGetQUCBNoBeta(self):
        self.assertRaises(
            ValueError,
            utils.get_acquisition_function,
            acquisition_function_name="qUCB",
            model=self.model,
            X_observed=self.X_observed,
            objective=dummy_objective,
            X_pending=self.X_pending,
        )

    def testGetAcquisitionNotImplemented(self):
        self.assertRaises(
            NotImplementedError,
            utils.get_acquisition_function,
            "qES",
            self.model,
            self.X_observed,
            objective=dummy_objective,
            X_pending=self.X_pending,
        )

    @mock.patch(f"{utils.__name__}.qExpectedImprovement")
    def testAcquisitionFunctionArgs(self, mock_acquisition):
        acquisition_function = utils.get_acquisition_function(
            acquisition_function_name="qEI",
            model=self.model,
            X_observed=self.X_observed,
            objective=dummy_objective,
            constraints=self.constraints,
            infeasible_cost=self.infeasible_cost,
            X_pending=self.X_pending,
            acquisition_function_args={"mc_samples": 2},
            seed=self.seed,
        )
        self.assertTrue(acquisition_function == mock_acquisition.return_value)
        mock_acquisition.assert_called_once_with(
            model=self.model,
            mc_samples=2,
            best_f=self.model.posterior(self.X_observed).mean.max().item(),
            objective=dummy_objective,
            constraints=self.constraints,
            infeasible_cost=self.infeasible_cost,
            X_pending=self.X_pending,
            seed=self.seed,
        )
