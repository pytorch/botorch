#!/usr/bin/env python3

import unittest
from unittest import mock

import torch
from botorch.acquisition import utils


def objective(Y):
    return Y


class TestGetAcquisitionFunction(unittest.TestCase):
    def setUp(self):
        self.model = mock.MagicMock()
        self.X_observed = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
        self.X_pending = torch.tensor([[1.0, 3.0, 4.0]])

    @mock.patch(f"{utils.__name__}.qExpectedImprovement")
    def testGetQEI(self, mock_acquisition):
        acquisition_function = utils.get_acquisition_function(
            "qEI",
            self.model,
            self.X_observed,
            objective=objective,
            constraints=None,
            X_pending=self.X_pending,
            seed=None,
            acquisition_function_args=None,
        )
        self.assertTrue(acquisition_function == mock_acquisition.return_value)
        mock_acquisition.assert_called_once_with(
            self.model,
            best_f=self.model.posterior(self.X_observed).mean.max().item(),
            objective=objective,
            constraints=None,
            seed=None,
            X_pending=self.X_pending,
        )

    @mock.patch(f"{utils.__name__}.qProbabilityOfImprovement")
    def testGetQPI(self, mock_acquisition):
        acquisition_function = utils.get_acquisition_function(
            "qPI",
            self.model,
            self.X_observed,
            objective=objective,
            constraints=None,
            X_pending=self.X_pending,
            seed=None,
            acquisition_function_args=None,
        )
        self.assertTrue(acquisition_function == mock_acquisition.return_value)
        mock_acquisition.assert_called_once_with(
            self.model,
            best_f=self.model.posterior(self.X_observed).mean.max().item(),
            objective=objective,
            constraints=None,
            X_pending=self.X_pending,
            seed=None,
        )

    @mock.patch(f"{utils.__name__}.qNoisyExpectedImprovement")
    def testGetQNEI(self, mock_acquisition):
        acquisition_function = utils.get_acquisition_function(
            "qNEI",
            self.model,
            self.X_observed,
            objective=objective,
            constraints=None,
            X_pending=self.X_pending,
            seed=None,
            acquisition_function_args=None,
        )
        self.assertTrue(acquisition_function == mock_acquisition.return_value)
        mock_acquisition.assert_called_once_with(
            self.model,
            self.X_observed,
            objective=objective,
            constraints=None,
            X_pending=self.X_pending,
            seed=None,
        )

    @mock.patch(f"{utils.__name__}.qUpperConfidenceBound")
    def testGetQUCB(self, mock_acquisition):
        acquisition_function = utils.get_acquisition_function(
            "qUCB",
            self.model,
            self.X_observed,
            objective=objective,
            constraints=None,
            X_pending=self.X_pending,
            seed=None,
            acquisition_function_args={"beta": 2.0},
        )
        self.assertTrue(acquisition_function == mock_acquisition.return_value)
        mock_acquisition.assert_called_once_with(
            self.model, beta=2.0, X_pending=self.X_pending, seed=None
        )

    def testGetQUCBNoBeta(self):
        self.assertRaises(
            ValueError,
            utils.get_acquisition_function,
            "qUCB",
            self.model,
            self.X_observed,
            objective=objective,
            constraints=None,
            X_pending=self.X_pending,
            seed=None,
            acquisition_function_args={},
        )

    def testGetAcquisitionNotImplemented(self):
        self.assertRaises(
            NotImplementedError,
            utils.get_acquisition_function,
            "qES",
            self.model,
            self.X_observed,
            objective=objective,
            constraints=None,
            X_pending=self.X_pending,
            seed=None,
        )
