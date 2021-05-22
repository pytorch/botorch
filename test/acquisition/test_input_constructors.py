#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest import mock

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.analytic import (
    # ConstrainedExpectedImprovement,
    ExpectedImprovement,
    NoisyExpectedImprovement,
    PosteriorMean,
    UpperConfidenceBound,
)
from botorch.acquisition.input_constructors import (
    acqf_input_constructor,
    get_acqf_input_constructor,
    get_best_f_analytic,
    get_best_f_mc,
)
from botorch.acquisition.monte_carlo import (
    qExpectedImprovement,
    qNoisyExpectedImprovement,
    qProbabilityOfImprovement,
    qSimpleRegret,
    qUpperConfidenceBound,
)
from botorch.acquisition.multi_objective import (
    ExpectedHypervolumeImprovement,
    qExpectedHypervolumeImprovement,
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.objective import (
    IdentityAnalyticMultiOutputObjective,
    WeightedMCMultiOutputObjective,
)
from botorch.acquisition.multi_objective.utils import get_default_partitioning_alpha
from botorch.acquisition.objective import LinearMCObjective
from botorch.acquisition.objective import (
    ScalarizedObjective,
)
from botorch.exceptions.errors import UnsupportedError
from botorch.sampling.samplers import IIDNormalSampler, SobolQMCNormalSampler
from botorch.utils.constraints import get_outcome_constraint_transforms
from botorch.utils.containers import TrainingData
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    NondominatedPartitioning,
)
from botorch.utils.testing import BotorchTestCase, MockModel, MockPosterior


class DummyAcquisitionFunction(AcquisitionFunction):
    ...


class InputConstructorBaseTestCase:
    def setUp(self):
        X = torch.rand(3, 2)
        Y = torch.rand(3, 1)
        self.bd_td = TrainingData.from_block_design(X=X, Y=Y)
        self.bd_td_mo = TrainingData.from_block_design(X=X, Y=torch.rand(3, 2))
        Xs = [torch.rand(2, 2), torch.rand(2, 2)]
        Ys = [torch.rand(2, 1), torch.rand(2, 1)]
        self.nbd_td = TrainingData(Xs=Xs, Ys=Ys)


class TestInputConstructorUtils(InputConstructorBaseTestCase, BotorchTestCase):
    def test_get_best_f_analytic(self):
        with self.assertRaises(NotImplementedError):
            get_best_f_analytic(training_data=self.nbd_td)
        best_f = get_best_f_analytic(training_data=self.bd_td)
        best_f_expected = self.bd_td.Y.squeeze().max()
        self.assertEqual(best_f, best_f_expected)
        with self.assertRaises(NotImplementedError):
            get_best_f_analytic(training_data=self.bd_td_mo)
        obj = ScalarizedObjective(weights=torch.rand(2))
        best_f = get_best_f_analytic(training_data=self.bd_td_mo, objective=obj)
        best_f_expected = obj.evaluate(self.bd_td_mo.Y).max()
        self.assertEqual(best_f, best_f_expected)

    def test_get_best_f_mc(self):
        with self.assertRaises(NotImplementedError):
            get_best_f_mc(training_data=self.nbd_td)
        best_f = get_best_f_mc(training_data=self.bd_td)
        best_f_expected = self.bd_td.Y.squeeze().max()
        self.assertEqual(best_f, best_f_expected)
        with self.assertRaises(UnsupportedError):
            get_best_f_mc(training_data=self.bd_td_mo)
        obj = LinearMCObjective(weights=torch.rand(2))
        best_f = get_best_f_mc(training_data=self.bd_td_mo, objective=obj)
        best_f_expected = (self.bd_td_mo.Y @ obj.weights).max()
        self.assertEqual(best_f, best_f_expected)


class TestAnalyticAcquisitionFunctionInputConstructors(
    InputConstructorBaseTestCase, BotorchTestCase
):
    def test_acqf_input_constructor(self):
        with self.assertRaises(RuntimeError) as e:
            get_acqf_input_constructor(DummyAcquisitionFunction)
            self.assertTrue("not registered" in str(e))

        with self.assertRaises(ValueError) as e:

            @acqf_input_constructor(ExpectedImprovement)
            class NewAcquisitionFunction(AcquisitionFunction):
                ...

            self.assertTrue("duplicate" in str(e))

    def test_construct_inputs_analytic_base(self):
        c = get_acqf_input_constructor(PosteriorMean)
        mock_model = mock.Mock()
        kwargs = c(model=mock_model, training_data=self.bd_td)
        self.assertEqual(kwargs["model"], mock_model)
        self.assertIsNone(kwargs["objective"])
        mock_obj = mock.Mock()
        kwargs = c(model=mock_model, training_data=self.bd_td, objective=mock_obj)
        self.assertEqual(kwargs["model"], mock_model)
        self.assertEqual(kwargs["objective"], mock_obj)

    def test_construct_inputs_best_f(self):
        c = get_acqf_input_constructor(ExpectedImprovement)
        mock_model = mock.Mock()
        kwargs = c(model=mock_model, training_data=self.bd_td)
        best_f_expected = self.bd_td.Y.squeeze().max()
        self.assertEqual(kwargs["model"], mock_model)
        self.assertIsNone(kwargs["objective"])
        self.assertEqual(kwargs["best_f"], best_f_expected)
        self.assertTrue(kwargs["maximize"])
        kwargs = c(model=mock_model, training_data=self.bd_td, best_f=0.1)
        self.assertEqual(kwargs["model"], mock_model)
        self.assertIsNone(kwargs["objective"])
        self.assertEqual(kwargs["best_f"], 0.1)
        self.assertTrue(kwargs["maximize"])

    def test_construct_inputs_ucb(self):
        c = get_acqf_input_constructor(UpperConfidenceBound)
        mock_model = mock.Mock()
        kwargs = c(model=mock_model, training_data=self.bd_td)
        self.assertEqual(kwargs["model"], mock_model)
        self.assertIsNone(kwargs["objective"])
        self.assertEqual(kwargs["beta"], 0.2)
        self.assertTrue(kwargs["maximize"])
        kwargs = c(model=mock_model, training_data=self.bd_td, beta=0.1, maximize=False)
        self.assertEqual(kwargs["model"], mock_model)
        self.assertIsNone(kwargs["objective"])
        self.assertEqual(kwargs["beta"], 0.1)
        self.assertFalse(kwargs["maximize"])

    # def test_construct_inputs_constrained_ei(self):
    #     c = get_acqf_input_constructor(ConstrainedExpectedImprovement)
    #     mock_model = mock.Mock()

    def test_construct_inputs_noisy_ei(self):
        c = get_acqf_input_constructor(NoisyExpectedImprovement)
        mock_model = mock.Mock()
        kwargs = c(model=mock_model, training_data=self.bd_td)
        self.assertEqual(kwargs["model"], mock_model)
        self.assertTrue(torch.equal(kwargs["X_observed"], self.bd_td.X))
        self.assertEqual(kwargs["num_fantasies"], 20)
        self.assertTrue(kwargs["maximize"])
        kwargs = c(
            model=mock_model, training_data=self.bd_td, num_fantasies=10, maximize=False
        )
        self.assertEqual(kwargs["model"], mock_model)
        self.assertTrue(torch.equal(kwargs["X_observed"], self.bd_td.X))
        self.assertEqual(kwargs["num_fantasies"], 10)
        self.assertFalse(kwargs["maximize"])
        with self.assertRaisesRegex(NotImplementedError, "only block designs"):
            c(model=mock_model, training_data=self.nbd_td)


class TestMCAcquisitionFunctionInputConstructors(
    InputConstructorBaseTestCase, BotorchTestCase
):
    def test_construct_inputs_mc_base(self):
        c = get_acqf_input_constructor(qSimpleRegret)
        mock_model = mock.Mock()
        kwargs = c(model=mock_model, training_data=self.bd_td)
        self.assertEqual(kwargs["model"], mock_model)
        self.assertIsNone(kwargs["objective"])
        self.assertIsNone(kwargs["X_pending"])
        self.assertIsNone(kwargs["sampler"])
        X_pending = torch.rand(2, 2)
        objective = LinearMCObjective(torch.rand(2))
        kwargs = c(
            model=mock_model,
            training_data=self.bd_td,
            objective=objective,
            X_pending=X_pending,
        )
        self.assertEqual(kwargs["model"], mock_model)
        self.assertTrue(torch.equal(kwargs["objective"].weights, objective.weights))
        self.assertTrue(torch.equal(kwargs["X_pending"], X_pending))
        self.assertIsNone(kwargs["sampler"])
        # TODO: Test passing through of sampler

    def test_construct_inputs_qEI(self):
        c = get_acqf_input_constructor(qExpectedImprovement)
        mock_model = mock.Mock()
        kwargs = c(model=mock_model, training_data=self.bd_td)
        best_f_expected = self.bd_td.Y.squeeze().max()
        self.assertEqual(kwargs["model"], mock_model)
        self.assertIsNone(kwargs["objective"])
        self.assertIsNone(kwargs["X_pending"])
        self.assertIsNone(kwargs["sampler"])
        X_pending = torch.rand(2, 2)
        objective = LinearMCObjective(torch.rand(2))
        kwargs = c(
            model=mock_model,
            training_data=self.bd_td_mo,
            objective=objective,
            X_pending=X_pending,
        )
        self.assertEqual(kwargs["model"], mock_model)
        self.assertTrue(torch.equal(kwargs["objective"].weights, objective.weights))
        self.assertTrue(torch.equal(kwargs["X_pending"], X_pending))
        self.assertIsNone(kwargs["sampler"])
        best_f_expected = objective(self.bd_td_mo.Y).max()
        self.assertEqual(kwargs["best_f"], best_f_expected)

    def test_construct_inputs_qNEI(self):
        c = get_acqf_input_constructor(qNoisyExpectedImprovement)
        mock_model = mock.Mock()
        kwargs = c(model=mock_model, training_data=self.bd_td)
        self.assertEqual(kwargs["model"], mock_model)
        self.assertIsNone(kwargs["objective"])
        self.assertIsNone(kwargs["X_pending"])
        self.assertIsNone(kwargs["sampler"])
        self.assertFalse(kwargs["prune_baseline"])
        self.assertTrue(torch.equal(kwargs["X_baseline"], self.bd_td.X))
        with self.assertRaises(NotImplementedError):
            c(model=mock_model, training_data=self.nbd_td)
        X_baseline = torch.rand(2, 2)
        kwargs = c(
            model=mock_model,
            training_data=self.bd_td,
            X_baseline=X_baseline,
            prune_baseline=True,
        )
        self.assertEqual(kwargs["model"], mock_model)
        self.assertIsNone(kwargs["objective"])
        self.assertIsNone(kwargs["X_pending"])
        self.assertIsNone(kwargs["sampler"])
        self.assertTrue(kwargs["prune_baseline"])
        self.assertTrue(torch.equal(kwargs["X_baseline"], X_baseline))

    def test_construct_inputs_qPI(self):
        c = get_acqf_input_constructor(qProbabilityOfImprovement)
        mock_model = mock.Mock()
        kwargs = c(model=mock_model, training_data=self.bd_td)
        self.assertEqual(kwargs["model"], mock_model)
        self.assertIsNone(kwargs["objective"])
        self.assertIsNone(kwargs["X_pending"])
        self.assertIsNone(kwargs["sampler"])
        self.assertEqual(kwargs["tau"], 1e-3)
        X_pending = torch.rand(2, 2)
        objective = LinearMCObjective(torch.rand(2))
        kwargs = c(
            model=mock_model,
            training_data=self.bd_td,
            objective=objective,
            X_pending=X_pending,
            tau=1e-2,
        )
        self.assertEqual(kwargs["model"], mock_model)
        self.assertTrue(torch.equal(kwargs["objective"].weights, objective.weights))
        self.assertTrue(torch.equal(kwargs["X_pending"], X_pending))
        self.assertIsNone(kwargs["sampler"])
        self.assertEqual(kwargs["tau"], 1e-2)

    def test_construct_inputs_qUCB(self):
        c = get_acqf_input_constructor(qUpperConfidenceBound)
        mock_model = mock.Mock()
        kwargs = c(model=mock_model, training_data=self.bd_td)
        self.assertEqual(kwargs["model"], mock_model)
        self.assertIsNone(kwargs["objective"])
        self.assertIsNone(kwargs["X_pending"])
        self.assertIsNone(kwargs["sampler"])
        self.assertEqual(kwargs["beta"], 0.2)
        X_pending = torch.rand(2, 2)
        objective = LinearMCObjective(torch.rand(2))
        kwargs = c(
            model=mock_model,
            training_data=self.bd_td,
            objective=objective,
            X_pending=X_pending,
            beta=0.1,
        )
        self.assertEqual(kwargs["model"], mock_model)
        self.assertTrue(torch.equal(kwargs["objective"].weights, objective.weights))
        self.assertTrue(torch.equal(kwargs["X_pending"], X_pending))
        self.assertIsNone(kwargs["sampler"])
        self.assertEqual(kwargs["beta"], 0.1)


class TestMultiObjectiveAcquisitionFunctionInputConstructors(
    InputConstructorBaseTestCase, BotorchTestCase
):
    def test_construct_inputs_EHVI(self):
        c = get_acqf_input_constructor(ExpectedHypervolumeImprovement)
        mock_model = mock.Mock()
        objective_thresholds = torch.rand(2)

        # test error on unsupported outcome constraints
        with self.assertRaises(NotImplementedError):
            c(
                model=mock_model,
                training_data=self.bd_td,
                objective_thresholds=objective_thresholds,
                outcome_constraints=mock.Mock(),
            )

        # test with Y_pmean supplied explicitly
        Y_pmean = torch.rand(3, 2)
        kwargs = c(
            model=mock_model,
            training_data=self.bd_td,
            objective_thresholds=objective_thresholds,
            Y_pmean=Y_pmean,
        )
        self.assertEqual(kwargs["model"], mock_model)
        self.assertIsInstance(kwargs["objective"], IdentityAnalyticMultiOutputObjective)
        ref_point_expected = objective_thresholds
        self.assertTrue(torch.equal(kwargs["ref_point"], ref_point_expected))
        partitioning = kwargs["partitioning"]
        alpha_expected = get_default_partitioning_alpha(2)
        self.assertIsInstance(partitioning, NondominatedPartitioning)
        self.assertEqual(partitioning.alpha, alpha_expected)
        self.assertTrue(torch.equal(partitioning._neg_ref_point, -ref_point_expected))

        # test with custom objective
        weights = torch.rand(2)
        obj = WeightedMCMultiOutputObjective(weights=weights)
        kwargs = c(
            model=mock_model,
            training_data=self.bd_td,
            objective_thresholds=objective_thresholds,
            objective=obj,
            Y_pmean=Y_pmean,
            alpha=0.05,
        )
        self.assertEqual(kwargs["model"], mock_model)
        self.assertIsInstance(kwargs["objective"], WeightedMCMultiOutputObjective)
        ref_point_expected = objective_thresholds * weights
        self.assertTrue(torch.equal(kwargs["ref_point"], ref_point_expected))
        partitioning = kwargs["partitioning"]
        self.assertIsInstance(partitioning, NondominatedPartitioning)
        self.assertEqual(partitioning.alpha, 0.05)
        self.assertTrue(torch.equal(partitioning._neg_ref_point, -ref_point_expected))

        # Test without providing Y_pmean (computed from model)
        mean = torch.rand(1, 2)
        variance = torch.ones(1, 1)
        mm = MockModel(MockPosterior(mean=mean, variance=variance))
        kwargs = c(
            model=mm,
            training_data=self.bd_td,
            objective_thresholds=objective_thresholds,
        )
        self.assertIsInstance(kwargs["objective"], IdentityAnalyticMultiOutputObjective)
        ref_point_expected = objective_thresholds
        self.assertTrue(torch.equal(kwargs["ref_point"], ref_point_expected))
        partitioning = kwargs["partitioning"]
        alpha_expected = get_default_partitioning_alpha(2)
        self.assertIsInstance(partitioning, NondominatedPartitioning)
        self.assertEqual(partitioning.alpha, alpha_expected)
        self.assertTrue(torch.equal(partitioning._neg_ref_point, -ref_point_expected))
        self.assertTrue(torch.equal(partitioning._neg_Y, -mean))

    def test_construct_inputs_qEHVI(self):
        c = get_acqf_input_constructor(qExpectedHypervolumeImprovement)
        objective_thresholds = torch.rand(2)

        # Test defaults
        mean = torch.rand(1, 2)
        variance = torch.ones(1, 1)
        mm = MockModel(MockPosterior(mean=mean, variance=variance))
        kwargs = c(
            model=mm,
            training_data=self.bd_td,
            objective_thresholds=objective_thresholds,
        )
        self.assertIsInstance(kwargs["objective"], IdentityAnalyticMultiOutputObjective)
        ref_point_expected = objective_thresholds
        self.assertTrue(torch.equal(kwargs["ref_point"], ref_point_expected))
        partitioning = kwargs["partitioning"]
        alpha_expected = get_default_partitioning_alpha(2)
        self.assertIsInstance(partitioning, NondominatedPartitioning)
        self.assertEqual(partitioning.alpha, alpha_expected)
        self.assertTrue(torch.equal(partitioning._neg_ref_point, -ref_point_expected))
        self.assertTrue(torch.equal(partitioning._neg_Y, -mean))
        sampler = kwargs["sampler"]
        self.assertIsInstance(sampler, SobolQMCNormalSampler)
        self.assertEqual(sampler.sample_shape, torch.Size([128]))
        self.assertIsNone(kwargs["X_pending"])
        self.assertIsNone(kwargs["constraints"])
        self.assertEqual(kwargs["eta"], 1e-3)

        # Test outcome constraints and custom inputs
        mean = torch.tensor([[1.0, 0.25], [0.5, 1.0]])
        variance = torch.ones(1, 1)
        mm = MockModel(MockPosterior(mean=mean, variance=variance))
        weights = torch.rand(2)
        obj = WeightedMCMultiOutputObjective(weights=weights)
        outcome_constraints = (torch.tensor([[0.0, 1.0]]), torch.tensor([[0.5]]))
        X_pending = torch.rand(1, 2)
        kwargs = c(
            model=mm,
            training_data=self.bd_td,
            objective_thresholds=objective_thresholds,
            objective=obj,
            outcome_constraints=outcome_constraints,
            X_pending=X_pending,
            alpha=0.05,
            eta=1e-2,
            qmc=False,
            mc_samples=64,
        )
        self.assertIsInstance(kwargs["objective"], WeightedMCMultiOutputObjective)
        ref_point_expected = objective_thresholds * weights
        self.assertTrue(torch.equal(kwargs["ref_point"], ref_point_expected))
        partitioning = kwargs["partitioning"]
        self.assertIsInstance(partitioning, NondominatedPartitioning)
        self.assertEqual(partitioning.alpha, 0.05)
        self.assertTrue(torch.equal(partitioning._neg_ref_point, -ref_point_expected))
        Y_expected = mean[:1] * weights
        self.assertTrue(torch.equal(partitioning._neg_Y, -Y_expected))
        sampler = kwargs["sampler"]
        self.assertIsInstance(sampler, IIDNormalSampler)
        self.assertEqual(sampler.sample_shape, torch.Size([64]))
        self.assertTrue(torch.equal(kwargs["X_pending"], X_pending))
        cons_tfs = kwargs["constraints"]
        self.assertEqual(len(cons_tfs), 1)
        cons_eval = cons_tfs[0](mean)
        cons_eval_expected = torch.tensor([-0.25, 0.5])
        self.assertTrue(torch.equal(cons_eval, cons_eval_expected))
        self.assertEqual(kwargs["eta"], 1e-2)

        # Test custom sampler
        custom_sampler = SobolQMCNormalSampler(num_samples=16, seed=1234)
        kwargs = c(
            model=mm,
            training_data=self.bd_td,
            objective_thresholds=objective_thresholds,
            sampler=custom_sampler,
        )
        sampler = kwargs["sampler"]
        self.assertIsInstance(sampler, SobolQMCNormalSampler)
        self.assertEqual(sampler.sample_shape, torch.Size([16]))
        self.assertEqual(sampler.seed, 1234)

    def test_construct_inputs_qNEHVI(self):
        c = get_acqf_input_constructor(qNoisyExpectedHypervolumeImprovement)
        objective_thresholds = torch.rand(2)
        mock_model = mock.Mock()

        # Test defaults
        kwargs = c(
            model=mock_model,
            training_data=self.bd_td,
            objective_thresholds=objective_thresholds,
        )
        ref_point_expected = objective_thresholds
        self.assertTrue(torch.equal(kwargs["ref_point"], ref_point_expected))
        self.assertTrue(torch.equal(kwargs["X_baseline"], self.bd_td.X))
        self.assertIsNone(kwargs["sampler"])
        self.assertIsInstance(kwargs["objective"], IdentityAnalyticMultiOutputObjective)
        self.assertIsNone(kwargs["constraints"])
        self.assertIsNone(kwargs["X_pending"])
        self.assertEqual(kwargs["eta"], 1e-3)
        self.assertFalse(kwargs["prune_baseline"])
        self.assertEqual(kwargs["alpha"], 0.0)
        self.assertTrue(kwargs["cache_pending"])
        self.assertEqual(kwargs["max_iep"], 0)
        self.assertTrue(kwargs["incremental_nehvi"])

        # Test custom inputs
        weights = torch.rand(2)
        objective = WeightedMCMultiOutputObjective(weights=weights)
        X_baseline = torch.rand(2, 2)
        sampler = IIDNormalSampler(num_samples=4)
        outcome_constraints = (torch.tensor([[0.0, 1.0]]), torch.tensor([[0.5]]))
        X_pending = torch.rand(1, 2)
        kwargs = c(
            model=mock_model,
            training_data=self.bd_td,
            objective_thresholds=objective_thresholds,
            objective=objective,
            X_baseline=X_baseline,
            sampler=sampler,
            outcome_constraints=outcome_constraints,
            X_pending=X_pending,
            eta=1e-2,
            prune_baseline=True,
            alpha=0.1,
            cache_pending=False,
            max_iep=1,
            incremental_nehvi=False,
        )
        ref_point_expected = objective(objective_thresholds)
        self.assertTrue(torch.equal(kwargs["ref_point"], ref_point_expected))
        self.assertTrue(torch.equal(kwargs["X_baseline"], X_baseline))
        sampler_ = kwargs["sampler"]
        self.assertIsInstance(sampler_, IIDNormalSampler)
        self.assertEqual(sampler_.sample_shape, torch.Size([4]))
        self.assertEqual(kwargs["objective"], objective)
        cons_tfs_expected = get_outcome_constraint_transforms(outcome_constraints)
        cons_tfs = kwargs["constraints"]
        self.assertEqual(len(cons_tfs), 1)
        test_Y = torch.rand(1, 2)
        self.assertTrue(torch.equal(cons_tfs[0](test_Y), cons_tfs_expected[0](test_Y)))
        self.assertTrue(torch.equal(kwargs["X_pending"], X_pending))
        self.assertEqual(kwargs["eta"], 1e-2)
        self.assertTrue(kwargs["prune_baseline"])
        self.assertEqual(kwargs["alpha"], 0.1)
        self.assertFalse(kwargs["cache_pending"])
        self.assertEqual(kwargs["max_iep"], 1)
        self.assertFalse(kwargs["incremental_nehvi"])
