#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Callable
from unittest import mock

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.analytic import (
    ExpectedImprovement,
    NoisyExpectedImprovement,
    PosteriorMean,
    UpperConfidenceBound,
)
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.acquisition.input_constructors import (
    _deprecate_objective_arg,
    _field_is_shared,
    acqf_input_constructor,
    construct_inputs_mf_base,
    get_acqf_input_constructor,
    get_best_f_analytic,
    get_best_f_mc,
)
from botorch.acquisition.joint_entropy_search import (
    qJointEntropySearch,
)
from botorch.acquisition.knowledge_gradient import (
    qKnowledgeGradient,
    qMultiFidelityKnowledgeGradient,
)
from botorch.acquisition.max_value_entropy_search import (
    qMaxValueEntropy,
    qMultiFidelityMaxValueEntropy,
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
from botorch.acquisition.multi_objective.multi_output_risk_measures import (
    MultiOutputExpectation,
)
from botorch.acquisition.multi_objective.objective import (
    IdentityAnalyticMultiOutputObjective,
    IdentityMCMultiOutputObjective,
    WeightedMCMultiOutputObjective,
)
from botorch.acquisition.multi_objective.utils import get_default_partitioning_alpha
from botorch.acquisition.objective import (
    AcquisitionObjective,
    LinearMCObjective,
    ScalarizedObjective,
    ScalarizedPosteriorTransform,
)
from botorch.acquisition.preference import AnalyticExpectedUtilityOfBestOption
from botorch.acquisition.utils import (
    expand_trace_observations,
    project_to_target_fidelity,
)
from botorch.exceptions.errors import UnsupportedError
from botorch.models import SingleTaskGP
from botorch.models.deterministic import FixedSingleSampleModel
from botorch.sampling.normal import IIDNormalSampler, SobolQMCNormalSampler
from botorch.utils.constraints import get_outcome_constraint_transforms
from botorch.utils.datasets import SupervisedDataset
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
    NondominatedPartitioning,
)
from botorch.utils.testing import BotorchTestCase, MockModel, MockPosterior


class DummyAcquisitionFunction(AcquisitionFunction):
    ...


class DummyObjective(AcquisitionObjective):
    ...


class InputConstructorBaseTestCase:
    def setUp(self):
        X1 = torch.rand(3, 2)
        X2 = torch.rand(3, 2)
        Y1 = torch.rand(3, 1)
        Y2 = torch.rand(3, 1)

        self.blockX_blockY = SupervisedDataset.dict_from_iter(X1, Y1)
        self.blockX_multiY = SupervisedDataset.dict_from_iter(X1, (Y1, Y2))
        self.multiX_multiY = SupervisedDataset.dict_from_iter((X1, X2), (Y1, Y2))
        self.bounds = 2 * [(0.0, 1.0)]


class TestInputConstructorUtils(InputConstructorBaseTestCase, BotorchTestCase):
    def test_field_is_shared(self):
        self.assertTrue(_field_is_shared(self.blockX_multiY, "X"))
        self.assertFalse(_field_is_shared(self.blockX_multiY, "Y"))
        with self.assertRaisesRegex(AttributeError, "has no field"):
            self.assertFalse(_field_is_shared(self.blockX_multiY, "foo"))

    def test_get_best_f_analytic(self):
        with self.assertRaises(NotImplementedError):
            get_best_f_analytic(training_data=self.multiX_multiY)

        best_f = get_best_f_analytic(training_data=self.blockX_blockY)
        self.assertEqual(best_f, get_best_f_analytic(self.blockX_blockY[0]))

        best_f_expected = self.blockX_blockY[0].Y().squeeze().max()
        self.assertEqual(best_f, best_f_expected)
        with self.assertRaises(NotImplementedError):
            get_best_f_analytic(training_data=self.blockX_multiY)
        weights = torch.rand(2)
        obj = ScalarizedObjective(weights=weights)
        best_f_obj = get_best_f_analytic(
            training_data=self.blockX_multiY, objective=obj
        )
        post_tf = ScalarizedPosteriorTransform(weights=weights)
        best_f_tf = get_best_f_analytic(
            training_data=self.blockX_multiY, posterior_transform=post_tf
        )

        multi_Y = torch.cat([d.Y() for d in self.blockX_multiY.values()], dim=-1)
        best_f_expected = post_tf.evaluate(multi_Y).max()
        self.assertEqual(best_f_obj, best_f_expected)
        self.assertEqual(best_f_tf, best_f_expected)

    def test_get_best_f_mc(self):
        with self.assertRaises(NotImplementedError):
            get_best_f_mc(training_data=self.multiX_multiY)

        best_f = get_best_f_mc(training_data=self.blockX_blockY)
        self.assertEqual(best_f, get_best_f_mc(self.blockX_blockY[0]))

        best_f_expected = self.blockX_blockY[0].Y().squeeze().max()
        self.assertEqual(best_f, best_f_expected)
        with self.assertRaisesRegex(UnsupportedError, "require an objective"):
            get_best_f_mc(training_data=self.blockX_multiY)
        obj = LinearMCObjective(weights=torch.rand(2))
        best_f = get_best_f_mc(training_data=self.blockX_multiY, objective=obj)

        multi_Y = torch.cat([d.Y() for d in self.blockX_multiY.values()], dim=-1)
        best_f_expected = (multi_Y @ obj.weights).max()
        self.assertEqual(best_f, best_f_expected)
        post_tf = ScalarizedPosteriorTransform(weights=torch.ones(2))
        best_f = get_best_f_mc(
            training_data=self.blockX_multiY, posterior_transform=post_tf
        )
        best_f_expected = (multi_Y.sum(dim=-1)).max()
        self.assertEqual(best_f, best_f_expected)

    def test_deprecate_objective_arg(self):
        objective = ScalarizedObjective(weights=torch.ones(1))
        post_tf = ScalarizedPosteriorTransform(weights=torch.zeros(1))
        with self.assertRaises(RuntimeError):
            _deprecate_objective_arg(posterior_transform=post_tf, objective=objective)
        with self.assertWarns(DeprecationWarning):
            new_tf = _deprecate_objective_arg(objective=objective)
        self.assertTrue(torch.equal(new_tf.weights, objective.weights))
        self.assertIsInstance(new_tf, ScalarizedPosteriorTransform)
        new_tf = _deprecate_objective_arg(posterior_transform=post_tf)
        self.assertEqual(id(new_tf), id(post_tf))
        self.assertIsNone(_deprecate_objective_arg())
        with self.assertRaises(UnsupportedError):
            _deprecate_objective_arg(objective=DummyObjective())

    @mock.patch("botorch.acquisition.input_constructors.optimize_acqf")
    def test_optimize_objective(self, mock_optimize_acqf):
        from botorch.acquisition.input_constructors import optimize_objective

        mock_model = MockModel(posterior=MockPosterior(mean=None, variance=None))
        bounds = torch.rand(2, len(self.bounds))

        A = torch.rand(1, bounds.shape[-1])
        b = torch.zeros([1, 1])
        idx = A[0].nonzero(as_tuple=False).squeeze()
        inequality_constraints = ((idx, -A[0, idx], -b[0, 0]),)

        with self.subTest("scalarObjective_linearConstraints"):
            post_tf = ScalarizedPosteriorTransform(weights=torch.rand(bounds.shape[-1]))
            _ = optimize_objective(
                model=mock_model,
                bounds=bounds,
                q=1,
                posterior_transform=post_tf,
                linear_constraints=(A, b),
                fixed_features=None,
            )

            kwargs = mock_optimize_acqf.call_args[1]
            self.assertIsInstance(kwargs["acq_function"], PosteriorMean)
            self.assertTrue(torch.equal(kwargs["bounds"], bounds))
            self.assertEqual(len(kwargs["inequality_constraints"]), 1)
            for a, b in zip(
                kwargs["inequality_constraints"][0], inequality_constraints[0]
            ):
                self.assertTrue(torch.equal(a, b))

        with self.subTest("mcObjective_fixedFeatures"):
            _ = optimize_objective(
                model=mock_model,
                bounds=bounds,
                q=1,
                objective=LinearMCObjective(weights=torch.rand(bounds.shape[-1])),
                fixed_features={0: 0.5},
            )

            kwargs = mock_optimize_acqf.call_args[1]
            self.assertIsInstance(
                kwargs["acq_function"], FixedFeatureAcquisitionFunction
            )
            self.assertIsInstance(kwargs["acq_function"].acq_func, qSimpleRegret)
            self.assertTrue(torch.equal(kwargs["bounds"], bounds[:, 1:]))


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
        kwargs = c(model=mock_model, training_data=self.blockX_blockY)
        self.assertEqual(kwargs["model"], mock_model)
        self.assertIsNone(kwargs["posterior_transform"])
        mock_obj = mock.Mock()
        kwargs = c(
            model=mock_model,
            training_data=self.blockX_blockY,
            posterior_transform=mock_obj,
        )
        self.assertEqual(kwargs["model"], mock_model)
        self.assertEqual(kwargs["posterior_transform"], mock_obj)

    def test_construct_inputs_best_f(self):
        c = get_acqf_input_constructor(ExpectedImprovement)
        mock_model = mock.Mock()
        kwargs = c(model=mock_model, training_data=self.blockX_blockY)
        best_f_expected = self.blockX_blockY[0].Y().squeeze().max()
        self.assertEqual(kwargs["model"], mock_model)
        self.assertIsNone(kwargs["posterior_transform"])
        self.assertEqual(kwargs["best_f"], best_f_expected)
        self.assertTrue(kwargs["maximize"])
        kwargs = c(model=mock_model, training_data=self.blockX_blockY, best_f=0.1)
        self.assertEqual(kwargs["model"], mock_model)
        self.assertIsNone(kwargs["posterior_transform"])
        self.assertEqual(kwargs["best_f"], 0.1)
        self.assertTrue(kwargs["maximize"])

    def test_construct_inputs_ucb(self):
        c = get_acqf_input_constructor(UpperConfidenceBound)
        mock_model = mock.Mock()
        kwargs = c(model=mock_model, training_data=self.blockX_blockY)
        self.assertEqual(kwargs["model"], mock_model)
        self.assertIsNone(kwargs["posterior_transform"])
        self.assertEqual(kwargs["beta"], 0.2)
        self.assertTrue(kwargs["maximize"])
        kwargs = c(
            model=mock_model, training_data=self.blockX_blockY, beta=0.1, maximize=False
        )
        self.assertEqual(kwargs["model"], mock_model)
        self.assertIsNone(kwargs["posterior_transform"])
        self.assertEqual(kwargs["beta"], 0.1)
        self.assertFalse(kwargs["maximize"])

    # def test_construct_inputs_constrained_ei(self):
    #     c = get_acqf_input_constructor(ConstrainedExpectedImprovement)
    #     mock_model = mock.Mock()

    def test_construct_inputs_noisy_ei(self):
        c = get_acqf_input_constructor(NoisyExpectedImprovement)
        mock_model = mock.Mock()
        kwargs = c(model=mock_model, training_data=self.blockX_blockY)
        self.assertEqual(kwargs["model"], mock_model)
        self.assertTrue(torch.equal(kwargs["X_observed"], self.blockX_blockY[0].X()))
        self.assertEqual(kwargs["num_fantasies"], 20)
        self.assertTrue(kwargs["maximize"])
        kwargs = c(
            model=mock_model,
            training_data=self.blockX_blockY,
            num_fantasies=10,
            maximize=False,
        )
        self.assertEqual(kwargs["model"], mock_model)
        self.assertTrue(torch.equal(kwargs["X_observed"], self.blockX_blockY[0].X()))
        self.assertEqual(kwargs["num_fantasies"], 10)
        self.assertFalse(kwargs["maximize"])
        with self.assertRaisesRegex(ValueError, "Field `X` must be shared"):
            c(model=mock_model, training_data=self.multiX_multiY)

    def test_construct_inputs_constrained_analytic_eubo(self):
        c = get_acqf_input_constructor(AnalyticExpectedUtilityOfBestOption)
        mock_model = mock.Mock()
        mock_model.num_outputs = 3
        mock_model.train_inputs = [None]
        mock_pref_model = mock.Mock()
        kwargs = c(model=mock_model, pref_model=mock_pref_model)
        self.assertTrue(isinstance(kwargs["outcome_model"], FixedSingleSampleModel))
        self.assertTrue(kwargs["pref_model"] is mock_pref_model)
        self.assertTrue(kwargs["previous_winner"] is None)

        previous_winner = torch.randn(3)
        kwargs = c(
            model=mock_model,
            pref_model=mock_pref_model,
            previous_winner=previous_winner,
        )
        self.assertTrue(torch.equal(kwargs["previous_winner"], previous_winner))


class TestMCAcquisitionFunctionInputConstructors(
    InputConstructorBaseTestCase, BotorchTestCase
):
    def test_construct_inputs_mc_base(self):
        c = get_acqf_input_constructor(qSimpleRegret)
        mock_model = mock.Mock()
        kwargs = c(model=mock_model, training_data=self.blockX_blockY)
        self.assertEqual(kwargs["model"], mock_model)
        self.assertIsNone(kwargs["objective"])
        self.assertIsNone(kwargs["X_pending"])
        self.assertIsNone(kwargs["sampler"])
        X_pending = torch.rand(2, 2)
        objective = LinearMCObjective(torch.rand(2))
        kwargs = c(
            model=mock_model,
            training_data=self.blockX_blockY,
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
        kwargs = c(model=mock_model, training_data=self.blockX_blockY)
        self.assertEqual(kwargs["model"], mock_model)
        self.assertIsNone(kwargs["objective"])
        self.assertIsNone(kwargs["X_pending"])
        self.assertIsNone(kwargs["sampler"])
        X_pending = torch.rand(2, 2)
        objective = LinearMCObjective(torch.rand(2))
        kwargs = c(
            model=mock_model,
            training_data=self.blockX_multiY,
            objective=objective,
            X_pending=X_pending,
        )
        self.assertEqual(kwargs["model"], mock_model)
        self.assertTrue(torch.equal(kwargs["objective"].weights, objective.weights))
        self.assertTrue(torch.equal(kwargs["X_pending"], X_pending))
        self.assertIsNone(kwargs["sampler"])
        multi_Y = torch.cat([d.Y() for d in self.blockX_multiY.values()], dim=-1)
        best_f_expected = objective(multi_Y).max()
        self.assertEqual(kwargs["best_f"], best_f_expected)
        # Check explicitly specifying `best_f`.
        best_f_expected = best_f_expected - 1  # Random value.
        kwargs = c(
            model=mock_model,
            training_data=self.blockX_multiY,
            objective=objective,
            X_pending=X_pending,
            best_f=best_f_expected,
        )
        self.assertEqual(kwargs["best_f"], best_f_expected)

    def test_construct_inputs_qNEI(self):
        c = get_acqf_input_constructor(qNoisyExpectedImprovement)
        mock_model = mock.Mock()
        kwargs = c(model=mock_model, training_data=self.blockX_blockY)
        self.assertEqual(kwargs["model"], mock_model)
        self.assertIsNone(kwargs["objective"])
        self.assertIsNone(kwargs["X_pending"])
        self.assertIsNone(kwargs["sampler"])
        self.assertFalse(kwargs["prune_baseline"])
        self.assertTrue(torch.equal(kwargs["X_baseline"], self.blockX_blockY[0].X()))
        with self.assertRaisesRegex(ValueError, "Field `X` must be shared"):
            c(model=mock_model, training_data=self.multiX_multiY)
        X_baseline = torch.rand(2, 2)
        kwargs = c(
            model=mock_model,
            training_data=self.blockX_blockY,
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
        kwargs = c(model=mock_model, training_data=self.blockX_blockY)
        self.assertEqual(kwargs["model"], mock_model)
        self.assertIsNone(kwargs["objective"])
        self.assertIsNone(kwargs["X_pending"])
        self.assertIsNone(kwargs["sampler"])
        self.assertEqual(kwargs["tau"], 1e-3)
        X_pending = torch.rand(2, 2)
        objective = LinearMCObjective(torch.rand(2))
        kwargs = c(
            model=mock_model,
            training_data=self.blockX_multiY,
            objective=objective,
            X_pending=X_pending,
            tau=1e-2,
        )
        self.assertEqual(kwargs["model"], mock_model)
        self.assertTrue(torch.equal(kwargs["objective"].weights, objective.weights))
        self.assertTrue(torch.equal(kwargs["X_pending"], X_pending))
        self.assertIsNone(kwargs["sampler"])
        self.assertEqual(kwargs["tau"], 1e-2)
        multi_Y = torch.cat([d.Y() for d in self.blockX_multiY.values()], dim=-1)
        best_f_expected = objective(multi_Y).max()
        self.assertEqual(kwargs["best_f"], best_f_expected)
        # Check explicitly specifying `best_f`.
        best_f_expected = best_f_expected - 1  # Random value.
        kwargs = c(
            model=mock_model,
            training_data=self.blockX_multiY,
            objective=objective,
            X_pending=X_pending,
            tau=1e-2,
            best_f=best_f_expected,
        )
        self.assertEqual(kwargs["best_f"], best_f_expected)

    def test_construct_inputs_qUCB(self):
        c = get_acqf_input_constructor(qUpperConfidenceBound)
        mock_model = mock.Mock()
        kwargs = c(model=mock_model, training_data=self.blockX_blockY)
        self.assertEqual(kwargs["model"], mock_model)
        self.assertIsNone(kwargs["objective"])
        self.assertIsNone(kwargs["X_pending"])
        self.assertIsNone(kwargs["sampler"])
        self.assertEqual(kwargs["beta"], 0.2)
        X_pending = torch.rand(2, 2)
        objective = LinearMCObjective(torch.rand(2))
        kwargs = c(
            model=mock_model,
            training_data=self.blockX_blockY,
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
        objective_thresholds = torch.rand(6)

        # test error on non-block designs
        with self.assertRaisesRegex(ValueError, "Field `X` must be shared"):
            c(
                model=mock_model,
                training_data=self.multiX_multiY,
                objective_thresholds=objective_thresholds,
            )

        # test error on unsupported outcome constraints
        with self.assertRaises(NotImplementedError):
            c(
                model=mock_model,
                training_data=self.blockX_blockY,
                objective_thresholds=objective_thresholds,
                outcome_constraints=mock.Mock(),
            )

        # test with Y_pmean supplied explicitly
        Y_pmean = torch.rand(3, 6)
        kwargs = c(
            model=mock_model,
            training_data=self.blockX_blockY,
            objective_thresholds=objective_thresholds,
            Y_pmean=Y_pmean,
        )
        self.assertEqual(kwargs["model"], mock_model)
        self.assertIsInstance(kwargs["objective"], IdentityAnalyticMultiOutputObjective)
        self.assertTrue(torch.equal(kwargs["ref_point"], objective_thresholds))
        partitioning = kwargs["partitioning"]
        alpha_expected = get_default_partitioning_alpha(6)
        self.assertIsInstance(partitioning, NondominatedPartitioning)
        self.assertEqual(partitioning.alpha, alpha_expected)
        self.assertTrue(torch.equal(partitioning._neg_ref_point, -objective_thresholds))

        Y_pmean = torch.rand(3, 2)
        objective_thresholds = torch.rand(2)
        kwargs = c(
            model=mock_model,
            training_data=self.blockX_blockY,
            objective_thresholds=objective_thresholds,
            Y_pmean=Y_pmean,
        )
        partitioning = kwargs["partitioning"]
        self.assertIsInstance(partitioning, FastNondominatedPartitioning)
        self.assertTrue(torch.equal(partitioning.ref_point, objective_thresholds))

        # test with custom objective
        weights = torch.rand(2)
        obj = WeightedMCMultiOutputObjective(weights=weights)
        kwargs = c(
            model=mock_model,
            training_data=self.blockX_blockY,
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
            training_data=self.blockX_blockY,
            objective_thresholds=objective_thresholds,
        )
        self.assertIsInstance(kwargs["objective"], IdentityAnalyticMultiOutputObjective)
        self.assertTrue(torch.equal(kwargs["ref_point"], objective_thresholds))
        partitioning = kwargs["partitioning"]
        self.assertIsInstance(partitioning, FastNondominatedPartitioning)
        self.assertTrue(torch.equal(partitioning.ref_point, objective_thresholds))
        self.assertTrue(torch.equal(partitioning._neg_Y, -mean))

        # Test with risk measures.
        for use_preprocessing in (True, False):
            obj = MultiOutputExpectation(
                n_w=3,
                preprocessing_function=WeightedMCMultiOutputObjective(
                    torch.tensor([-1.0, -1.0])
                )
                if use_preprocessing
                else None,
            )
            kwargs = c(
                model=mm,
                training_data=self.blockX_blockY,
                objective_thresholds=objective_thresholds,
                objective=obj,
            )
            expected_obj_t = (
                -objective_thresholds if use_preprocessing else objective_thresholds
            )
            self.assertIs(kwargs["objective"], obj)
            self.assertTrue(torch.equal(kwargs["ref_point"], expected_obj_t))
            partitioning = kwargs["partitioning"]
            self.assertIsInstance(partitioning, FastNondominatedPartitioning)
            self.assertTrue(torch.equal(partitioning.ref_point, expected_obj_t))

    def test_construct_inputs_qEHVI(self):
        c = get_acqf_input_constructor(qExpectedHypervolumeImprovement)
        objective_thresholds = torch.rand(2)

        # Test defaults
        mm = SingleTaskGP(torch.rand(1, 2), torch.rand(1, 2))
        mean = mm.posterior(self.blockX_blockY[0].X()).mean
        kwargs = c(
            model=mm,
            training_data=self.blockX_blockY,
            objective_thresholds=objective_thresholds,
        )
        self.assertIsInstance(kwargs["objective"], IdentityMCMultiOutputObjective)
        ref_point_expected = objective_thresholds
        self.assertTrue(torch.equal(kwargs["ref_point"], ref_point_expected))
        partitioning = kwargs["partitioning"]
        self.assertIsInstance(partitioning, FastNondominatedPartitioning)
        self.assertTrue(torch.equal(partitioning.ref_point, ref_point_expected))
        self.assertTrue(torch.equal(partitioning._neg_Y, -mean))
        sampler = kwargs["sampler"]
        self.assertIsInstance(sampler, SobolQMCNormalSampler)
        self.assertEqual(sampler.sample_shape, torch.Size([128]))
        self.assertIsNone(kwargs["X_pending"])
        self.assertIsNone(kwargs["constraints"])
        self.assertEqual(kwargs["eta"], 1e-3)

        # Test IID sampler
        kwargs = c(
            model=mm,
            training_data=self.blockX_blockY,
            objective_thresholds=objective_thresholds,
            qmc=False,
            mc_samples=64,
        )
        sampler = kwargs["sampler"]
        self.assertIsInstance(sampler, IIDNormalSampler)
        self.assertEqual(sampler.sample_shape, torch.Size([64]))

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
            training_data=self.blockX_blockY,
            objective_thresholds=objective_thresholds,
            objective=obj,
            outcome_constraints=outcome_constraints,
            X_pending=X_pending,
            alpha=0.05,
            eta=1e-2,
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
        self.assertTrue(torch.equal(kwargs["X_pending"], X_pending))
        cons_tfs = kwargs["constraints"]
        self.assertEqual(len(cons_tfs), 1)
        cons_eval = cons_tfs[0](mean)
        cons_eval_expected = torch.tensor([-0.25, 0.5])
        self.assertTrue(torch.equal(cons_eval, cons_eval_expected))
        self.assertEqual(kwargs["eta"], 1e-2)

        # Test check for block designs
        with self.assertRaisesRegex(ValueError, "Field `X` must be shared"):
            c(
                model=mm,
                training_data=self.multiX_multiY,
                objective_thresholds=objective_thresholds,
                objective=obj,
                outcome_constraints=outcome_constraints,
                X_pending=X_pending,
                alpha=0.05,
                eta=1e-2,
            )

        # Test custom sampler
        custom_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([16]), seed=1234)
        kwargs = c(
            model=mm,
            training_data=self.blockX_blockY,
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

        # Test defaults
        kwargs = c(
            model=SingleTaskGP(torch.rand(1, 2), torch.rand(1, 2)),
            training_data=self.blockX_blockY,
            objective_thresholds=objective_thresholds,
        )
        ref_point_expected = objective_thresholds
        self.assertTrue(torch.equal(kwargs["ref_point"], ref_point_expected))
        self.assertTrue(torch.equal(kwargs["X_baseline"], self.blockX_blockY[0].X()))
        self.assertIsInstance(kwargs["sampler"], SobolQMCNormalSampler)
        self.assertEqual(kwargs["sampler"].sample_shape, torch.Size([128]))
        self.assertIsInstance(kwargs["objective"], IdentityMCMultiOutputObjective)
        self.assertIsNone(kwargs["constraints"])
        self.assertIsNone(kwargs["X_pending"])
        self.assertEqual(kwargs["eta"], 1e-3)
        self.assertTrue(kwargs["prune_baseline"])
        self.assertEqual(kwargs["alpha"], 0.0)
        self.assertTrue(kwargs["cache_pending"])
        self.assertEqual(kwargs["max_iep"], 0)
        self.assertTrue(kwargs["incremental_nehvi"])
        self.assertTrue(kwargs["cache_root"])

        # Test check for block designs
        mock_model = mock.Mock()
        mock_model.num_outputs = 2
        with self.assertRaisesRegex(ValueError, "Field `X` must be shared"):
            c(
                model=mock_model,
                training_data=self.multiX_multiY,
                objective_thresholds=objective_thresholds,
            )

        # Test custom inputs
        weights = torch.rand(2)
        objective = WeightedMCMultiOutputObjective(weights=weights)
        X_baseline = torch.rand(2, 2)
        sampler = IIDNormalSampler(sample_shape=torch.Size([4]))
        outcome_constraints = (torch.tensor([[0.0, 1.0]]), torch.tensor([[0.5]]))
        X_pending = torch.rand(1, 2)
        kwargs = c(
            model=mock_model,
            training_data=self.blockX_blockY,
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
            cache_root=False,
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
        self.assertFalse(kwargs["cache_root"])

        # Test with risk measures.
        with self.assertRaisesRegex(UnsupportedError, "feasibility-weighted"):
            kwargs = c(
                model=mock_model,
                training_data=self.blockX_blockY,
                objective_thresholds=objective_thresholds,
                objective=MultiOutputExpectation(n_w=3),
                outcome_constraints=outcome_constraints,
            )
        for use_preprocessing in (True, False):
            obj = MultiOutputExpectation(
                n_w=3,
                preprocessing_function=WeightedMCMultiOutputObjective(
                    torch.tensor([-1.0, -1.0])
                )
                if use_preprocessing
                else None,
            )
            kwargs = c(
                model=mock_model,
                training_data=self.blockX_blockY,
                objective_thresholds=objective_thresholds,
                objective=obj,
            )
            expected_obj_t = (
                -objective_thresholds if use_preprocessing else objective_thresholds
            )
            self.assertIs(kwargs["objective"], obj)
            self.assertTrue(torch.equal(kwargs["ref_point"], expected_obj_t))

        # Test default alpha for many objectives/
        mock_model.num_outputs = 5
        kwargs = c(
            model=mock_model,
            training_data=self.blockX_blockY,
            objective_thresholds=objective_thresholds,
        )
        self.assertEqual(kwargs["alpha"], 1e-3)

    def test_construct_inputs_kg(self):
        current_value = torch.tensor(1.23)
        with mock.patch(
            target="botorch.acquisition.input_constructors.optimize_objective",
            return_value=(None, current_value),
        ):
            from botorch.acquisition import input_constructors

            func = input_constructors.get_acqf_input_constructor(qKnowledgeGradient)
            kwargs = func(
                model=mock.Mock(),
                training_data=self.blockX_blockY,
                objective=LinearMCObjective(torch.rand(2)),
                bounds=self.bounds,
                num_fantasies=33,
            )

            self.assertEqual(kwargs["num_fantasies"], 33)
            self.assertEqual(kwargs["current_value"], current_value)

    def test_construct_inputs_mes(self):
        func = get_acqf_input_constructor(qMaxValueEntropy)
        kwargs = func(
            model=mock.Mock(),
            training_data=self.blockX_blockY,
            objective=LinearMCObjective(torch.rand(2)),
            bounds=self.bounds,
            candidate_size=17,
            maximize=False,
        )

        self.assertFalse(kwargs["maximize"])
        self.assertGreaterEqual(kwargs["candidate_set"].min(), 0.0)
        self.assertLessEqual(kwargs["candidate_set"].max(), 1.0)
        self.assertEqual(
            [int(s) for s in kwargs["candidate_set"].shape], [17, len(self.bounds)]
        )

    def test_construct_inputs_mf_base(self):
        target_fidelities = {0: 0.123}
        fidelity_weights = {0: 0.456}
        cost_intercept = 0.789
        num_trace_observations = 0

        with self.subTest("test_fully_specified"):
            kwargs = construct_inputs_mf_base(
                model=mock.Mock(),
                training_data=self.blockX_blockY,
                objective=LinearMCObjective(torch.rand(2)),
                target_fidelities=target_fidelities,
                fidelity_weights=fidelity_weights,
                cost_intercept=cost_intercept,
                num_trace_observations=num_trace_observations,
            )

            self.assertEqual(kwargs["target_fidelities"], target_fidelities)

            X = torch.rand(3, 2)
            self.assertTrue(isinstance(kwargs["expand"], Callable))
            self.assertTrue(
                torch.equal(
                    kwargs["expand"](X),
                    expand_trace_observations(
                        X=X,
                        fidelity_dims=sorted(target_fidelities),
                        num_trace_obs=num_trace_observations,
                    ),
                )
            )

            self.assertTrue(isinstance(kwargs["project"], Callable))
            self.assertTrue(
                torch.equal(
                    kwargs["project"](X),
                    project_to_target_fidelity(X, target_fidelities=target_fidelities),
                )
            )

            cm = kwargs["cost_aware_utility"].cost_model
            w = torch.tensor(list(fidelity_weights.values()), dtype=cm.weights.dtype)
            self.assertEqual(cm.fixed_cost, cost_intercept)
            self.assertAllClose(cm.weights, w)

        with self.subTest("test_missing_fidelity_weights"):
            kwargs = construct_inputs_mf_base(
                model=mock.Mock(),
                training_data=self.blockX_blockY,
                objective=LinearMCObjective(torch.rand(2)),
                target_fidelities=target_fidelities,
                cost_intercept=cost_intercept,
            )
            cm = kwargs["cost_aware_utility"].cost_model
            self.assertAllClose(cm.weights, torch.ones_like(cm.weights))

        with self.subTest("test_mismatched_weights"):
            with self.assertRaisesRegex(
                RuntimeError, "Must provide the same indices for"
            ):
                _ = construct_inputs_mf_base(
                    model=mock.Mock(),
                    training_data=self.blockX_blockY,
                    objective=LinearMCObjective(torch.rand(2)),
                    target_fidelities={0: 1.0},
                    fidelity_weights={1: 0.5},
                    cost_intercept=cost_intercept,
                )

    def test_construct_inputs_mfkg(self):
        constructor_args = {
            "model": None,
            "training_data": self.blockX_blockY,
            "objective": None,
            "bounds": self.bounds,
            "num_fantasies": 123,
            "target_fidelities": {0: 0.987},
            "fidelity_weights": {0: 0.654},
            "cost_intercept": 0.321,
        }
        with mock.patch(
            target="botorch.acquisition.input_constructors.construct_inputs_mf_base",
            return_value={"foo": 0},
        ), mock.patch(
            target="botorch.acquisition.input_constructors.construct_inputs_qKG",
            return_value={"bar": 1},
        ):
            from botorch.acquisition import input_constructors

            input_constructor = input_constructors.get_acqf_input_constructor(
                qMultiFidelityKnowledgeGradient
            )
            inputs_mfkg = input_constructor(**constructor_args)
            inputs_test = {"foo": 0, "bar": 1}
            self.assertEqual(inputs_mfkg, inputs_test)

    def test_construct_inputs_mfmes(self):
        constructor_args = {
            "model": None,
            "training_data": self.blockX_blockY,
            "objective": None,
            "bounds": self.bounds,
            "num_fantasies": 123,
            "candidate_size": 17,
            "target_fidelities": {0: 0.987},
            "fidelity_weights": {0: 0.654},
            "cost_intercept": 0.321,
        }
        current_value = torch.tensor(1.23)
        with mock.patch(
            target="botorch.acquisition.input_constructors.construct_inputs_mf_base",
            return_value={"foo": 0},
        ), mock.patch(
            target="botorch.acquisition.input_constructors.construct_inputs_qMES",
            return_value={"bar": 1},
        ), mock.patch(
            target="botorch.acquisition.input_constructors.optimize_objective",
            return_value=(None, current_value),
        ):
            from botorch.acquisition import input_constructors

            input_constructor = input_constructors.get_acqf_input_constructor(
                qMultiFidelityMaxValueEntropy
            )
            inputs_mfmes = input_constructor(**constructor_args)
            inputs_test = {"foo": 0, "bar": 1, "current_value": current_value}
            self.assertEqual(inputs_mfmes, inputs_test)

    def test_construct_inputs_jes(self):
        func = get_acqf_input_constructor(qJointEntropySearch)
        # we need to run optimize_posterior_samples, so we sort of need
        # a real model as there is no other (apparent) option
        model = SingleTaskGP(self.blockX_blockY[0].X(), self.blockX_blockY[0].Y())

        kwargs = func(
            model=model,
            training_data=self.blockX_blockY,
            objective=LinearMCObjective(torch.rand(2)),
            bounds=self.bounds,
            num_optima=17,
            maximize=False,
        )

        self.assertFalse(kwargs["maximize"])
        self.assertEqual(
            self.blockX_blockY[0].X().dtype, kwargs["optimal_inputs"].dtype
        )
        self.assertEqual(len(kwargs["optimal_inputs"]), 17)
        self.assertEqual(len(kwargs["optimal_outputs"]), 17)
        # asserting that, for the non-batch case, the optimal inputs are
        # of shape N x D and outputs are N x 1
        self.assertEqual(len(kwargs["optimal_inputs"].shape), 2)
        self.assertEqual(len(kwargs["optimal_outputs"].shape), 2)
