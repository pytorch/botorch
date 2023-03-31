#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
from unittest import mock

import torch
from botorch.acquisition import monte_carlo
from botorch.acquisition.multi_objective import (
    MCMultiOutputObjective,
    monte_carlo as moo_monte_carlo,
)
from botorch.acquisition.objective import (
    GenericMCObjective,
    MCAcquisitionObjective,
    ScalarizedPosteriorTransform,
)
from botorch.acquisition.utils import (
    expand_trace_observations,
    get_acquisition_function,
    get_infeasible_cost,
    project_to_sample_points,
    project_to_target_fidelity,
    prune_inferior_points,
    get_optimal_samples,
)
from botorch.exceptions.errors import UnsupportedError
from botorch.models import SingleTaskGP
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
    NondominatedPartitioning,
)
from botorch.utils.testing import BotorchTestCase, MockModel, MockPosterior
from gpytorch.distributions import MultivariateNormal
from torch import Tensor


class DummyMCObjective(MCAcquisitionObjective):
    def forward(self, samples: Tensor, X=None) -> Tensor:
        return samples.sum(-1)


class DummyMCMultiOutputObjective(MCMultiOutputObjective):
    def forward(self, samples: Tensor, X=None) -> Tensor:
        return samples


class TestGetAcquisitionFunction(BotorchTestCase):
    def setUp(self):
        super().setUp()
        self.model = MockModel(MockPosterior())
        self.objective = DummyMCObjective()
        self.X_observed = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
        self.X_pending = torch.tensor([[1.0, 3.0, 4.0]])
        self.mc_samples = 250
        self.qmc = True
        self.ref_point = [0.0, 0.0]
        self.mo_objective = DummyMCMultiOutputObjective()
        self.Y = torch.tensor([[1.0, 2.0]])
        self.seed = 1

    @mock.patch(f"{monte_carlo.__name__}.qExpectedImprovement")
    def test_GetQEI(self, mock_acqf):
        self.model = MockModel(MockPosterior(mean=torch.zeros(1, 2)))
        acqf = get_acquisition_function(
            acquisition_function_name="qEI",
            model=self.model,
            objective=self.objective,
            X_observed=self.X_observed,
            X_pending=self.X_pending,
            mc_samples=self.mc_samples,
            seed=self.seed,
            marginalize_dim=0,
        )
        self.assertTrue(acqf == mock_acqf.return_value)
        best_f = self.objective(self.model.posterior(self.X_observed).mean).max().item()
        mock_acqf.assert_called_once_with(
            model=self.model,
            best_f=best_f,
            sampler=mock.ANY,
            objective=self.objective,
            posterior_transform=None,
            X_pending=self.X_pending,
        )
        # test batched model
        self.model = MockModel(MockPosterior(mean=torch.zeros(1, 2, 1)))
        acqf = get_acquisition_function(
            acquisition_function_name="qEI",
            model=self.model,
            objective=self.objective,
            X_observed=self.X_observed,
            X_pending=self.X_pending,
            mc_samples=self.mc_samples,
            seed=self.seed,
        )
        self.assertTrue(acqf == mock_acqf.return_value)
        # test batched model without marginalize dim
        args, kwargs = mock_acqf.call_args
        self.assertEqual(args, ())
        sampler = kwargs["sampler"]
        self.assertEqual(sampler.sample_shape, torch.Size([self.mc_samples]))
        self.assertEqual(sampler.seed, 1)
        self.assertTrue(torch.equal(kwargs["X_pending"], self.X_pending))

        # test w/ posterior transform
        pm = torch.tensor([1.0, 2.0])
        mvn = MultivariateNormal(pm, torch.eye(2))
        self.model._posterior.distribution = mvn
        self.model._posterior._mean = pm.unsqueeze(-1)
        pt = ScalarizedPosteriorTransform(weights=torch.tensor([-1]))
        acqf = get_acquisition_function(
            acquisition_function_name="qEI",
            model=self.model,
            objective=self.objective,
            X_observed=self.X_observed,
            posterior_transform=pt,
            X_pending=self.X_pending,
            mc_samples=self.mc_samples,
            seed=self.seed,
            marginalize_dim=0,
        )
        self.assertEqual(mock_acqf.call_args[-1]["best_f"].item(), -1.0)

    @mock.patch(f"{monte_carlo.__name__}.qProbabilityOfImprovement")
    def test_GetQPI(self, mock_acqf):
        # basic test
        self.model = MockModel(MockPosterior(mean=torch.zeros(1, 2)))
        acqf = get_acquisition_function(
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
            posterior_transform=None,
            X_pending=self.X_pending,
            tau=1e-3,
        )
        args, kwargs = mock_acqf.call_args
        self.assertEqual(args, ())
        sampler = kwargs["sampler"]
        self.assertEqual(sampler.sample_shape, torch.Size([self.mc_samples]))
        self.assertEqual(sampler.seed, 1)
        self.assertTrue(torch.equal(kwargs["X_pending"], self.X_pending))
        # test with different tau, non-qmc
        acqf = get_acquisition_function(
            acquisition_function_name="qPI",
            model=self.model,
            objective=self.objective,
            X_observed=self.X_observed,
            X_pending=self.X_pending,
            mc_samples=self.mc_samples,
            seed=2,
            tau=1.0,
        )
        self.assertTrue(mock_acqf.call_count, 2)
        args, kwargs = mock_acqf.call_args
        self.assertEqual(args, ())
        self.assertEqual(kwargs["tau"], 1.0)
        sampler = kwargs["sampler"]
        self.assertEqual(sampler.sample_shape, torch.Size([self.mc_samples]))
        self.assertEqual(sampler.seed, 2)
        self.assertTrue(torch.equal(kwargs["X_pending"], self.X_pending))
        acqf = get_acquisition_function(
            acquisition_function_name="qPI",
            model=self.model,
            objective=self.objective,
            X_observed=self.X_observed,
            X_pending=self.X_pending,
            mc_samples=self.mc_samples,
            seed=2,
            tau=1.0,
        )
        # test batched model
        self.model = MockModel(MockPosterior(mean=torch.zeros(1, 2, 1)))
        acqf = get_acquisition_function(
            acquisition_function_name="qPI",
            model=self.model,
            objective=self.objective,
            X_observed=self.X_observed,
            X_pending=self.X_pending,
            mc_samples=self.mc_samples,
            seed=self.seed,
        )
        self.assertTrue(acqf == mock_acqf.return_value)

    @mock.patch(f"{monte_carlo.__name__}.qNoisyExpectedImprovement")
    def test_GetQNEI(self, mock_acqf):
        # basic test
        acqf = get_acquisition_function(
            acquisition_function_name="qNEI",
            model=self.model,
            objective=self.objective,
            X_observed=self.X_observed,
            X_pending=self.X_pending,
            mc_samples=self.mc_samples,
            seed=self.seed,
            marginalize_dim=0,
        )
        self.assertTrue(acqf == mock_acqf.return_value)
        self.assertTrue(mock_acqf.call_count, 1)
        args, kwargs = mock_acqf.call_args
        self.assertEqual(args, ())
        self.assertTrue(torch.equal(kwargs["X_baseline"], self.X_observed))
        self.assertTrue(torch.equal(kwargs["X_pending"], self.X_pending))
        sampler = kwargs["sampler"]
        self.assertEqual(sampler.sample_shape, torch.Size([self.mc_samples]))
        self.assertEqual(sampler.seed, 1)
        self.assertEqual(kwargs["marginalize_dim"], 0)
        self.assertEqual(kwargs["cache_root"], True)
        # test with cache_root = False
        acqf = get_acquisition_function(
            acquisition_function_name="qNEI",
            model=self.model,
            objective=self.objective,
            X_observed=self.X_observed,
            X_pending=self.X_pending,
            mc_samples=self.mc_samples,
            seed=self.seed,
            marginalize_dim=0,
            cache_root=False,
        )
        self.assertTrue(acqf == mock_acqf.return_value)
        self.assertTrue(mock_acqf.call_count, 1)
        args, kwargs = mock_acqf.call_args
        self.assertEqual(kwargs["cache_root"], False)
        # test with non-qmc, no X_pending
        acqf = get_acquisition_function(
            acquisition_function_name="qNEI",
            model=self.model,
            objective=self.objective,
            X_observed=self.X_observed,
            X_pending=None,
            mc_samples=self.mc_samples,
            seed=2,
        )
        self.assertTrue(mock_acqf.call_count, 2)
        args, kwargs = mock_acqf.call_args
        self.assertEqual(args, ())
        self.assertTrue(torch.equal(kwargs["X_baseline"], self.X_observed))
        self.assertEqual(kwargs["X_pending"], None)
        sampler = kwargs["sampler"]
        self.assertEqual(sampler.sample_shape, torch.Size([self.mc_samples]))
        self.assertEqual(sampler.seed, 2)
        self.assertTrue(torch.equal(kwargs["X_baseline"], self.X_observed))

    @mock.patch(f"{monte_carlo.__name__}.qSimpleRegret")
    def test_GetQSR(self, mock_acqf):
        # basic test
        acqf = get_acquisition_function(
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
            model=self.model,
            sampler=mock.ANY,
            objective=self.objective,
            posterior_transform=None,
            X_pending=self.X_pending,
        )
        args, kwargs = mock_acqf.call_args
        self.assertEqual(args, ())
        sampler = kwargs["sampler"]
        self.assertEqual(sampler.sample_shape, torch.Size([self.mc_samples]))
        self.assertEqual(sampler.seed, 1)
        self.assertTrue(torch.equal(kwargs["X_pending"], self.X_pending))
        # test with non-qmc
        acqf = get_acquisition_function(
            acquisition_function_name="qSR",
            model=self.model,
            objective=self.objective,
            X_observed=self.X_observed,
            X_pending=self.X_pending,
            mc_samples=self.mc_samples,
            seed=2,
        )
        self.assertTrue(mock_acqf.call_count, 2)
        args, kwargs = mock_acqf.call_args
        self.assertEqual(args, ())
        sampler = kwargs["sampler"]
        self.assertEqual(sampler.sample_shape, torch.Size([self.mc_samples]))
        self.assertEqual(sampler.seed, 2)
        self.assertTrue(torch.equal(kwargs["X_pending"], self.X_pending))

    @mock.patch(f"{monte_carlo.__name__}.qUpperConfidenceBound")
    def test_GetQUCB(self, mock_acqf):
        # make sure beta is specified
        with self.assertRaises(ValueError):
            acqf = get_acquisition_function(
                acquisition_function_name="qUCB",
                model=self.model,
                objective=self.objective,
                X_observed=self.X_observed,
                X_pending=self.X_pending,
                mc_samples=self.mc_samples,
                seed=self.seed,
            )
        acqf = get_acquisition_function(
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
            model=self.model,
            beta=0.3,
            sampler=mock.ANY,
            objective=self.objective,
            posterior_transform=None,
            X_pending=self.X_pending,
        )
        args, kwargs = mock_acqf.call_args
        self.assertEqual(args, ())
        sampler = kwargs["sampler"]
        self.assertEqual(sampler.sample_shape, torch.Size([self.mc_samples]))
        self.assertEqual(sampler.seed, 1)
        self.assertTrue(torch.equal(kwargs["X_pending"], self.X_pending))
        # test with different tau, non-qmc
        acqf = get_acquisition_function(
            acquisition_function_name="qUCB",
            model=self.model,
            objective=self.objective,
            X_observed=self.X_observed,
            X_pending=self.X_pending,
            mc_samples=self.mc_samples,
            seed=2,
            beta=0.2,
        )
        self.assertTrue(mock_acqf.call_count, 2)
        args, kwargs = mock_acqf.call_args
        self.assertEqual(args, ())
        self.assertEqual(kwargs["beta"], 0.2)
        sampler = kwargs["sampler"]
        self.assertEqual(sampler.sample_shape, torch.Size([self.mc_samples]))
        self.assertEqual(sampler.seed, 2)
        self.assertTrue(torch.equal(kwargs["X_pending"], self.X_pending))

    @mock.patch(f"{moo_monte_carlo.__name__}.qExpectedHypervolumeImprovement")
    def test_GetQEHVI(self, mock_acqf):
        # make sure ref_point is specified
        with self.assertRaises(ValueError):
            acqf = get_acquisition_function(
                acquisition_function_name="qEHVI",
                model=self.model,
                objective=self.mo_objective,
                X_observed=self.X_observed,
                X_pending=self.X_pending,
                mc_samples=self.mc_samples,
                seed=self.seed,
                Y=self.Y,
            )
        # make sure Y is specified
        with self.assertRaises(ValueError):
            acqf = get_acquisition_function(
                acquisition_function_name="qEHVI",
                model=self.model,
                objective=self.mo_objective,
                X_observed=self.X_observed,
                X_pending=self.X_pending,
                mc_samples=self.mc_samples,
                seed=self.seed,
                ref_point=self.ref_point,
            )
        # posterior transforms are not supported
        with self.assertRaises(NotImplementedError):
            acqf = get_acquisition_function(
                acquisition_function_name="qEHVI",
                model=self.model,
                objective=self.mo_objective,
                posterior_transform=ScalarizedPosteriorTransform(weights=torch.rand(2)),
                X_observed=self.X_observed,
                X_pending=self.X_pending,
                mc_samples=self.mc_samples,
                seed=self.seed,
                ref_point=self.ref_point,
            )
        acqf = get_acquisition_function(
            acquisition_function_name="qEHVI",
            model=self.model,
            objective=self.mo_objective,
            X_observed=self.X_observed,
            X_pending=self.X_pending,
            mc_samples=self.mc_samples,
            seed=self.seed,
            ref_point=self.ref_point,
            Y=self.Y,
        )
        self.assertTrue(acqf == mock_acqf.return_value)
        mock_acqf.assert_called_once_with(
            constraints=None,
            eta=1e-3,
            model=self.model,
            objective=self.mo_objective,
            ref_point=self.ref_point,
            partitioning=mock.ANY,
            sampler=mock.ANY,
            X_pending=self.X_pending,
        )
        args, kwargs = mock_acqf.call_args
        self.assertEqual(args, ())
        sampler = kwargs["sampler"]
        self.assertEqual(sampler.sample_shape, torch.Size([self.mc_samples]))
        self.assertEqual(sampler.seed, 1)

        acqf = get_acquisition_function(
            acquisition_function_name="qEHVI",
            model=self.model,
            objective=self.mo_objective,
            X_observed=self.X_observed,
            X_pending=self.X_pending,
            mc_samples=self.mc_samples,
            seed=2,
            ref_point=self.ref_point,
            Y=self.Y,
        )
        self.assertTrue(mock_acqf.call_count, 2)
        args, kwargs = mock_acqf.call_args
        self.assertEqual(args, ())
        self.assertEqual(kwargs["ref_point"], self.ref_point)
        sampler = kwargs["sampler"]
        self.assertIsInstance(kwargs["objective"], DummyMCMultiOutputObjective)
        partitioning = kwargs["partitioning"]
        self.assertIsInstance(partitioning, FastNondominatedPartitioning)
        self.assertEqual(sampler.sample_shape, torch.Size([self.mc_samples]))
        self.assertEqual(sampler.seed, 2)
        # test that approximate partitioning is used when alpha > 0
        acqf = get_acquisition_function(
            acquisition_function_name="qEHVI",
            model=self.model,
            objective=self.mo_objective,
            X_observed=self.X_observed,
            X_pending=self.X_pending,
            mc_samples=self.mc_samples,
            seed=2,
            ref_point=self.ref_point,
            Y=self.Y,
            alpha=0.1,
        )
        _, kwargs = mock_acqf.call_args
        partitioning = kwargs["partitioning"]
        self.assertIsInstance(partitioning, NondominatedPartitioning)
        self.assertEqual(partitioning.alpha, 0.1)
        # test constraints
        acqf = get_acquisition_function(
            acquisition_function_name="qEHVI",
            model=self.model,
            objective=self.mo_objective,
            X_observed=self.X_observed,
            X_pending=self.X_pending,
            mc_samples=self.mc_samples,
            constraints=[lambda Y: Y[..., -1]],
            eta=1e-2,
            seed=2,
            ref_point=self.ref_point,
            Y=self.Y,
        )
        _, kwargs = mock_acqf.call_args
        partitioning = kwargs["partitioning"]
        self.assertEqual(partitioning.pareto_Y.shape[0], 0)
        self.assertEqual(kwargs["eta"], 1e-2)

    @mock.patch(f"{moo_monte_carlo.__name__}.qNoisyExpectedHypervolumeImprovement")
    def test_GetQNEHVI(self, mock_acqf):
        # make sure ref_point is specified
        with self.assertRaises(ValueError):
            acqf = get_acquisition_function(
                acquisition_function_name="qNEHVI",
                model=self.model,
                objective=self.objective,
                X_observed=self.X_observed,
                X_pending=self.X_pending,
                mc_samples=self.mc_samples,
                seed=self.seed,
            )
        acqf = get_acquisition_function(
            acquisition_function_name="qNEHVI",
            model=self.model,
            objective=self.objective,
            X_observed=self.X_observed,
            X_pending=self.X_pending,
            mc_samples=self.mc_samples,
            seed=self.seed,
            ref_point=self.ref_point,
        )
        self.assertTrue(acqf == mock_acqf.return_value)
        mock_acqf.assert_called_once_with(
            constraints=None,
            eta=1e-3,
            model=self.model,
            X_baseline=self.X_observed,
            objective=self.objective,
            ref_point=self.ref_point,
            sampler=mock.ANY,
            prune_baseline=True,
            alpha=0.0,
            X_pending=self.X_pending,
            marginalize_dim=None,
            cache_root=True,
        )
        args, kwargs = mock_acqf.call_args
        self.assertEqual(args, ())
        sampler = kwargs["sampler"]
        self.assertEqual(sampler.sample_shape, torch.Size([self.mc_samples]))
        self.assertEqual(sampler.seed, 1)
        # test with non-qmc
        acqf = get_acquisition_function(
            acquisition_function_name="qNEHVI",
            model=self.model,
            objective=self.objective,
            X_observed=self.X_observed,
            X_pending=self.X_pending,
            mc_samples=self.mc_samples,
            seed=2,
            ref_point=self.ref_point,
        )
        self.assertTrue(mock_acqf.call_count, 2)
        args, kwargs = mock_acqf.call_args
        self.assertEqual(args, ())
        self.assertEqual(kwargs["ref_point"], self.ref_point)
        sampler = kwargs["sampler"]
        ref_point = kwargs["ref_point"]
        self.assertEqual(ref_point, self.ref_point)
        self.assertEqual(sampler.sample_shape, torch.Size([self.mc_samples]))
        self.assertEqual(sampler.seed, 2)

        # test passing alpha
        acqf = get_acquisition_function(
            acquisition_function_name="qNEHVI",
            model=self.model,
            objective=self.objective,
            X_observed=self.X_observed,
            X_pending=self.X_pending,
            mc_samples=self.mc_samples,
            seed=2,
            ref_point=self.ref_point,
            alpha=0.01,
        )
        self.assertTrue(mock_acqf.call_count, 3)
        args, kwargs = mock_acqf.call_args
        self.assertEqual(kwargs["alpha"], 0.01)

    def test_GetUnknownAcquisitionFunction(self):
        with self.assertRaises(NotImplementedError):
            get_acquisition_function(
                acquisition_function_name="foo",
                model=self.model,
                objective=self.objective,
                X_observed=self.X_observed,
                X_pending=self.X_pending,
                mc_samples=self.mc_samples,
                seed=self.seed,
            )


class TestGetInfeasibleCost(BotorchTestCase):
    def test_get_infeasible_cost(self):
        for dtype in (torch.float, torch.double):
            tkwargs = {"dtype": dtype, "device": self.device}
            X = torch.ones(5, 1, **tkwargs)
            means = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], **tkwargs).view(-1, 1)
            variances = torch.tensor([0.09, 0.25, 0.36, 0.25, 0.09], **tkwargs).view(
                -1, 1
            )
            mm = MockModel(MockPosterior(mean=means, variance=variances))
            # means - 6 * std = [-0.8, -1, -0.6, 1, 3.2]. After applying the
            # objective, the minimum becomes -6.0, so 6.0 should be returned.
            M = get_infeasible_cost(
                X=X, model=mm, objective=lambda Y, X: Y.squeeze(-1) - 5.0
            )
            self.assertAllClose(M, torch.tensor([6.0], **tkwargs))
            M = get_infeasible_cost(
                X=X, model=mm, objective=lambda Y, X: Y.squeeze(-1) - 5.0 - X[0, 0]
            )
            self.assertAllClose(M, torch.tensor([7.0], **tkwargs))
            # test it with using also X in the objective
            # Test default objective (squeeze last dim).
            M2 = get_infeasible_cost(X=X, model=mm)
            self.assertAllClose(M2, torch.tensor([1.0], **tkwargs))
            # Test multi-output.
            m_ = means.repeat(1, 2)
            m_[:, 1] -= 10
            mm = MockModel(MockPosterior(mean=m_, variance=variances.expand(-1, 2)))
            M3 = get_infeasible_cost(X=X, model=mm)
            self.assertAllClose(M3, torch.tensor([1.0, 11.0], **tkwargs))
            # With a batched model.
            means = means.expand(2, 4, -1, -1)
            variances = variances.expand(2, 4, -1, -1)
            mm = MockModel(MockPosterior(mean=means, variance=variances))
            M4 = get_infeasible_cost(X=X, model=mm)
            self.assertAllClose(M4, torch.tensor([1.0], **tkwargs))


class TestPruneInferiorPoints(BotorchTestCase):
    def test_prune_inferior_points(self):
        for dtype in (torch.float, torch.double):
            X = torch.rand(3, 2, device=self.device, dtype=dtype)
            # the event shape is `q x t` = 3 x 1
            samples = torch.tensor(
                [[-1.0], [0.0], [1.0]], device=self.device, dtype=dtype
            )
            mm = MockModel(MockPosterior(samples=samples))
            # test that a batched X raises errors
            with self.assertRaises(UnsupportedError):
                prune_inferior_points(model=mm, X=X.expand(2, 3, 2))
            # test marginalize_dim
            mm2 = MockModel(MockPosterior(samples=samples.expand(2, 3, 1)))
            X_pruned = prune_inferior_points(model=mm2, X=X, marginalize_dim=-3)
            with self.assertRaises(UnsupportedError):
                # test error raised when marginalize_dim is not specified with
                # a batch model
                prune_inferior_points(model=mm2, X=X)
            self.assertTrue(torch.equal(X_pruned, X[[-1]]))
            # test that a batched model raises errors when there are multiple batch dims
            mm2 = MockModel(MockPosterior(samples=samples.expand(1, 2, 3, 1)))
            with self.assertRaises(UnsupportedError):
                prune_inferior_points(model=mm2, X=X)
            # test that invalid max_frac is checked properly
            with self.assertRaises(ValueError):
                prune_inferior_points(model=mm, X=X, max_frac=1.1)
            # test basic behaviour
            X_pruned = prune_inferior_points(model=mm, X=X)
            self.assertTrue(torch.equal(X_pruned, X[[-1]]))
            # test custom objective
            neg_id_obj = GenericMCObjective(lambda Y, X: -(Y.squeeze(-1)))
            X_pruned = prune_inferior_points(model=mm, X=X, objective=neg_id_obj)
            self.assertTrue(torch.equal(X_pruned, X[[0]]))
            # test non-repeated samples (requires mocking out MockPosterior's rsample)
            samples = torch.tensor(
                [[[3.0], [0.0], [0.0]], [[0.0], [2.0], [0.0]], [[0.0], [0.0], [1.0]]],
                device=self.device,
                dtype=dtype,
            )
            with mock.patch.object(MockPosterior, "rsample", return_value=samples):
                mm = MockModel(MockPosterior(samples=samples))
                X_pruned = prune_inferior_points(model=mm, X=X)
            self.assertTrue(torch.equal(X_pruned, X))
            # test max_frac limiting
            with mock.patch.object(MockPosterior, "rsample", return_value=samples):
                mm = MockModel(MockPosterior(samples=samples))
                X_pruned = prune_inferior_points(model=mm, X=X, max_frac=2 / 3)
            if self.device.type == "cuda":
                # sorting has different order on cuda
                self.assertTrue(torch.equal(X_pruned, torch.stack([X[2], X[1]], dim=0)))
            else:
                self.assertTrue(torch.equal(X_pruned, X[:2]))
            # test that zero-probability is in fact pruned
            samples[2, 0, 0] = 10
            with mock.patch.object(MockPosterior, "rsample", return_value=samples):
                mm = MockModel(MockPosterior(samples=samples))
                X_pruned = prune_inferior_points(model=mm, X=X)
            self.assertTrue(torch.equal(X_pruned, X[:2]))


class TestFidelityUtils(BotorchTestCase):
    def test_project_to_target_fidelity(self):
        for batch_shape, dtype in itertools.product(
            ([], [2]), (torch.float, torch.double)
        ):
            X = torch.rand(*batch_shape, 3, 4, device=self.device, dtype=dtype)
            # test default behavior
            X_proj = project_to_target_fidelity(X)
            ones = torch.ones(*X.shape[:-1], 1, device=self.device, dtype=dtype)
            self.assertTrue(torch.equal(X_proj[..., :, [-1]], ones))
            self.assertTrue(torch.equal(X_proj[..., :-1], X[..., :-1]))
            # test custom target fidelity
            target_fids = {2: 0.5}
            X_proj = project_to_target_fidelity(X, target_fidelities=target_fids)
            self.assertTrue(torch.equal(X_proj[..., :, [2]], 0.5 * ones))
            # test multiple target fidelities
            target_fids = {2: 0.5, 0: 0.1}
            X_proj = project_to_target_fidelity(X, target_fidelities=target_fids)
            self.assertTrue(torch.equal(X_proj[..., :, [0]], 0.1 * ones))
            self.assertTrue(torch.equal(X_proj[..., :, [2]], 0.5 * ones))
            # test gradients
            X.requires_grad_(True)
            X_proj = project_to_target_fidelity(X, target_fidelities=target_fids)
            out = (X_proj**2).sum()
            out.backward()
            self.assertTrue(torch.all(X.grad[..., [0, 2]] == 0))
            self.assertTrue(torch.equal(X.grad[..., [1, 3]], 2 * X[..., [1, 3]]))

    def test_expand_trace_observations(self):
        for batch_shape, dtype in itertools.product(
            ([], [2]), (torch.float, torch.double)
        ):
            q, d = 3, 4
            X = torch.rand(*batch_shape, q, d, device=self.device, dtype=dtype)
            # test nullop behavior
            self.assertTrue(torch.equal(expand_trace_observations(X), X))
            self.assertTrue(
                torch.equal(expand_trace_observations(X, fidelity_dims=[1]), X)
            )
            # test default behavior
            num_tr = 2
            X_expanded = expand_trace_observations(X, num_trace_obs=num_tr)
            self.assertEqual(
                X_expanded.shape, torch.Size(batch_shape + [q * (1 + num_tr), d])
            )
            for i in range(num_tr):
                X_sub = X_expanded[..., q * i : q * (i + 1), :]
                self.assertTrue(torch.equal(X_sub[..., :-1], X[..., :-1]))
                X_sub_expected = (1 - i / (num_tr + 1)) * X[..., :q, -1]
                self.assertTrue(torch.equal(X_sub[..., -1], X_sub_expected))
            # test custom fidelity dims
            fdims = [0, 2]
            num_tr = 3
            X_expanded = expand_trace_observations(
                X, fidelity_dims=fdims, num_trace_obs=num_tr
            )
            self.assertEqual(
                X_expanded.shape, torch.Size(batch_shape + [q * (1 + num_tr), d])
            )
            for j, i in itertools.product([1, 3], range(num_tr)):
                X_sub = X_expanded[..., q * i : q * (i + 1), j]
                self.assertTrue(torch.equal(X_sub, X[..., j]))
            for j, i in itertools.product(fdims, range(num_tr)):
                X_sub = X_expanded[..., q * i : q * (i + 1), j]
                X_sub_expected = (1 - i / (1 + num_tr)) * X[..., :q, j]
                self.assertTrue(torch.equal(X_sub, X_sub_expected))
            # test gradients
            num_tr = 2
            fdims = [1]
            X.requires_grad_(True)
            X_expanded = expand_trace_observations(
                X, fidelity_dims=fdims, num_trace_obs=num_tr
            )
            out = X_expanded.sum()
            out.backward()
            grad_exp = torch.full_like(X, 1 + num_tr)
            grad_exp[..., fdims] = 1 + sum(
                (i + 1) / (num_tr + 1) for i in range(num_tr)
            )
            self.assertAllClose(X.grad, grad_exp)

    def test_project_to_sample_points(self):
        for batch_shape, dtype in itertools.product(
            ([], [2]), (torch.float, torch.double)
        ):
            q, d, p, d_prime = 1, 12, 7, 4
            X = torch.rand(*batch_shape, q, d, device=self.device, dtype=dtype)
            sample_points = torch.rand(p, d_prime, device=self.device, dtype=dtype)
            X_augmented = project_to_sample_points(X=X, sample_points=sample_points)
            self.assertEqual(X_augmented.shape, torch.Size(batch_shape + [p, d]))
            if batch_shape == [2]:
                self.assertAllClose(X_augmented[0, :, -d_prime:], sample_points)
            else:
                self.assertAllClose(X_augmented[:, -d_prime:], sample_points)


class TestGetOptimalSamples(BotorchTestCase):
    def test_get_optimal_samples(self):
        dims = 3
        dtype = torch.float64
        for_testing_speed_kwargs = {"raw_samples": 50, "num_restarts": 3}
        num_optima = 7
        batch_shape = (3,)

        bounds = torch.tensor([[0, 1]] * dims, dtype=dtype).T
        X = torch.rand(*batch_shape, 4, dims, dtype=dtype)
        Y = torch.sin(X).sum(dim=-1, keepdim=True).to(dtype)
        model = SingleTaskGP(X, Y)
        X_opt, f_opt = get_optimal_samples(
            model, bounds, num_optima=num_optima, **for_testing_speed_kwargs
        )
        X_opt, f_opt_min = get_optimal_samples(
            model,
            bounds,
            num_optima=num_optima,
            maximize=False,
            **for_testing_speed_kwargs,
        )

        correct_X_shape = (num_optima,) + batch_shape + (dims,)
        correct_f_shape = (num_optima,) + batch_shape + (1,)
        self.assertEqual(X_opt.shape, correct_X_shape)
        self.assertEqual(f_opt.shape, correct_f_shape)
        # asserting that the solutions found by minimization the samples are smaller
        # than those found by maximization
        self.assertTrue(torch.all(f_opt_min < f_opt))
