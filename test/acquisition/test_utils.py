#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import warnings
from contextlib import ExitStack
from unittest import mock

import torch
from botorch import settings
from botorch.acquisition import monte_carlo
from botorch.acquisition.multi_objective import (
    MCMultiOutputObjective,
    monte_carlo as moo_monte_carlo,
)
from botorch.acquisition.objective import GenericMCObjective, MCAcquisitionObjective
from botorch.acquisition.utils import (
    expand_trace_observations,
    get_acquisition_function,
    get_infeasible_cost,
    project_to_sample_points,
    project_to_target_fidelity,
    prune_inferior_points,
)
from botorch.exceptions.errors import UnsupportedError
from botorch.exceptions.warnings import SamplingWarning
from botorch.sampling.samplers import IIDNormalSampler, SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    NondominatedPartitioning,
)
from botorch.utils.testing import BotorchTestCase, MockModel, MockPosterior
from torch import Tensor


class DummyMCObjective(MCAcquisitionObjective):
    def forward(self, samples: Tensor) -> Tensor:
        return samples.sum(-1)


class DummyMCMultiOutputObjective(MCMultiOutputObjective):
    def forward(self, samples: Tensor) -> Tensor:
        return samples


class TestGetAcquisitionFunction(BotorchTestCase):
    def setUp(self):
        super().setUp()
        self.model = mock.MagicMock()
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
        best_f = self.objective(self.model.posterior(self.X_observed).mean).max().item()
        mock_acqf.assert_called_once_with(
            model=self.model,
            best_f=best_f,
            sampler=mock.ANY,
            objective=self.objective,
            X_pending=self.X_pending,
        )
        args, kwargs = mock_acqf.call_args
        self.assertEqual(args, ())
        sampler = kwargs["sampler"]
        self.assertIsInstance(sampler, SobolQMCNormalSampler)
        self.assertEqual(sampler.sample_shape, torch.Size([self.mc_samples]))
        self.assertEqual(sampler.seed, 1)
        self.assertTrue(torch.equal(kwargs["X_pending"], self.X_pending))

    @mock.patch(f"{monte_carlo.__name__}.qProbabilityOfImprovement")
    def test_GetQPI(self, mock_acqf):
        # basic test
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
            X_pending=self.X_pending,
            tau=1e-3,
        )
        args, kwargs = mock_acqf.call_args
        self.assertEqual(args, ())
        sampler = kwargs["sampler"]
        self.assertIsInstance(sampler, SobolQMCNormalSampler)
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
        self.assertTrue(torch.equal(kwargs["X_pending"], self.X_pending))
        acqf = get_acquisition_function(
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
        )
        self.assertTrue(acqf == mock_acqf.return_value)
        self.assertTrue(mock_acqf.call_count, 1)
        args, kwargs = mock_acqf.call_args
        self.assertEqual(args, ())
        self.assertTrue(torch.equal(kwargs["X_baseline"], self.X_observed))
        self.assertTrue(torch.equal(kwargs["X_pending"], self.X_pending))
        sampler = kwargs["sampler"]
        self.assertIsInstance(sampler, SobolQMCNormalSampler)
        self.assertEqual(sampler.sample_shape, torch.Size([self.mc_samples]))
        self.assertEqual(sampler.seed, 1)
        # test with non-qmc, no X_pending
        acqf = get_acquisition_function(
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
        self.assertEqual(kwargs["X_pending"], None)
        sampler = kwargs["sampler"]
        self.assertIsInstance(sampler, IIDNormalSampler)
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
            X_pending=self.X_pending,
        )
        args, kwargs = mock_acqf.call_args
        self.assertEqual(args, ())
        sampler = kwargs["sampler"]
        self.assertIsInstance(sampler, SobolQMCNormalSampler)
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
            X_pending=self.X_pending,
        )
        args, kwargs = mock_acqf.call_args
        self.assertEqual(args, ())
        sampler = kwargs["sampler"]
        self.assertIsInstance(sampler, SobolQMCNormalSampler)
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
        self.assertIsInstance(sampler, SobolQMCNormalSampler)
        self.assertEqual(sampler.sample_shape, torch.Size([self.mc_samples]))
        self.assertEqual(sampler.seed, 1)
        # test with non-qmc
        acqf = get_acquisition_function(
            acquisition_function_name="qEHVI",
            model=self.model,
            objective=self.mo_objective,
            X_observed=self.X_observed,
            X_pending=self.X_pending,
            mc_samples=self.mc_samples,
            seed=2,
            qmc=False,
            ref_point=self.ref_point,
            Y=self.Y,
        )
        self.assertTrue(mock_acqf.call_count, 2)
        args, kwargs = mock_acqf.call_args
        self.assertEqual(args, ())
        self.assertEqual(kwargs["ref_point"], self.ref_point)
        sampler = kwargs["sampler"]
        self.assertIsInstance(sampler, IIDNormalSampler)
        self.assertIsInstance(kwargs["objective"], DummyMCMultiOutputObjective)
        partitioning = kwargs["partitioning"]
        self.assertIsInstance(partitioning, NondominatedPartitioning)

        self.assertEqual(sampler.sample_shape, torch.Size([self.mc_samples]))
        self.assertEqual(sampler.seed, 2)
        # test constraints
        acqf = get_acquisition_function(
            acquisition_function_name="qEHVI",
            model=self.model,
            objective=self.mo_objective,
            X_observed=self.X_observed,
            X_pending=self.X_pending,
            mc_samples=self.mc_samples,
            constraints=[lambda Y: Y[..., -1]],
            seed=2,
            qmc=False,
            ref_point=self.ref_point,
            Y=self.Y,
        )
        _, kwargs = mock_acqf.call_args
        partitioning = kwargs["partitioning"]
        self.assertEqual(partitioning.pareto_Y.shape[0], 0)

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
            X = torch.zeros(5, 1, device=self.device, dtype=dtype)
            means = torch.tensor(
                [1.0, 2.0, 3.0, 4.0, 5.0], device=self.device, dtype=dtype
            )
            variances = torch.tensor(
                [0.09, 0.25, 0.36, 0.25, 0.09], device=self.device, dtype=dtype
            )
            mm = MockModel(MockPosterior(mean=means, variance=variances))
            # means - 6 * std = [-0.8, -1, -0.6, 1, 3.2]. After applying the
            # objective, the minimum becomes -6.0, so 6.0 should be returned.
            M = get_infeasible_cost(
                X=X, model=mm, objective=lambda Y: Y.squeeze(-1) - 5.0
            )
            self.assertEqual(M, 6.0)
            # test default objective (squeeze last dim)
            M2 = get_infeasible_cost(X=X, model=mm)
            self.assertEqual(M2, 1.0)


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
            # test that a batched model raises errors (event shape is `q x t` = 3 x 1)
            mm2 = MockModel(MockPosterior(samples=samples.expand(2, 3, 1)))
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
            if self.device == torch.device("cuda"):
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
            # test high-dim sampling
            with ExitStack() as es:
                mock_event_shape = es.enter_context(
                    mock.patch(
                        "botorch.utils.testing.MockPosterior.event_shape",
                        new_callable=mock.PropertyMock,
                    )
                )
                mock_event_shape.return_value = torch.Size(
                    [1, 1, torch.quasirandom.SobolEngine.MAXDIM + 1]
                )
                es.enter_context(
                    mock.patch.object(MockPosterior, "rsample", return_value=samples)
                )
                mm = MockModel(MockPosterior(samples=samples))
                with warnings.catch_warnings(record=True) as ws, settings.debug(True):
                    prune_inferior_points(model=mm, X=X)
                    self.assertTrue(issubclass(ws[-1].category, SamplingWarning))


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
            out = (X_proj ** 2).sum()
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
            self.assertTrue(torch.allclose(X.grad, grad_exp))

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
                self.assertTrue(
                    torch.allclose(X_augmented[0, :, -d_prime:], sample_points)
                )
            else:
                self.assertTrue(
                    torch.allclose(X_augmented[:, -d_prime:], sample_points)
                )
