#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
from unittest import mock

import torch
from botorch.acquisition.objective import GenericMCObjective
from botorch.acquisition.utils import (
    compute_best_feasible_objective,
    expand_trace_observations,
    get_acquisition_function,
    get_infeasible_cost,
    get_optimal_samples,
    project_to_sample_points,
    project_to_target_fidelity,
    prune_inferior_points,
)
from botorch.exceptions.errors import DeprecationError, UnsupportedError
from botorch.models import SingleTaskGP

from botorch.utils.testing import BotorchTestCase, MockModel, MockPosterior


class TestGetAcquisitionFunctionDeprecation(BotorchTestCase):
    def test_get_acquisition_function_deprecation(self):
        msg = (
            "`get_acquisition_function` has been moved to"
            " `botorch.acquisition.factory`."
        )
        with self.assertRaisesRegex(DeprecationError, msg):
            get_acquisition_function()


class TestConstraintUtils(BotorchTestCase):
    def test_compute_best_feasible_objective(self):
        for dtype in (torch.float, torch.double):
            with self.subTest(dtype=dtype):
                tkwargs = {"dtype": dtype, "device": self.device}
                n = 5
                X = torch.arange(n, **tkwargs).view(-1, 1)
                for batch_shape, sample_shape in itertools.product(
                    (torch.Size([]), torch.Size([2])),
                    (torch.Size([1]), torch.Size([3])),
                ):
                    means = torch.arange(n, **tkwargs).view(-1, 1)
                    if len(batch_shape) > 0:
                        view_means = means.view(1, *means.shape)
                        means = view_means.expand(batch_shape + means.shape)
                    if sample_shape[0] == 1:
                        samples = means.unsqueeze(0)
                    else:
                        samples = torch.stack([means, means + 1, means + 4], dim=0)
                    variances = torch.tensor(
                        [0.09, 0.25, 0.36, 0.25, 0.09], **tkwargs
                    ).view(-1, 1)
                    mm = MockModel(MockPosterior(mean=means, variance=variances))

                    # testing all feasible points
                    obj = samples.squeeze(-1)
                    constraints = [lambda samples: -torch.ones_like(samples[..., 0])]
                    best_f = compute_best_feasible_objective(
                        samples=samples, obj=obj, constraints=constraints
                    )
                    self.assertAllClose(best_f, obj.amax(dim=-1, keepdim=True))

                    # testing with some infeasible points
                    con_cutoff = 3.0
                    best_f = compute_best_feasible_objective(
                        samples=samples,
                        obj=obj,
                        constraints=[
                            lambda samples: samples[..., 0] - (con_cutoff + 1 / 2)
                        ],
                        model=mm,
                        X_baseline=X,
                    )

                    if sample_shape[0] == 3:
                        # under some samples, all baseline points are infeasible, so
                        # the best_f is set to the negative infeasible cost for
                        # for samples where no point is feasible
                        expected_best_f = torch.tensor(
                            [
                                3.0,
                                3.0,
                                -get_infeasible_cost(
                                    X=X,
                                    model=mm,
                                ).item(),
                            ],
                            **tkwargs,
                        ).view(-1, 1)
                        if len(batch_shape) > 0:
                            expected_best_f = expected_best_f.unsqueeze(1).repeat(
                                *[1] * len(sample_shape), *batch_shape, 1
                            )
                    else:
                        expected_best_f = torch.full(
                            sample_shape + batch_shape + torch.Size([1]),
                            con_cutoff,
                            **tkwargs,
                        )
                    self.assertAllClose(best_f, expected_best_f)
                    # test some feasible points with infeasible obi
                    if sample_shape[0] == 3:
                        best_f = compute_best_feasible_objective(
                            samples=samples,
                            obj=obj,
                            constraints=[
                                lambda samples: samples[..., 0] - (con_cutoff + 1 / 2)
                            ],
                            infeasible_obj=torch.ones(1, **tkwargs),
                        )
                        expected_best_f[-1] = 1
                        self.assertAllClose(best_f, expected_best_f)

                    # testing with no feasible points and infeasible obj
                    infeasible_obj = torch.tensor(torch.pi, **tkwargs)
                    expected_best_f = torch.full(
                        sample_shape + batch_shape + torch.Size([1]),
                        torch.pi,
                        **tkwargs,
                    )

                    best_f = compute_best_feasible_objective(
                        samples=samples,
                        obj=obj,
                        constraints=[lambda X: torch.ones_like(X[..., 0])],
                        infeasible_obj=infeasible_obj,
                    )
                    self.assertAllClose(best_f, expected_best_f)

                    # testing with no feasible points and not infeasible obj
                    def objective(Y, X):
                        return Y.squeeze(-1) - 5.0

                    best_f = compute_best_feasible_objective(
                        samples=samples,
                        obj=obj,
                        constraints=[lambda X: torch.ones_like(X[..., 0])],
                        model=mm,
                        X_baseline=X,
                        objective=objective,
                    )
                    expected_best_f = torch.full(
                        sample_shape + batch_shape + torch.Size([1]),
                        -get_infeasible_cost(X=X, model=mm, objective=objective).item(),
                        **tkwargs,
                    )
                    self.assertAllClose(best_f, expected_best_f)

                    with self.assertRaisesRegex(ValueError, "Must specify `model`"):
                        best_f = compute_best_feasible_objective(
                            samples=means,
                            obj=obj,
                            constraints=[lambda X: torch.ones_like(X[..., 0])],
                            X_baseline=X,
                        )
                    with self.assertRaisesRegex(
                        ValueError, "Must specify `X_baseline`"
                    ):
                        best_f = compute_best_feasible_objective(
                            samples=means,
                            obj=obj,
                            constraints=[lambda X: torch.ones_like(X[..., 0])],
                            model=mm,
                        )

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
