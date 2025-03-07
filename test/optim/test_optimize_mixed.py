#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from itertools import product
from typing import Any, Callable
from unittest import mock

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.exceptions.errors import CandidateGenerationError, UnsupportedError
from botorch.exceptions.warnings import OptimizationWarning
from botorch.generation.gen import gen_candidates_scipy
from botorch.models.deterministic import DeterministicModel
from botorch.models.gp_regression import SingleTaskGP
from botorch.optim.optimize import _optimize_acqf, OptimizeAcqfInputs
from botorch.optim.optimize_mixed import (
    _setup_continuous_relaxation,
    complement_indices,
    continuous_step,
    discrete_step,
    generate_starting_points,
    get_nearest_neighbors,
    get_spray_points,
    MAX_DISCRETE_VALUES,
    optimize_acqf_mixed_alternating,
    sample_feasible_points,
)
from botorch.utils.testing import BotorchTestCase, MockAcquisitionFunction
from pyre_extensions import assert_is_instance
from torch import Tensor

OPT_MODULE = f"{optimize_acqf_mixed_alternating.__module__}"


def _make_opt_inputs(
    acq_function: AcquisitionFunction,
    bounds: Tensor,
    q: int = 1,
    num_restarts: int = 20,
    raw_samples: int | None = 1024,
    options: dict[str, bool | float | int | str] | None = None,
    inequality_constraints: list[tuple[Tensor, Tensor, float]] | None = None,
    equality_constraints: list[tuple[Tensor, Tensor, float]] | None = None,
    nonlinear_inequality_constraints: list[tuple[Callable, bool]] | None = None,
    fixed_features: dict[int, float] | None = None,
) -> OptimizeAcqfInputs:
    r"""Helper to construct `OptimizeAcqfInputs` from limited inputs."""
    return OptimizeAcqfInputs(
        acq_function=acq_function,
        bounds=bounds,
        q=q,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        options=options or {},
        inequality_constraints=inequality_constraints,
        equality_constraints=equality_constraints,
        nonlinear_inequality_constraints=nonlinear_inequality_constraints,
        fixed_features=fixed_features or {},
        post_processing_func=None,
        batch_initial_conditions=None,
        return_best_only=True,
        gen_candidates=gen_candidates_scipy,
        sequential=True,
    )


def get_hamming_neighbors(x_discrete: Tensor) -> Tensor:
    r"""Generate all 1-Hamming distance neighbors of a binary input."""
    aye = torch.eye(
        x_discrete.shape[-1], dtype=x_discrete.dtype, device=x_discrete.device
    )
    X_loc = (x_discrete - aye).abs()
    return X_loc


class QuadraticDeterministicModel(DeterministicModel):
    """A simple quadratic model for testing."""

    def __init__(self, root: Tensor):
        """Initialize the model with the given root."""
        super().__init__()
        self.register_buffer("root", root)
        self._num_outputs = 1

    def forward(self, X: Tensor):
        # `keepdim`` is necessary for optimize_acqf to work correctly.
        return -(X - self.root).square().sum(dim=-1, keepdim=True)


class TestOptimizeAcqfMixed(BotorchTestCase):
    def setUp(self):
        super().setUp()
        self.single_bound = torch.tensor([[0.0], [1.0]], device=self.device)
        self.tkwargs: dict[str, Any] = {"device": self.device, "dtype": torch.double}

    def _get_random_binary(self, d: int, k: int) -> Tensor:
        """d: dimensionality of vector, k: number of ones."""
        X = torch.zeros(d, device=self.device)
        X[:k] = 1
        return X[torch.randperm(d, device=self.device)]

    def _get_data(self) -> tuple[Tensor, Tensor, list[int], list[int]]:
        with torch.random.fork_rng():
            torch.manual_seed(0)
            binary_dims, cont_dims, dim = [0, 3, 4], [1, 2], 5
            train_X = torch.rand(3, dim, **self.tkwargs)
            train_X[:, binary_dims] = train_X[:, binary_dims].round()
            train_Y = train_X.sin().sum(dim=-1).unsqueeze(-1)
        return train_X, train_Y, binary_dims, cont_dims

    def test_get_nearest_neighbors(self) -> None:
        # For binary inputs, this should be equivalent to get_hamming_neighbors,
        # with potentially different ordering of the outputs.
        current_x = self._get_random_binary(16, 7)
        bounds = self.single_bound.repeat(1, 16)
        discrete_dims = torch.arange(16, dtype=torch.long, device=self.device)
        self.assertTrue(
            torch.equal(
                get_nearest_neighbors(
                    current_x=current_x, bounds=bounds, discrete_dims=discrete_dims
                )
                .sort(dim=0)
                .values,
                get_hamming_neighbors(current_x).sort(dim=0).values,
            )
        )
        # Test with integer and continuous inputs.
        current_x = torch.tensor([1.0, 0.0, 0.5], device=self.device)
        bounds = torch.tensor([[0.0, 0.0, 0.0], [3.0, 2.0, 1.0]], device=self.device)
        discrete_dims = torch.tensor([0, 1], device=self.device)
        expected_neighbors = torch.tensor(
            [[0.0, 0.0, 0.5], [2.0, 0.0, 0.5], [1.0, 1.0, 0.5]], device=self.device
        )
        self.assertTrue(
            torch.equal(
                expected_neighbors.sort(dim=0).values,
                get_nearest_neighbors(
                    current_x=current_x, bounds=bounds, discrete_dims=discrete_dims
                )
                .sort(dim=0)
                .values,
            )
        )

    def test_sample_feasible_points(self, with_constraints: bool = False) -> None:
        bounds = torch.tensor([[0.0, 2.0, 0.0], [1.0, 5.0, 1.0]], **self.tkwargs)
        opt_inputs = _make_opt_inputs(
            acq_function=MockAcquisitionFunction(),
            bounds=bounds,
            fixed_features={0: 0.5},
            inequality_constraints=(
                [
                    (  # X[1] >= 4.0
                        torch.tensor([1], device=self.device),
                        torch.tensor([1.0], **self.tkwargs),
                        4.0,
                    )
                ]
                if with_constraints
                else None
            ),
        )
        # Check for error if feasible points cannot be found.
        with self.assertRaisesRegex(
            CandidateGenerationError, "Could not generate"
        ), mock.patch(
            f"{OPT_MODULE}._filter_infeasible",
            return_value=torch.empty(0, 3, **self.tkwargs),
        ):
            sample_feasible_points(
                opt_inputs=opt_inputs,
                discrete_dims=torch.tensor([0, 2], device=self.device),
                num_points=10,
            )
        # Generate a number of points.
        X = sample_feasible_points(
            opt_inputs=opt_inputs,
            discrete_dims=torch.tensor([1], device=self.device),
            num_points=10,
        )
        self.assertEqual(X.shape, torch.Size([10, 3]))
        self.assertTrue(torch.all(X[..., 0] == 0.5))
        if with_constraints:
            self.assertTrue(torch.all(X[..., 1] >= 4.0))
        self.assertAllClose(X[..., 1], X[..., 1].round())

    def test_sample_feasible_points_with_constraints(self) -> None:
        self.test_sample_feasible_points(with_constraints=True)

    def test_discrete_step(self):
        d = 16
        bounds = self.single_bound.repeat(1, d)
        root = torch.zeros(d, device=self.device)
        model = QuadraticDeterministicModel(root)
        k = 7  # number of ones
        X = self._get_random_binary(d, k)
        best_f = model(X)
        ei = ExpectedImprovement(model, best_f=best_f)

        # this just tests that the quadratic model + ei works correctly
        ei_x_none = ei(X[None])
        self.assertAllClose(ei_x_none, torch.zeros_like(ei_x_none), atol=1e-3)
        self.assertGreaterEqual(ei_x_none.min(), 0.0)
        ei_root_none = ei(root[None])
        self.assertAllClose(ei_root_none, torch.full_like(ei_root_none, k))
        self.assertGreaterEqual(ei_root_none.min(), 0.0)

        # each discrete step should reduce the best_f value by exactly 1
        binary_dims = torch.arange(d)
        for i in range(k):
            X, ei_val = discrete_step(
                opt_inputs=_make_opt_inputs(
                    acq_function=ei,
                    bounds=bounds,
                    options={"maxiter_discrete": 1, "tol": 0, "init_batch_limit": 32},
                ),
                discrete_dims=binary_dims,
                current_x=X,
            )
            ei_x_none = ei(X[None])
            self.assertAllClose(ei_x_none, torch.full_like(ei_x_none, i + 1))
            self.assertGreaterEqual(ei_x_none.min(), 0.0)

        self.assertAllClose(X, root)

        # Test with integer variables.
        bounds[1, :2] = 2.0
        X = self._get_random_binary(d, k)
        for i in range(k):
            X, ei_val = discrete_step(
                opt_inputs=_make_opt_inputs(
                    acq_function=ei,
                    bounds=bounds,
                    options={"maxiter_discrete": 1, "tol": 0, "init_batch_limit": 2},
                ),
                discrete_dims=binary_dims,
                current_x=X,
            )
            ei_x_none = ei(X[None])
            self.assertAllClose(ei_x_none, torch.full_like(ei_x_none, i + 1))

        self.assertAllClose(X, root)

        # Testing that convergence_tol exits early.
        X = self._get_random_binary(d, k)
        X_clone = X.clone()
        # Setting convergence_tol to above one should ensure that we only take one step.
        mock_acqf = mock.MagicMock(wraps=ei)
        discrete_step(
            opt_inputs=_make_opt_inputs(
                acq_function=mock_acqf,
                bounds=bounds,
                options={"maxiter_discrete": 1, "tol": 1.5},
            ),
            discrete_dims=binary_dims,
            current_x=X_clone,
        )
        # One call when entering, one call in the loop.
        self.assertEqual(mock_acqf.call_count, 2)

        # Test that no steps are taken if there's no improvement.
        mock_acqf = mock.MagicMock(
            side_effect=lambda x: torch.zeros(
                x.shape[:-1], device=x.device, dtype=x.dtype
            )
        )
        X_clone, _ = discrete_step(
            opt_inputs=_make_opt_inputs(
                acq_function=mock_acqf,
                bounds=bounds,
                options={"maxiter_discrete": 1, "tol": 1.5, "init_batch_limit": 2},
            ),
            discrete_dims=binary_dims,
            current_x=X_clone,
        )
        self.assertAllClose(X_clone, X)

        # test with fixed continuous dimensions
        X = self._get_random_binary(d, k)
        X[:2] = 1.0  # To satisfy the constraint.
        k = int(X.sum().item())
        X_cont = torch.rand(3, device=self.device)
        X = torch.cat((X, X_cont))  # appended continuous dimensions

        root = torch.zeros(d + 3, device=self.device)
        bounds = self.single_bound.repeat(1, d + 3)
        model = QuadraticDeterministicModel(root)
        best_f = model(X)
        ei = ExpectedImprovement(model, best_f=best_f)
        for i in range(k - 2):
            X, ei_val = discrete_step(
                opt_inputs=_make_opt_inputs(
                    acq_function=ei,
                    bounds=bounds,
                    options={"maxiter_discrete": 1, "tol": 0, "init_batch_limit": 2},
                    inequality_constraints=[
                        (  # X[..., 0] + X[..., 1] >= 2.
                            torch.arange(2, dtype=torch.long, device=self.device),
                            torch.ones(2, device=self.device),
                            2.0,
                        )
                    ],
                ),
                discrete_dims=binary_dims,
                current_x=X,
            )
            self.assertAllClose(ei_val, torch.full_like(ei_val, i + 1))
        self.assertAllClose(
            X[:2], torch.ones(2, device=self.device)
        )  # satisfies constraints.
        self.assertAllClose(X[2:d], root[2:d])  # binary optimized
        self.assertAllClose(X[d:], X_cont)  # continuous unchanged

        # Test with super-tight constraint.
        X = torch.ones(d + 3, device=self.device)
        X_new, _ = discrete_step(
            opt_inputs=_make_opt_inputs(
                acq_function=ei,
                bounds=bounds,
                inequality_constraints=[  # sum(X) >= d + 3
                    (
                        torch.arange(d + 3, dtype=torch.long, device=self.device),
                        torch.ones(d + 3, device=self.device),
                        d + 3,
                    )
                ],
            ),
            discrete_dims=binary_dims,
            current_x=X,
        )
        # No feasible neighbors, so we should get the same point back.
        self.assertAllClose(X_new, X)

    def test_continuous_step(self):
        d_cont = 16
        d_bin = 5
        d = d_cont + d_bin
        bounds = self.single_bound.repeat(1, d)

        root = torch.rand(d, device=self.device)
        model = QuadraticDeterministicModel(root)

        indices = torch.randperm(d, device=self.device)
        binary_dims = indices[:d_bin]
        cont_dims = indices[d_bin:]

        X = torch.zeros(d, device=self.device)
        k = 7  # number of ones in binary vector
        X[binary_dims] = self._get_random_binary(d_bin, k)
        X[cont_dims] = torch.rand(d_cont, device=self.device)

        best_f = model(X)
        ei = ExpectedImprovement(model, best_f=best_f)
        X_new, ei_val = continuous_step(
            opt_inputs=_make_opt_inputs(
                acq_function=ei,
                bounds=bounds,
                options={"maxiter_continuous": 32},
            ),
            discrete_dims=binary_dims,
            current_x=X.clone(),
        )
        self.assertAllClose(X_new[cont_dims], root[cont_dims])
        self.assertAllClose(X_new[binary_dims], X[binary_dims])

        # Test with fixed features and constraints.
        fixed_binary = int(binary_dims[0])
        # We don't want fixed cont to be one of the first two indices,
        # to avoid it being a part of the constraint. This ensures that.
        # The fixed value of 0.5 cannot satisfy the constraint.
        fixed_cont = int(cont_dims[:3].max())
        X_ = X.clone()
        X_[:2] = 1.0  # To satisfy the constraint.
        X_new, ei_val = continuous_step(
            opt_inputs=_make_opt_inputs(
                acq_function=ei,
                bounds=bounds,
                options={"maxiter_continuous": 32},
                fixed_features={fixed_binary: 1, fixed_cont: 0.5},
                inequality_constraints=[
                    (  # X[..., 0] + X[..., 1] >= 2.
                        torch.tensor([0, 1], device=self.device),
                        torch.ones(2, device=self.device),
                        2.0,
                    )
                ],
            ),
            discrete_dims=binary_dims,
            current_x=X_,
        )
        self.assertTrue(
            torch.equal(
                X_new[[fixed_binary, fixed_cont]],
                torch.tensor([1.0, 0.5], device=self.device),
            )
        )
        self.assertAllClose(X_new[:2], X_[:2])

        # test edge case when all parameters are binary
        root = torch.rand(d_bin)
        model = QuadraticDeterministicModel(root)
        ei = ExpectedImprovement(model, best_f=best_f)
        X = self._get_random_binary(d_bin, k)
        bounds = self.single_bound.repeat(1, d_bin)
        binary_dims = torch.arange(d_bin)
        X_out, ei_val = continuous_step(
            opt_inputs=_make_opt_inputs(
                acq_function=ei,
                bounds=bounds,
                options={"maxiter_continuous": 32},
            ),
            discrete_dims=binary_dims,
            current_x=X,
        )
        self.assertTrue(X is X_out)  # testing pointer equality for due to short cut
        self.assertAllClose(ei_val, ei(X[None]))

    def test_optimize_acqf_mixed_binary_only(self) -> None:
        train_X, train_Y, binary_dims, cont_dims = self._get_data()
        dim = len(binary_dims) + len(cont_dims)
        bounds = self.single_bound.repeat(1, dim)
        torch.manual_seed(0)
        model = SingleTaskGP(train_X=train_X, train_Y=train_Y)
        acqf = ExpectedImprovement(model=model, best_f=torch.max(train_Y))
        options = {
            "initialization_strategy": "random",
            "maxiter_alternating": 2,
            "maxiter_discrete": 8,
            "maxiter_continuous": 32,
            "num_spray_points": 32,
            "std_cont_perturbation": 1e-2,
        }
        X_baseline = train_X[torch.argmax(train_Y)].unsqueeze(0)

        # testing spray points
        perturb_nbors = get_spray_points(
            X_baseline=X_baseline,
            discrete_dims=binary_dims,
            cont_dims=cont_dims,
            bounds=bounds,
            num_spray_points=assert_is_instance(options["num_spray_points"], int),
        )
        self.assertEqual(perturb_nbors.shape, (options["num_spray_points"], dim))
        # get single candidate
        candidates, _ = optimize_acqf_mixed_alternating(
            acq_function=acqf,
            bounds=bounds,
            discrete_dims=binary_dims,
            options=options,
            q=1,
            raw_samples=32,
            num_restarts=2,
        )
        self.assertEqual(candidates.shape[-1], dim)
        c_binary = candidates[:, binary_dims]
        self.assertTrue(((c_binary == 0) | (c_binary == 1)).all())

        # testing that continuous perturbations lead to lower acquisition values
        std_pert = 1e-2
        perturbed_candidates = candidates.clone()
        perturbed_candidates[..., cont_dims] += std_pert * torch.randn_like(
            perturbed_candidates[..., cont_dims], device=self.device
        )
        perturbed_candidates.clamp_(0, 1)
        # Needs a loose tolerance to avoid flakiness
        self.assertLess((acqf(perturbed_candidates) - acqf(candidates)).max(), 0.0)

        # testing that a discrete perturbation leads to a lower acquisition values
        for i in binary_dims:
            perturbed_candidates = candidates.clone()
            perturbed_candidates[..., i] = 0 if perturbed_candidates[..., i] == 1 else 1
            self.assertLess((acqf(perturbed_candidates) - acqf(candidates)).max(), 0.0)

        # get multiple candidates
        root = torch.zeros(dim, device=self.device)
        model = QuadraticDeterministicModel(root)
        acqf = qLogNoisyExpectedImprovement(
            model=model, X_baseline=train_X, prune_baseline=False
        )
        options["initialization_strategy"] = "equally_spaced"
        candidates, _ = optimize_acqf_mixed_alternating(
            acq_function=acqf,
            bounds=bounds,
            discrete_dims=binary_dims,
            options=options,
            q=3,
            raw_samples=32,
            num_restarts=2,
        )
        self.assertEqual(candidates.shape, torch.Size([3, dim]))
        c_binary = candidates[:, binary_dims]
        self.assertTrue(((c_binary == 0) | (c_binary == 1)).all())

        # testing that continuous perturbations lead to lower acquisition values
        perturbed_candidates = candidates.clone()
        perturbed_candidates[..., cont_dims] += std_pert * torch.randn_like(
            perturbed_candidates[..., cont_dims], device=self.device
        )
        # need to project continuous variables into [0, 1] for test to work
        # since binaries are in [0, 1] too, we can clamp the entire tensor
        perturbed_candidates.clamp_(0, 1)
        self.assertLess((acqf(perturbed_candidates) - acqf(candidates)).max(), 0.0)

        # testing that any bit flip leads to a lower acquisition values
        for i in binary_dims:
            perturbed_candidates = candidates.clone()
            perturbed_candidates[..., i] = torch.where(
                perturbed_candidates[..., i] == 1, 0, 1
            )
            self.assertLess((acqf(perturbed_candidates) - acqf(candidates)).max(), 0.0)

        # Test only using one continuous variable
        cont_dims = [1]
        binary_dims = complement_indices(cont_dims, dim)
        X_baseline[:, binary_dims] = X_baseline[:, binary_dims].round()
        candidates, _ = optimize_acqf_mixed_alternating(
            acq_function=acqf,
            bounds=bounds,
            discrete_dims=binary_dims,
            options=options,
            q=1,
            raw_samples=20,
            num_restarts=2,
            post_processing_func=lambda x: x,
        )
        self.assertEqual(candidates.shape[-1], dim)
        c_binary = candidates[:, binary_dims + [2]]
        self.assertTrue(((c_binary == 0) | (c_binary == 1)).all())
        # Only continuous parameters should fallback to optimize_acqf.
        with mock.patch(
            f"{OPT_MODULE}._optimize_acqf", wraps=_optimize_acqf
        ) as wrapped_optimize:
            optimize_acqf_mixed_alternating(
                acq_function=acqf,
                bounds=bounds,
                discrete_dims=[],
                options=options,
                q=1,
                raw_samples=20,
                num_restarts=2,
            )
        wrapped_optimize.assert_called_once_with(
            opt_inputs=_make_opt_inputs(
                acq_function=acqf,
                bounds=bounds,
                options=options,
                q=1,
                raw_samples=20,
                num_restarts=2,
            )
        )
        # Only discrete works fine.
        candidates, _ = optimize_acqf_mixed_alternating(
            acq_function=acqf,
            bounds=bounds,
            discrete_dims=list(range(dim)),
            options=options,
            q=1,
            raw_samples=20,
            num_restarts=20,
        )
        self.assertTrue(((candidates == 0) | (candidates == 1)).all())
        # Invalid indices will raise an error.
        with self.assertRaisesRegex(
            ValueError,
            "with unique integers between 0 and num_dims - 1",
        ):
            optimize_acqf_mixed_alternating(
                acq_function=acqf,
                bounds=bounds,
                discrete_dims=[-1],
                options=options,
                q=1,
                raw_samples=20,
                num_restarts=2,
            )

    def test_optimize_acqf_mixed_integer(self) -> None:
        # Testing with integer variables.
        train_X, train_Y, binary_dims, cont_dims = self._get_data()
        dim = len(binary_dims) + len(cont_dims)
        # Update the data to introduce integer dimensions.
        binary_dims = [0]
        integer_dims = [3, 4]
        discrete_dims = binary_dims + integer_dims
        bounds = self.single_bound.repeat(1, dim)
        bounds[1, 3:5] = 4.0
        # Update the model to have a different optimizer.
        root = torch.tensor([0.0, 0.0, 0.0, 4.0, 4.0], device=self.device)
        model = QuadraticDeterministicModel(root)
        acqf = qLogNoisyExpectedImprovement(model=model, X_baseline=train_X)
        with mock.patch(
            f"{OPT_MODULE}._optimize_acqf", wraps=_optimize_acqf
        ) as wrapped_optimize:
            candidates, _ = optimize_acqf_mixed_alternating(
                acq_function=acqf,
                bounds=bounds,
                discrete_dims=discrete_dims,
                q=3,
                raw_samples=32,
                num_restarts=4,
                options={
                    "batch_limit": 5,
                    "init_batch_limit": 20,
                    "maxiter_alternating": 1,
                },
            )
        self.assertEqual(candidates.shape, torch.Size([3, dim]))
        self.assertEqual(candidates.shape[-1], dim)
        c_binary = candidates[:, binary_dims]
        self.assertTrue(((c_binary == 0) | (c_binary == 1)).all())
        c_integer = candidates[:, integer_dims]
        self.assertTrue(torch.equal(c_integer, c_integer.round()))
        self.assertTrue((c_integer == 4.0).any())
        # Check that we used continuous relaxation for initialization.
        first_call_options = (
            wrapped_optimize.call_args_list[0].kwargs["opt_inputs"].options
        )
        self.assertEqual(
            first_call_options,
            {"maxiter": 100, "batch_limit": 5, "init_batch_limit": 20},
        )

        # Testing that continuous perturbations lead to lower acquisition values.
        perturbed_candidates = candidates.clone()
        perturbed_candidates[..., cont_dims] += 1e-2 * torch.randn_like(
            perturbed_candidates[..., cont_dims], device=self.device
        )
        perturbed_candidates[..., cont_dims].clamp_(0, 1)
        self.assertLess((acqf(perturbed_candidates) - acqf(candidates)).max(), 1e-12)
        # Testing that integer value change leads to a lower acquisition values.
        for i, j in product(integer_dims, range(3)):
            perturbed_candidates = candidates.repeat(2, 1, 1)
            perturbed_candidates[0, j, i] += 1.0
            perturbed_candidates[1, j, i] -= 1.0
            perturbed_candidates.clamp_(bounds[0], bounds[1])
            self.assertLess(
                (acqf(perturbed_candidates) - acqf(candidates)).max(), 1e-12
            )

        # Test gracious fallback when continuous relaxation fails.
        with mock.patch(
            f"{OPT_MODULE}._optimize_acqf",
            side_effect=RuntimeError,
        ), self.assertWarnsRegex(OptimizationWarning, "Failed to initialize"):
            candidates, _ = generate_starting_points(
                opt_inputs=_make_opt_inputs(
                    acq_function=acqf,
                    bounds=bounds,
                    raw_samples=32,
                    num_restarts=4,
                    options={"batch_limit": 2, "init_batch_limit": 2},
                ),
                discrete_dims=torch.tensor(discrete_dims, device=self.device),
                cont_dims=torch.tensor(cont_dims, device=self.device),
            )
        self.assertEqual(candidates.shape, torch.Size([4, dim]))

        # Test unsupported options.
        with self.assertRaisesRegex(UnsupportedError, "unsupported option"):
            optimize_acqf_mixed_alternating(
                acq_function=acqf,
                bounds=bounds,
                discrete_dims=discrete_dims,
                options={"invalid": 5, "init_batch_limit": 20},
            )

        # Test with fixed features and constraints. Using both discrete and continuous.
        constraint = (  # X[..., 0] + X[..., 1] >= 1.
            torch.tensor([0, 1], device=self.device),
            torch.ones(2, device=self.device),
            1.0,
        )
        candidates, _ = optimize_acqf_mixed_alternating(
            acq_function=acqf,
            bounds=bounds,
            discrete_dims=integer_dims,
            q=3,
            raw_samples=32,
            num_restarts=4,
            options={"batch_limit": 5, "init_batch_limit": 20},
            fixed_features={1: 0.5, 3: 2},
            inequality_constraints=[constraint],
        )
        self.assertAllClose(
            candidates[:, [0, 1, 3]],
            torch.tensor([0.5, 0.5, 2.0], device=self.device).repeat(3, 1),
        )

        # Test fallback when initializer cannot generate enough feasible points.
        with mock.patch(
            f"{OPT_MODULE}._optimize_acqf",
            return_value=(
                torch.zeros(4, 1, dim, **self.tkwargs),
                torch.zeros(4, **self.tkwargs),
            ),
        ), mock.patch(
            f"{OPT_MODULE}.sample_feasible_points", wraps=sample_feasible_points
        ) as wrapped_sample_feasible:
            generate_starting_points(
                opt_inputs=_make_opt_inputs(
                    acq_function=acqf,
                    bounds=bounds,
                    raw_samples=32,
                    num_restarts=4,
                    inequality_constraints=[constraint],
                ),
                discrete_dims=torch.tensor(discrete_dims, device=self.device),
                cont_dims=torch.tensor(cont_dims, device=self.device),
            )
        wrapped_sample_feasible.assert_called_once()
        # Should request 4 candidates, since all 4 are infeasible.
        self.assertEqual(wrapped_sample_feasible.call_args.kwargs["num_points"], 4)

    def test_optimize_acqf_mixed_continuous_relaxation(self) -> None:
        # Testing with integer variables.
        train_X, train_Y, binary_dims, cont_dims = self._get_data()
        # Update the data to introduce integer dimensions.
        binary_dims = [0]
        integer_dims = [3, 4]
        discrete_dims = binary_dims + integer_dims
        bounds = torch.tensor(
            [[0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 40.0, 15.0]],
            dtype=torch.double,
            device=self.device,
        )
        # Update the model to have a different optimizer.
        root = torch.tensor([0.0, 0.0, 0.0, 25.0, 10.0], device=self.device)
        model = QuadraticDeterministicModel(root)
        acqf = qLogNoisyExpectedImprovement(model=model, X_baseline=train_X)

        for max_discrete_values, post_processing_func in (
            (None, None),
            (5, lambda X: X + 10),
        ):
            options = {
                "batch_limit": 5,
                "init_batch_limit": 20,
                "maxiter_alternating": 1,
            }
            if max_discrete_values is not None:
                options["max_discrete_values"] = max_discrete_values
            with mock.patch(
                f"{OPT_MODULE}._setup_continuous_relaxation",
                wraps=_setup_continuous_relaxation,
            ) as wrapped_setup, mock.patch(
                f"{OPT_MODULE}.discrete_step", wraps=discrete_step
            ) as wrapped_discrete:
                candidates, _ = optimize_acqf_mixed_alternating(
                    acq_function=acqf,
                    bounds=bounds,
                    discrete_dims=discrete_dims,
                    q=3,
                    raw_samples=32,
                    num_restarts=4,
                    options=options,
                    post_processing_func=post_processing_func,
                )
            wrapped_setup.assert_called_once_with(
                discrete_dims=discrete_dims,
                bounds=bounds,
                max_discrete_values=max_discrete_values or MAX_DISCRETE_VALUES,
                post_processing_func=post_processing_func,
            )
            discrete_call_args = wrapped_discrete.call_args.kwargs
            expected_dims = [0, 4] if max_discrete_values is None else [0]
            self.assertAllClose(
                discrete_call_args["discrete_dims"],
                torch.tensor(expected_dims, device=self.device),
            )
            # Check that dim 3 is rounded.
            X = torch.ones(1, 5, device=self.device) * 0.6
            X_expected = X.clone()
            X_expected[0, 3] = 1.0
            if max_discrete_values is not None:
                X_expected[0, 4] = 1.0
            if post_processing_func is not None:
                X_expected = post_processing_func(X_expected)
            self.assertAllClose(
                discrete_call_args["opt_inputs"].post_processing_func(X), X_expected
            )
