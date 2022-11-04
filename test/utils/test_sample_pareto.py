#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from itertools import product

import numpy as np
import torch
from botorch.exceptions.errors import UnsupportedError
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from botorch.utils.gp_sampling import get_gp_samples
from botorch.utils.multi_objective import is_non_dominated
from botorch.utils.multi_objective.hypervolume import infer_reference_point
from botorch.utils.sample_pareto import pareto_solver, sample_pareto_sets_and_fronts
from botorch.utils.testing import BotorchTestCase


def get_model(
    dtype,
    device,
    num_points,
    input_dim,
    num_objectives,
    use_model_list,
    standardize_model,
):
    torch.manual_seed(123)
    tkwargs = {"dtype": dtype, "device": device}
    train_X = torch.rand(num_points, input_dim, **tkwargs)
    train_Y = torch.rand(num_points, num_objectives, **tkwargs)

    if standardize_model:
        if use_model_list:
            outcome_transform = Standardize(m=1)
        else:
            outcome_transform = Standardize(m=num_objectives)
    else:
        outcome_transform = None

    if use_model_list and num_objectives > 1:
        model = ModelListGP(
            *[
                SingleTaskGP(
                    train_X=train_X,
                    train_Y=train_Y[:, i : i + 1],
                    outcome_transform=outcome_transform,
                )
                for i in range(num_objectives)
            ]
        )
    else:
        model = SingleTaskGP(
            train_X=train_X,
            train_Y=train_Y,
            outcome_transform=outcome_transform,
        )

    return model.eval(), train_X, train_Y


class TestParetoSampling(BotorchTestCase):
    def test_pareto_solver(self):
        torch.manual_seed(1)
        np.random.seed(1)
        input_dim = 3
        num_points = 5
        tkwargs = {"device": self.device}
        genetic_kwargs = {
            "num_generations": 10,
            "pop_size": 20,
            "num_offsprings": 5,
        }
        for (
            dtype,
            maximize,
            num_objectives,
            use_model_list,
            standardize_model,
        ) in product(
            (torch.float, torch.double),
            (True, False),
            (1, 2),
            (False, True),
            (False, True),
        ):
            tkwargs["dtype"] = dtype

            model, X, Y = get_model(
                num_points=num_points,
                input_dim=input_dim,
                num_objectives=num_objectives,
                use_model_list=use_model_list,
                standardize_model=standardize_model,
                **tkwargs,
            )

            model_sample = get_gp_samples(
                model=model,
                num_outputs=num_objectives,
                n_samples=1,
            )

            input_dim = X.shape[-1]
            # fake bounds
            bounds = torch.zeros((2, input_dim), **tkwargs)
            bounds[1] = 1.0

            pareto_set, pareto_front = pareto_solver(
                model=model_sample,
                bounds=bounds,
                num_objectives=num_objectives,
                maximize=maximize,
                **genetic_kwargs,
            )

            # check shape
            self.assertTrue(pareto_set.ndim == 2)
            self.assertTrue(pareto_front.ndim == 2)
            self.assertTrue(pareto_set.shape[-1] == X.shape[-1])
            self.assertTrue(pareto_front.shape[-1] == Y.shape[-1])
            self.assertTrue(pareto_front.shape[-2] == pareto_set.shape[-2])
            num_optimal_points = pareto_front.shape[-2]

            # check if samples are non-dominated
            weight = 1.0 if maximize else -1.0
            count = torch.sum(is_non_dominated(Y=weight * pareto_front))
            self.assertTrue(count == num_optimal_points)

    def test_sample_pareto_sets_and_fronts(self):
        torch.manual_seed(1)
        np.random.seed(1)
        input_dim = 3
        num_points = 5
        tkwargs = {"device": self.device}
        genetic_kwargs = {
            "num_generations": 10,
            "pop_size": 20,
            "num_offsprings": 5,
        }

        for dtype, maximize, use_model_list, standardize_model in product(
            (torch.float, torch.double), (True, False), (False, True), (False, True)
        ):
            tkwargs["dtype"] = dtype

            num_pareto_samples = 2

            # test single-objective setting
            num_objectives = 1
            model, X, Y = get_model(
                num_points=num_points,
                input_dim=input_dim,
                num_objectives=num_objectives,
                use_model_list=use_model_list,
                standardize_model=standardize_model,
                **tkwargs,
            )

            input_dim = X.shape[-1]
            bounds = torch.zeros((2, input_dim), **tkwargs)
            bounds[1] = 1.0

            num_pareto_points_list = [1, 2, 1]
            num_greedy_list = [1, 0, 0]

            for i in range(len(num_pareto_points_list)):
                num_pareto_points = num_pareto_points_list[i]
                num_greedy = num_greedy_list[i]

                # should be an error when `num_greedy > 0`
                # should be an error when `num_pareto_points > 1`
                if i in (0, 1):
                    with self.assertRaises(UnsupportedError):
                        sample_pareto_sets_and_fronts(
                            model=model,
                            bounds=bounds,
                            num_pareto_samples=num_pareto_samples,
                            num_pareto_points=num_pareto_points,
                            maximize=maximize,
                            num_greedy=num_greedy,
                            **genetic_kwargs,
                        )
                else:
                    pareto_sets, pareto_fronts = sample_pareto_sets_and_fronts(
                        model=model,
                        bounds=bounds,
                        num_pareto_samples=num_pareto_samples,
                        num_pareto_points=num_pareto_points,
                        maximize=maximize,
                        num_greedy=num_greedy,
                        **genetic_kwargs,
                    )

                    # check shape
                    ps_desired_shape = torch.Size(
                        [num_pareto_samples, num_pareto_points, input_dim]
                    )
                    pf_desired_shape = torch.Size(
                        [num_pareto_samples, num_pareto_points, num_objectives]
                    )

                    self.assertTrue(pareto_sets.shape == ps_desired_shape)
                    self.assertTrue(pareto_fronts.shape == pf_desired_shape)

            # test multi-objective setting
            num_objectives = 2
            model, X, Y = get_model(
                num_points=num_points,
                input_dim=input_dim,
                num_objectives=num_objectives,
                use_model_list=use_model_list,
                standardize_model=standardize_model,
                **tkwargs,
            )
            num_objectives = Y.shape[-1]
            ref_point = infer_reference_point(Y)

            # test basic usage with no greedily selected points
            num_pareto_points_list = [1, 1, 1]
            num_greedy_list = [0, 1, 2]

            # when `num_greedy > num_pareto_points`, we sample all
            # `num_pareto_points` points greedily.

            for i in range(len(num_pareto_points_list)):
                num_pareto_points = num_pareto_points_list[i]
                num_greedy = num_greedy_list[i]

                pareto_sets, pareto_fronts = sample_pareto_sets_and_fronts(
                    model=model,
                    bounds=bounds,
                    num_pareto_samples=num_pareto_samples,
                    num_pareto_points=num_pareto_points,
                    maximize=maximize,
                    num_greedy=num_greedy,
                    X_baseline=X,
                    ref_point=ref_point,
                    **genetic_kwargs,
                )

                # check shape
                ps_desired_shape = torch.Size(
                    [num_pareto_samples, num_pareto_points, input_dim]
                )
                pf_desired_shape = torch.Size(
                    [num_pareto_samples, num_pareto_points, num_objectives]
                )

                self.assertTrue(pareto_sets.shape == ps_desired_shape)
                self.assertTrue(pareto_fronts.shape == pf_desired_shape)

            # Not specifying the baseline inputs and reference point.
            with self.assertRaises(UnsupportedError):
                sample_pareto_sets_and_fronts(
                    model=model,
                    bounds=bounds,
                    num_pareto_samples=num_pareto_samples,
                    num_pareto_points=num_pareto_points,
                    maximize=maximize,
                    num_greedy=num_greedy,
                    **genetic_kwargs,
                )

            # We cannot generate more Pareto optimal points than the population
            # size of the genetic solver.
            with self.assertRaises(RuntimeError):
                num_pareto_points = 6
                genetic_kwargs = {
                    "num_generations": 5,
                    "pop_size": 5,
                    "num_offsprings": 1,
                }
                sample_pareto_sets_and_fronts(
                    model=model,
                    bounds=bounds,
                    num_pareto_samples=num_pareto_samples,
                    num_pareto_points=num_pareto_points,
                    maximize=maximize,
                    **genetic_kwargs,
                )
