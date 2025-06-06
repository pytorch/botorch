# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

import torch
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.acquisition.multi_objective.objective import (
    IdentityMCMultiOutputObjective,
    WeightedMCMultiOutputObjective,
)
from botorch.acquisition.multi_objective.parego import qLogNParEGO
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model import Model
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.utils.testing import BotorchTestCase


class TestqLogNParEGO(BotorchTestCase):
    def base_test_parego(
        self,
        with_constraints: bool = False,
        with_scalarization_weights: bool = False,
        with_objective: bool = False,
        model: Model | None = None,
        incremental: bool = True,
    ) -> None:
        if with_constraints:
            assert with_objective, "Objective must be specified if constraints are."
        tkwargs: dict[str, Any] = {"device": self.device, "dtype": torch.double}
        num_objectives = 2
        num_constraints = 1 if with_constraints else 0
        num_outputs = num_objectives + num_constraints
        model = model or SingleTaskGP(
            train_X=torch.rand(5, 2, **tkwargs),
            train_Y=torch.rand(5, num_outputs, **tkwargs),
        )
        scalarization_weights = (
            torch.rand(num_objectives, **tkwargs)
            if with_scalarization_weights
            else None
        )
        objective = (
            WeightedMCMultiOutputObjective(
                weights=torch.tensor([2.0, -0.5], **tkwargs), outcomes=[0, 1]
            )
            if with_objective
            else None
        )
        constraints = [lambda samples: samples[..., -1]] if with_constraints else None
        acqf = qLogNParEGO(
            model=model,
            X_baseline=torch.rand(3, 2, **tkwargs),
            scalarization_weights=scalarization_weights,
            objective=objective,
            constraints=constraints,
            prune_baseline=True,
            incremental=incremental,
        )
        self.assertEqual(acqf.Y_baseline.shape, torch.Size([3, 2]))
        # Scalarization weights should be set if given and sampled otherwise.
        if scalarization_weights is not None:
            self.assertIs(acqf.scalarization_weights, scalarization_weights)
        else:
            self.assertEqual(
                acqf.scalarization_weights.shape, torch.Size([num_objectives])
            )
            # Should sum to 1 since they're sampled from simplex.
            self.assertAlmostEqual(acqf.scalarization_weights.sum().item(), 1.0)
        # Original objective should default to identity.
        if with_objective:
            self.assertIs(acqf._org_objective, objective)
        else:
            self.assertIsInstance(acqf._org_objective, IdentityMCMultiOutputObjective)
        # Acqf objective should be the chebyshev scalarization compounded
        # with the original objective.
        test_samples = torch.rand(32, 5, num_outputs, **tkwargs)
        expected_objective = acqf.chebyshev_scalarization(
            acqf._org_objective(test_samples)
        )
        self.assertEqual(expected_objective.shape, torch.Size([32, 5]))
        self.assertAllClose(acqf.objective(test_samples), expected_objective)
        # Evaluate the acquisition function.
        self.assertEqual(acqf(torch.rand(5, 2, **tkwargs)).shape, torch.Size([1]))
        test_X = torch.rand(32, 5, 2, **tkwargs)
        acqf_val = acqf(test_X)
        self.assertEqual(acqf_val.shape, torch.Size([32]))
        # Check that we're indeed using qLogNEI.
        self.assertIs(
            acqf.forward.__code__, qLogNoisyExpectedImprovement.forward.__code__
        )
        self.assertAllClose(
            acqf_val, qLogNoisyExpectedImprovement.forward(acqf, X=test_X)
        )

    def test_parego_simple(self) -> None:
        self.base_test_parego()

    def test_parego_with_constraints_objective_weights(self) -> None:
        self.base_test_parego(
            with_constraints=True, with_objective=True, with_scalarization_weights=True
        )

    def test_parego_with_non_incremental_ei(self) -> None:
        self.base_test_parego(incremental=False)

    def test_parego_with_ensemble_model(self) -> None:
        tkwargs: dict[str, Any] = {"device": self.device, "dtype": torch.double}
        models = []
        for _ in range(2):
            model = SaasFullyBayesianSingleTaskGP(
                train_X=torch.rand(5, 2, **tkwargs),
                train_Y=torch.randn(5, 1, **tkwargs),
                train_Yvar=torch.rand(5, 1, **tkwargs) * 0.05,
            )
            mcmc_samples = {
                "lengthscale": torch.rand(4, 1, 2, **tkwargs),
                "outputscale": torch.rand(4, **tkwargs),
                "mean": torch.randn(4, **tkwargs),
            }
            model.load_mcmc_samples(mcmc_samples)
            models.append(model)
        self.base_test_parego(model=ModelListGP(*models))
