#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
from botorch.utils import apply_constraints, get_objective_weights_transform
from botorch.utils.testing import BotorchTestCase
from torch import Tensor


def ones_f(samples: Tensor) -> Tensor:
    return torch.ones(samples.shape[0:-1], device=samples.device, dtype=samples.dtype)


def zeros_f(samples: Tensor) -> Tensor:
    return torch.zeros(samples.shape[0:-1], device=samples.device, dtype=samples.dtype)


def nonzeros_f(samples: Tensor) -> Tensor:
    t = torch.zeros(samples.shape[0:-1], device=samples.device, dtype=samples.dtype)
    t[:] = 0.1
    return t


def minus_one_f(samples: Tensor) -> Tensor:
    return -(
        torch.ones(samples.shape[0:-1], device=samples.device, dtype=samples.dtype)
    )


class TestApplyConstraints(BotorchTestCase):
    def test_apply_constraints(self):
        # nonnegative objective, one constraint
        samples = torch.randn(1)
        obj = ones_f(samples)
        obj = apply_constraints(
            obj=obj, constraints=[zeros_f], samples=samples, infeasible_cost=0.0
        )
        self.assertTrue(torch.equal(obj, ones_f(samples) * 0.5))
        # nonnegative objective, two constraint
        samples = torch.randn(1)
        obj = ones_f(samples)
        obj = apply_constraints(
            obj=obj,
            constraints=[zeros_f, zeros_f],
            samples=samples,
            infeasible_cost=0.0,
        )
        self.assertTrue(torch.equal(obj, ones_f(samples) * 0.5 * 0.5))
        # negative objective, one constraint, infeasible_cost
        samples = torch.randn(1)
        obj = minus_one_f(samples)
        obj = apply_constraints(
            obj=obj, constraints=[zeros_f], samples=samples, infeasible_cost=2.0
        )
        self.assertTrue(torch.equal(obj, ones_f(samples) * 0.5 - 2.0))

        # nonnegative objective, one constraint, eta = 0
        samples = torch.randn(1)
        obj = ones_f(samples)
        with self.assertRaises(ValueError):
            apply_constraints(
                obj=obj,
                constraints=[zeros_f],
                samples=samples,
                infeasible_cost=0.0,
                eta=0.0,
            )

    def test_apply_constraints_multi_output(self):
        # nonnegative objective, one constraint
        tkwargs = {"device": self.device}
        for dtype in (torch.float, torch.double):
            tkwargs["dtype"] = dtype
            samples = torch.rand(3, 2, **tkwargs)
            obj = samples.clone()
            obj = apply_constraints(
                obj=obj, constraints=[zeros_f], samples=samples, infeasible_cost=0.0
            )
            self.assertTrue(torch.equal(obj, samples * 0.5))
            # nonnegative objective, two constraint
            obj = samples.clone()
            obj = apply_constraints(
                obj=obj,
                constraints=[zeros_f, zeros_f],
                samples=samples,
                infeasible_cost=0.0,
            )
            self.assertTrue(torch.equal(obj, samples * 0.5 * 0.5))
            # nonnegative objective, two constraint explicit eta
            obj = samples.clone()
            obj = apply_constraints(
                obj=obj,
                constraints=[zeros_f, zeros_f],
                samples=samples,
                infeasible_cost=0.0,
                eta=torch.tensor([10e-3, 10e-3]).to(**tkwargs),
            )
            self.assertTrue(torch.equal(obj, samples * 0.5 * 0.5))
            # nonnegative objective, two constraint explicit different eta
            obj = samples.clone()
            obj = apply_constraints(
                obj=obj,
                constraints=[nonzeros_f, nonzeros_f],
                samples=samples,
                infeasible_cost=0.0,
                eta=torch.tensor([10e-1, 10e-2]).to(**tkwargs),
            )
            self.assertTrue(
                torch.allclose(
                    obj,
                    samples
                    * torch.sigmoid(torch.as_tensor(-0.1) / 10e-1)
                    * torch.sigmoid(torch.as_tensor(-0.1) / 10e-2),
                )
            )
            # nonnegative objective, two constraint explicit different eta
            # use ones_f
            obj = samples.clone()
            obj = apply_constraints(
                obj=obj,
                constraints=[ones_f, ones_f],
                samples=samples,
                infeasible_cost=0.0,
                eta=torch.tensor([1, 10]).to(**tkwargs),
            )
            self.assertTrue(
                torch.allclose(
                    obj,
                    samples
                    * torch.sigmoid(torch.as_tensor(-1.0) / 1.0)
                    * torch.sigmoid(torch.as_tensor(-1.0) / 10.0),
                )
            )
            # negative objective, one constraint, infeasible_cost
            obj = samples.clone().clamp_min(-1.0)
            obj = apply_constraints(
                obj=obj, constraints=[zeros_f], samples=samples, infeasible_cost=2.0
            )
            self.assertAllClose(obj, samples.clamp_min(-1.0) * 0.5 - 1.0)
            # negative objective, one constraint, infeasible_cost, explicit eta
            obj = samples.clone().clamp_min(-1.0)
            obj = apply_constraints(
                obj=obj,
                constraints=[zeros_f],
                samples=samples,
                infeasible_cost=2.0,
                eta=torch.tensor([10e-3]).to(**tkwargs),
            )
            self.assertAllClose(obj, samples.clamp_min(-1.0) * 0.5 - 1.0)
            # nonnegative objective, one constraint, eta = 0
            obj = samples
            with self.assertRaises(ValueError):
                apply_constraints(
                    obj=obj,
                    constraints=[zeros_f],
                    samples=samples,
                    infeasible_cost=0.0,
                    eta=0.0,
                )

    def test_apply_constraints_wrong_eta_dim(self):
        tkwargs = {"device": self.device}
        for dtype in (torch.float, torch.double):
            tkwargs["dtype"] = dtype
            samples = torch.rand(3, 2, **tkwargs)
            obj = samples.clone()
            with self.assertRaises(ValueError):
                obj = apply_constraints(
                    obj=obj,
                    constraints=[zeros_f, zeros_f],
                    samples=samples,
                    infeasible_cost=0.0,
                    eta=torch.tensor([0.1]).to(**tkwargs),
                )
            with self.assertRaises(ValueError):
                obj = apply_constraints(
                    obj=obj,
                    constraints=[zeros_f, zeros_f],
                    samples=samples,
                    infeasible_cost=0.0,
                    eta=torch.tensor([0.1, 0.1, 0.3]).to(**tkwargs),
                )


class TestGetObjectiveWeightsTransform(BotorchTestCase):
    def test_NoWeights(self):
        Y = torch.ones(5, 2, 4, 1)
        objective_transform = get_objective_weights_transform(None)
        Y_transformed = objective_transform(Y)
        self.assertTrue(torch.equal(Y.squeeze(-1), Y_transformed))

    def test_OneWeightBroadcasting(self):
        Y = torch.ones(5, 2, 4, 1)
        objective_transform = get_objective_weights_transform(torch.tensor([0.5]))
        Y_transformed = objective_transform(Y)
        self.assertTrue(torch.equal(0.5 * Y.sum(dim=-1), Y_transformed))

    def test_IncompatibleNumberOfWeights(self):
        Y = torch.ones(5, 2, 4, 3)
        objective_transform = get_objective_weights_transform(torch.tensor([1.0, 2.0]))
        with self.assertRaises(RuntimeError):
            objective_transform(Y)

    def test_MultiTaskWeights(self):
        Y = torch.ones(5, 2, 4, 2)
        objective_transform = get_objective_weights_transform(torch.tensor([1.0, 1.0]))
        Y_transformed = objective_transform(Y)
        self.assertTrue(torch.equal(torch.sum(Y, dim=-1), Y_transformed))

    def test_NoMCSamples(self):
        Y = torch.ones(2, 4, 2)
        objective_transform = get_objective_weights_transform(torch.tensor([1.0, 1.0]))
        Y_transformed = objective_transform(Y)
        self.assertTrue(torch.equal(torch.sum(Y, dim=-1), Y_transformed))
