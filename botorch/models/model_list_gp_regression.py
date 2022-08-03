#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Model List GP Regression models.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, List

from botorch.exceptions.errors import BotorchTensorDimensionError
from botorch.models.gpytorch import GPyTorchModel, ModelListGPyTorchModel
from gpytorch.models import IndependentModelList
from torch import Tensor


class ModelListGP(IndependentModelList, ModelListGPyTorchModel):
    r"""A multi-output GP model with independent GPs for the outputs.

    This model supports different-shaped training inputs for each of its
    sub-models. It can be used with any number of single-output
    `GPyTorchModel`\s and the models can be of different types. Use this model
    when you have independent outputs with different training data. When
    modeling correlations between outputs, use `MultiTaskGP`.

    Internally, this model is just a list of individual models, but it implements
    the same input/output interface as all other BoTorch models. This makes it
    very flexible and convenient to work with. The sequential evaluation comes
    at a performance cost though - if you are using a block design (i.e. the
    same number of training example for each output, and a similar model
    structure, you should consider using a batched GP model instead, such as
    `SingleTaskGP` with batched inputs).
    """

    def __init__(self, *gp_models: GPyTorchModel) -> None:
        r"""
        Args:
            *gp_models: A number of single-output `GPyTorchModel`\s.
                If models have input/output transforms, these are honored
                individually for each model.

        Example:
            >>> model1 = SingleTaskGP(train_X1, train_Y1)
            >>> model2 = SingleTaskGP(train_X2, train_Y2)
            >>> model = ModelListGP(model1, model2)
        """
        super().__init__(*gp_models)

    def condition_on_observations(
        self, X: List[Tensor], Y: Tensor, **kwargs: Any
    ) -> ModelListGP:
        r"""Condition the model on new observations.

        Args:
            X: A `m`-list of `batch_shape x n' x d`-dim Tensors, where `d` is the
                dimension of the feature space, `n'` is the number of points
                per batch, and `batch_shape` is the batch shape (must be compatible
                with the batch shape of the model).
            Y: A `batch_shape' x n' x m`-dim Tensor, where `m` is the number of
                model outputs, `n'` is the number of points per batch, and
                `batch_shape'` is the batch shape of the observations.
                `batch_shape'` must be broadcastable to `batch_shape` using
                standard broadcasting semantics. If `Y` has fewer batch dimensions
                than `X`, its is assumed that the missing batch dimensions are
                the same for all `Y`.
            kwargs: Keyword arguments passed to
                `IndependentModelList.get_fantasy_model`.

        Returns:
            A `ModelListGP` representing the original model
            conditioned on the new observations `(X, Y)` (and possibly noise
            observations passed in via kwargs). Here the `i`-th model has
            `n_i + n'` training examples, where the `n'` training examples have
            been added and all test-time caches have been updated.
        """
        if Y.shape[-1] != self.num_outputs:
            raise BotorchTensorDimensionError(
                "Incorrect number of outputs for observations. Received "
                f"{Y.shape[-1]} observation outputs, but model has "
                f"{self.num_outputs} outputs."
            )
        targets = [Y[..., i] for i in range(Y.shape[-1])]
        for i, model in enumerate(self.models):
            if hasattr(model, "outcome_transform"):
                noise = kwargs.get("noise")
                targets[i], noise = model.outcome_transform(targets[i], noise)

        # This should never trigger, posterior call would fail.
        assert len(targets) == len(X)
        if "noise" in kwargs:
            noise = kwargs.pop("noise")
            if noise.shape != Y.shape[-noise.dim() :]:
                raise BotorchTensorDimensionError(
                    "The shape of observation noise does not agree with the outcomes. "
                    f"Received {noise.shape} noise with {Y.shape} outcomes."
                )
            kwargs_ = {**kwargs, "noise": [noise[..., i] for i in range(Y.shape[-1])]}
        else:
            kwargs_ = kwargs
        return super().get_fantasy_model(X, targets, **kwargs_)

    def subset_output(self, idcs: List[int]) -> ModelListGP:
        r"""Subset the model along the output dimension.

        Args:
            idcs: The output indices to subset the model to.

        Returns:
            The current model, subset to the specified output indices.
        """
        return self.__class__(*[deepcopy(self.models[i]) for i in idcs])

    def _set_transformed_inputs(self) -> None:
        r"""Update training inputs with transformed inputs."""
        for m in self.models:
            m._set_transformed_inputs()

    def _revert_to_original_inputs(self) -> None:
        r"""Revert training inputs back to original."""
        for m in self.models:
            m._revert_to_original_inputs()
