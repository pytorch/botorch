#! /usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

r"""
Model List GP Regression models.
"""

from typing import Any

from gpytorch.models import IndependentModelList
from torch import Tensor

from .gpytorch import GPyTorchModel, ModelListGPyTorchModel


class ModelListGP(IndependentModelList, ModelListGPyTorchModel):
    r"""A multi-output GP model with independent GPs for the outputs.

    This model supports different-shaped training inputs for each of its
    sub-models. It can be used with any BoTorch models.

    Internally, this model is just a list of individual models, but it implements
    the same input/output interface as all other BoTorch models. This makes it
    very flexible and convenient to work with. The sequential evaluation comes
    at a performance cost though - if you are using a block design (i.e. the
    same number of training example for each output, and a similar model
    structure, you should consider using a batched GP model instead).
    """

    def __init__(self, *gp_models: GPyTorchModel) -> None:
        r"""A multi-output GP model with independent GPs for the outputs.

        Args:
            *gp_models: An variable number of single-output BoTorch models.

        Example:
            >>> model1 = SingleTaskGP(train_X1, train_Y1)
            >>> model2 = SingleTaskGP(train_X2, train_Y2)
            >>> model = ModelListGP(model1, model2)
        """
        super().__init__(*gp_models)

    def condition_on_observations(
        self, X: Tensor, Y: Tensor, **kwargs: Any
    ) -> "ModelListGP":
        r"""Condition the model on new observations.

        Args:
            X: A `batch_shape x m x d`-dim Tensor, where `d` is the dimension of
                the feature space, `m` is the number of points per batch, and
                `batch_shape` is the batch shape (must be compatible with the
                batch shape of the model).
            Y: A `batch_shape' x m x (o)`-dim Tensor, where `o` is the number of
                model outputs, `m` is the number of points per batch, and
                `batch_shape'` is the batch shape of the observations.
                `batch_shape'` must be broadcastable to `batch_shape` using
                standard broadcasting semantics. If `Y` has fewer batch dimensions
                than `X`, its is assumed that the missing batch dimensions are
                the same for all `Y`.

        Returns:
            A `ModelListGPyTorchModel` representing the original model
            conditioned on the new observations `(X, Y)` (and possibly noise
            observations passed in via kwargs). Here the `i`-th model has
            `n_i + m` training examples, where the `m` training examples have
            been added and all test-time caches have been updated.
        """
        inputs = [X] * self.num_outputs
        if Y.shape[-1] != self.num_outputs:
            raise ValueError(
                "Incorrect number of outputs for observations. Received "
                f"{Y.shape[-1]} observation outputs, but model has "
                f"{self.num_outputs} outputs."
            )
        targets = [Y[..., i] for i in range(Y.shape[-1])]
        if "noise" in kwargs:
            noise = kwargs.pop("noise")
            if noise.shape[-1] != self.num_outputs:
                raise ValueError(
                    "Incorrect number of outputs for noise observations. "
                    f"Received {noise.shape[-1]} observation outputs, but "
                    f"model has {self.num_outputs} outputs."
                )
            kwargs_ = {**kwargs, "noise": [noise[..., i] for i in range(Y.shape[-1])]}
        else:
            kwargs_ = kwargs
        return super().get_fantasy_model(inputs, targets, **kwargs_)
