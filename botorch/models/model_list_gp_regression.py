#! /usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

r"""
Model List GP Regression models.
"""

import typing  # noqa F401

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

    def get_fantasy_model(
        self, inputs: Tensor, targets: Tensor, **kwargs
    ) -> "ModelListGP":
        r"""Construct a fantasy model.

        This method wraps gpytorch's `IndependentModelList.get_fantasy_model`
        method to provide an API consistent with that of BoTorch's batched
        multi-output GP models.

        Args:
            inputs: A `batch_shape x m x d` or
                `f_batch_shape x batch_shape x m x d`-dim Tensor of inputs for the
                fantasy observations, where `f_batch_shape` are fantasy batch
                dimensions. Note: when using the same inputs for all fantasies,
                inputs should be `batch_shape x m x d` to avoid recomputing the
                repeated blocks of the covariance matrix. Additionally, if provided,
                the "noise" keyword argument should map to a `batch_shape x m`-dim
                Tensor of observed measurement noise for fastest performance.
            targets: `batch_shape x m x o` or
                `f_batch_shape x batch_shape x m x o`-dim Tensor of fantasy
                observations.

        Returns:
            A `ModelListGP`, where the `i`-th model has `n_i + m` training examples,
            where the `m` fantasy examples have been added and all test-time caches
            have been updated.
        """
        inputs_ = [inputs] * self.num_outputs
        if targets.shape[-1] != self.num_outputs:
            raise ValueError(
                "Incorrect number of outputs for fantasy observations. "
                f"Received {targets.shape[-1]} observation outputs, but "
                f"model has {self.num_outputs} outputs."
            )
        targets_ = [targets[..., i] for i in range(targets.shape[-1])]
        if "noise" in kwargs:
            noise = kwargs.pop("noise")
            if noise.shape[-1] != self.num_outputs:
                raise ValueError(
                    "Incorrect number of outputs for fantasy noise. "
                    f"Received {noise.shape[-1]} observation outputs, but "
                    f"model has {self.num_outputs} outputs."
                )
            kwargs_ = {
                **kwargs,
                "noise": [noise[..., i] for i in range(targets.shape[-1])],
            }
        else:
            kwargs_ = kwargs
        return super().get_fantasy_model(inputs_, targets_, **kwargs_)
