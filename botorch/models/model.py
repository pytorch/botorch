#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Abstract base module for all BoTorch models.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import torch
from botorch import settings
from botorch.models.utils import fantasize as fantasize_flag
from botorch.posteriors import Posterior
from botorch.sampling.samplers import MCSampler
from botorch.utils.containers import TrainingData
from torch import Tensor
from torch.nn import Module


class Model(Module, ABC):
    r"""Abstract base class for BoTorch models.

    Args:
        _has_transformed_inputs: A boolean denoting whether `train_inputs` are currently
            stored as transformed or not.
        _original_train_inputs: A Tensor storing the original train inputs for use in
            `_revert_to_original_inputs`. Note that this is necessary since
            transform / untransform cycle introduces numerical errors which lead
            to upstream errors during training.
    """

    _has_transformed_inputs: bool = False
    _original_train_inputs: Optional[Tensor] = None

    @abstractmethod
    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: bool = False,
        **kwargs: Any,
    ) -> Posterior:
        r"""Computes the posterior over model outputs at the provided points.

        Note: The input transforms should be applied here using
            `self.transform_inputs(X)` after the `self.eval()` call and before
            any `model.forward` or `model.likelihood` calls.

        Args:
            X: A `b x q x d`-dim Tensor, where `d` is the dimension of the
                feature space, `q` is the number of points considered jointly,
                and `b` is the batch dimension.
            output_indices: A list of indices, corresponding to the outputs over
                which to compute the posterior (if the model is multi-output).
                Can be used to speed up computation if only a subset of the
                model's outputs are required for optimization. If omitted,
                computes the posterior over all model outputs.
            observation_noise: If True, add observation noise to the posterior.

        Returns:
            A `Posterior` object, representing a batch of `b` joint distributions
            over `q` points and `m` outputs each.
        """
        pass  # pragma: no cover

    @property
    def batch_shape(self) -> torch.Size:
        r"""The batch shape of the model.

        This is a batch shape from an I/O perspective, independent of the internal
        representation of the model (as e.g. in BatchedMultiOutputGPyTorchModel).
        For a model with `m` outputs, a `test_batch_shape x q x d`-shaped input `X`
        to the `posterior` method returns a Posterior object over an output of
        shape `broadcast(test_batch_shape, model.batch_shape) x q x m`.
        """
        cls_name = self.__class__.__name__
        raise NotImplementedError(f"{cls_name} does not define batch_shape property")

    @property
    def num_outputs(self) -> int:
        r"""The number of outputs of the model."""
        cls_name = self.__class__.__name__
        raise NotImplementedError(f"{cls_name} does not define num_outputs property")

    def subset_output(self, idcs: List[int]) -> Model:
        r"""Subset the model along the output dimension.

        Args:
            idcs: The output indices to subset the model to.

        Returns:
            A `Model` object of the same type and with the same parameters as
            the current model, subset to the specified output indices.
        """
        raise NotImplementedError

    def condition_on_observations(self, X: Tensor, Y: Tensor, **kwargs: Any) -> Model:
        r"""Condition the model on new observations.

        Args:
            X: A `batch_shape x n' x d`-dim Tensor, where `d` is the dimension of
                the feature space, `n'` is the number of points per batch, and
                `batch_shape` is the batch shape (must be compatible with the
                batch shape of the model).
            Y: A `batch_shape' x n' x m`-dim Tensor, where `m` is the number of
                model outputs, `n'` is the number of points per batch, and
                `batch_shape'` is the batch shape of the observations.
                `batch_shape'` must be broadcastable to `batch_shape` using
                standard broadcasting semantics. If `Y` has fewer batch dimensions
                than `X`, it is assumed that the missing batch dimensions are
                the same for all `Y`.

        Returns:
            A `Model` object of the same type, representing the original model
            conditioned on the new observations `(X, Y)` (and possibly noise
            observations passed in via kwargs).
        """
        raise NotImplementedError

    def fantasize(
        self,
        X: Tensor,
        sampler: MCSampler,
        observation_noise: bool = True,
        **kwargs: Any,
    ) -> Model:
        r"""Construct a fantasy model.

        Constructs a fantasy model in the following fashion:
        (1) compute the model posterior at `X` (including observation noise if
        `observation_noise=True`).
        (2) sample from this posterior (using `sampler`) to generate "fake"
        observations.
        (3) condition the model on the new fake observations.

        Args:
            X: A `batch_shape x n' x d`-dim Tensor, where `d` is the dimension of
                the feature space, `n'` is the number of points per batch, and
                `batch_shape` is the batch shape (must be compatible with the
                batch shape of the model).
            sampler: The sampler used for sampling from the posterior at `X`.
            observation_noise: If True, include observation noise.

        Returns:
            The constructed fantasy model.
        """
        propagate_grads = kwargs.pop("propagate_grads", False)
        with fantasize_flag():
            with settings.propagate_grads(propagate_grads):
                post_X = self.posterior(X, observation_noise=observation_noise)
            Y_fantasized = sampler(post_X)  # num_fantasies x batch_shape x n' x m
            return self.condition_on_observations(X=X, Y=Y_fantasized, **kwargs)

    @classmethod
    def construct_inputs(
        cls, training_data: TrainingData, **kwargs: Any
    ) -> Dict[str, Any]:
        r"""Construct kwargs for the `Model` from `TrainingData` and other options."""
        raise NotImplementedError(
            f"`construct_inputs` not implemented for {cls.__name__}."
        )

    def transform_inputs(
        self,
        X: Tensor,
        input_transform: Optional[Module] = None,
    ) -> Tensor:
        r"""Transform inputs.

        Args:
            X: A tensor of inputs
            input_transform: A Module that performs the input transformation.

        Returns:
            A tensor of transformed inputs
        """
        if input_transform is not None:
            input_transform.to(X)
            return input_transform(X)
        try:
            return self.input_transform(X)
        except AttributeError:
            return X

    def _set_transformed_inputs(self) -> None:
        r"""Update training inputs with transformed inputs."""
        if hasattr(self, "input_transform") and not self._has_transformed_inputs:
            if hasattr(self, "train_inputs"):
                self._original_train_inputs = self.train_inputs[0]
                with torch.no_grad():
                    X_tf = self.input_transform.preprocess_transform(
                        self.train_inputs[0]
                    )
                self.set_train_data(X_tf, strict=False)
                self._has_transformed_inputs = True
            else:
                warnings.warn(
                    "Could not update `train_inputs` with transformed inputs"
                    f"since {self.__class__.__name__} does not have a `train_inputs`"
                    "attribute. Make sure that the `input_transform` is applied to"
                    "both the train inputs and test inputs.",
                    RuntimeWarning,
                )

    def _revert_to_original_inputs(self) -> None:
        r"""Revert training inputs back to original."""
        if hasattr(self, "input_transform") and self._has_transformed_inputs:
            self.set_train_data(self._original_train_inputs, strict=False)
            self._has_transformed_inputs = False

    def eval(self) -> Model:
        r"""Puts the model in `eval` mode and sets the transformed inputs."""
        self._set_transformed_inputs()
        return super().eval()

    def train(self, mode: bool = True) -> Model:
        r"""Puts the model in `train` mode and reverts to the original inputs.

        Args:
            mode: A boolean denoting whether to put in `train` or `eval` mode.
                If `False`, model is put in `eval` mode.
        """
        if mode:
            self._revert_to_original_inputs()
        else:
            self._set_transformed_inputs()
        return super().train(mode=mode)
