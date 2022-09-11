#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Abstract base module for all BoTorch models.

Contains `Model`, the abstract base class for all BoTorch models, and
`ModelList`, a container for a list of Models.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
from typing import Any, Callable, Dict, Hashable, List, Mapping, Optional, Union

import numpy as np
import torch
from botorch import settings
from botorch.models.utils.assorted import fantasize as fantasize_flag
from botorch.posteriors import Posterior, PosteriorList
from botorch.posteriors.fully_bayesian import FullyBayesianPosteriorList
from botorch.sampling.samplers import MCSampler
from botorch.utils.datasets import BotorchDataset
from botorch.utils.transforms import is_fully_bayesian
from torch import Tensor
from torch.nn import Module, ModuleList


class Model(Module, ABC):
    r"""Abstract base class for BoTorch models.

    Model cannot be used directly; it only defines an API for other BoTorch
    models.

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
        posterior_transform: Optional[Callable[[Posterior], Posterior]] = None,
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
            posterior_transform: An optional PosteriorTransform.

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
        raise NotImplementedError(
            f"`condition_on_observations` not implemented for {self.__class__.__name__}"
        )

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
            return self.condition_on_observations(
                X=self.transform_inputs(X), Y=Y_fantasized, **kwargs
            )

    @classmethod
    def construct_inputs(
        cls,
        training_data: Union[BotorchDataset, Dict[Hashable, BotorchDataset]],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        r"""Construct `Model` keyword arguments from a dict of `BotorchDataset`."""
        from botorch.models.utils.parse_training_data import parse_training_data

        return parse_training_data(cls, training_data, **kwargs)

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
                    "Could not update `train_inputs` with transformed inputs "
                    f"since {self.__class__.__name__} does not have a `train_inputs` "
                    "attribute. Make sure that the `input_transform` is applied to "
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


class ModelList(Model):
    r"""A multi-output Model represented by a list of independent models.

    All
    BoTorch models are acceptable as inputs. The cost of this flexibility is
    that `ModelList` does not support all methods that may be implemented by its
    component models. One use case for `ModelList` is combining a regression
    model and a deterministic model in one multi-output container model, e.g.
    for cost-aware or multi-objective optimization where one of the outcomes is
    a deterministic function of the inputs.
    """

    def __init__(self, *models: Model) -> None:
        r"""
        Args:
            *models: A variable number of models.

        Example:
            >>> m_1 = SingleTaskGP(train_X, train_Y)
            >>> m_2 = GenericDeterministicModel(lambda x: x.sum(dim=-1))
            >>> m_12 = ModelList(m_1, m_2)
            >>> m_12.predict(test_X)
        """
        super().__init__()
        self.models = ModuleList(models)

    def _get_group_subset_indices(
        self, idcs: Optional[List[int]]
    ) -> Dict[int, List[int]]:
        r"""Convert global subset indices to indices for the individual models.

        Args:
            idcs: A list of inidices to which the `ModelList` model is to be
                subset to.

        Returns:
            A dictionary mapping model indices to subset indices of the
                respective model in the `ModelList`.
        """
        if idcs is None:
            return {i: None for i in range(len(self.models))}
        output_sizes = [model.num_outputs for model in self.models]
        cum_output_sizes = np.cumsum(output_sizes)
        idcs = [idx % cum_output_sizes[-1] for idx in idcs]
        group_indices: Dict[int, List[int]] = defaultdict(list)
        for idx in idcs:
            grp_idx = int(np.argwhere(idx < cum_output_sizes)[0])
            sub_idx = idx - int(np.sum(output_sizes[:grp_idx]))
            group_indices[grp_idx].append(sub_idx)
        return group_indices

    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: bool = False,
        posterior_transform: Optional[Callable[[Posterior], Posterior]] = None,
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
            posterior_transform: An optional PosteriorTransform.

        Returns:
            A `Posterior` object, representing a batch of `b` joint distributions
            over `q` points and `m` outputs each.
        """
        group_indices = self._get_group_subset_indices(idcs=output_indices)
        posteriors = [
            self.models[i].posterior(
                X=X, output_indices=idcs, observation_noise=observation_noise
            )
            for i, idcs in group_indices.items()
        ]
        if any(is_fully_bayesian(m) for m in self.models):
            posterior = FullyBayesianPosteriorList(*posteriors)
        else:
            posterior = PosteriorList(*posteriors)
        if posterior_transform is not None:
            posterior = posterior_transform(posterior)
        return posterior

    @property
    def batch_shape(self) -> torch.Size:
        r"""The batch shape of the model.

        This is a batch shape from an I/O perspective, independent of the internal
        representation of the model (as e.g. in BatchedMultiOutputGPyTorchModel).
        For a model with `m` outputs, a `test_batch_shape x q x d`-shaped input `X`
        to the `posterior` method returns a Posterior object over an output of
        shape `broadcast(test_batch_shape, model.batch_shape) x q x m`.
        """
        batch_shape = self.models[0].batch_shape
        if all(batch_shape == m.batch_shape for m in self.models[1:]):
            return batch_shape
        # TODO: Allow broadcasting of model batch shapes
        raise NotImplementedError(
            f"`{self.__class__.__name__}.batch_shape` is only supported if all "
            "constituent models have the same `batch_shape`."
        )

    @property
    def num_outputs(self) -> int:
        r"""The number of outputs of the model.

        Equal to the sum of the number of outputs of the individual models
        in the ModelList.
        """
        return sum(model.num_outputs for model in self.models)

    def subset_output(self, idcs: List[int]) -> Model:
        r"""Subset the model along the output dimension.

        Args:
            idcs: The output indices to subset the model to. Relative to the
                overall number of outputs of the model.

        Returns:
            A `Model` (either a `ModelList` or one of the submodels) with
            the outputs subset to the indices in `idcs`.

        Internally, this drops (if single-output) or subsets (if multi-output)
        the constitutent models and returns them as a `ModelList`. If the
        result is a single (possibly subset) model from the list, returns this
        model (instead of forming a degenerate singe-model `ModelList`).
        For instance, if `m = ModelList(m1, m2)` with `m1` a two-output model
        and `m2` a single-output model, then `m.subset_output([1]) ` will return
        the model `m1` subset to its second output.
        """
        group_indices = self._get_group_subset_indices(idcs=idcs)
        subset_models = [
            deepcopy(self.models[grp_idx].subset_output(idcs=sub_idcs))
            for grp_idx, sub_idcs in group_indices.items()
        ]
        if len(subset_models) == 1:
            return subset_models[0]
        return self.__class__(*subset_models)

    def transform_inputs(self, X: Tensor) -> List[Tensor]:
        r"""Individually transform the inputs for each model.

        Args:
            X: A tensor of inputs.

        Returns:
            A list of tensors of transformed inputs.
        """
        transformed_X_list = []
        for model in self.models:
            try:
                transformed_X_list.append(model.input_transform(X))
            except AttributeError:
                transformed_X_list.append(X)
        return transformed_X_list

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True
    ) -> None:
        """Initialize the fully Bayesian models before loading the state dict."""
        for i, m in enumerate(self.models):
            if is_fully_bayesian(m):
                filtered_dict = {
                    k.replace(f"models.{i}.", ""): v
                    for k, v in state_dict.items()
                    if k.startswith(f"models.{i}")
                }
                m.load_state_dict(filtered_dict)
        super().load_state_dict(state_dict=state_dict, strict=strict)
