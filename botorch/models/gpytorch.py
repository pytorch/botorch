#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Abstract model class for all GPyTorch-based botorch models.

To implement your own, simply inherit from both the provided classes and a
GPyTorch Model class such as an ExactGP.
"""

from __future__ import annotations

import itertools
import warnings
from abc import ABC
from copy import deepcopy
from typing import Any, Iterator, List, Optional, Tuple, Union

import torch
from botorch.acquisition.objective import PosteriorTransform
from botorch.exceptions.errors import BotorchTensorDimensionError
from botorch.exceptions.warnings import BotorchTensorDimensionWarning
from botorch.models.model import Model, ModelList
from botorch.models.utils import (
    _make_X_full,
    add_output_dim,
    gpt_posterior_settings,
    mod_batch_shape,
    multioutput_to_batch_mode_transform,
)
from botorch.posteriors.fully_bayesian import FullyBayesianPosterior
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.utils.transforms import is_fully_bayesian
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from gpytorch.likelihoods.gaussian_likelihood import FixedNoiseGaussianLikelihood
from torch import Tensor


class GPyTorchModel(Model, ABC):
    r"""Abstract base class for models based on GPyTorch models.

    The easiest way to use this is to subclass a model from a GPyTorch model
    class (e.g. an `ExactGP`) and this `GPyTorchModel`. See e.g. `SingleTaskGP`.

    :meta private:
    """

    @staticmethod
    def _validate_tensor_args(
        X: Tensor, Y: Tensor, Yvar: Optional[Tensor] = None, strict: bool = True
    ) -> None:
        r"""Checks that `Y` and `Yvar` have an explicit output dimension if strict.

        This also checks that `Yvar` has the same trailing dimensions as `Y`. Note
        we only infer that an explicit output dimension exists when `X` and `Y` have
        the same `batch_shape`.

        Args:
            X: A `batch_shape x n x d`-dim Tensor, where `d` is the dimension of
                the feature space, `n` is the number of points per batch, and
                `batch_shape` is the batch shape (potentially empty).
            Y: A `batch_shape' x n x m`-dim Tensor, where `m` is the number of
                model outputs, `n'` is the number of points per batch, and
                `batch_shape'` is the batch shape of the observations.
            Yvar: A `batch_shape' x n x m` tensor of observed measurement noise.
                Note: this will be None when using a model that infers the noise
                level (e.g. a `SingleTaskGP`).
            strict: A boolean indicating whether to check that `Y` and `Yvar`
                have an explicit output dimension.
        """
        if strict:
            if X.dim() != Y.dim():
                if (X.dim() - Y.dim() == 1) and (X.shape[:-1] == Y.shape):
                    message = (
                        "An explicit output dimension is required for targets."
                        f" Expected Y with dimension: {Y.dim()} (got {X.dim()})."
                    )
                else:
                    message = (
                        "Expected X and Y to have the same number of dimensions"
                        f" (got X with dimension {X.dim()} and Y with dimension"
                        f" {Y.dim()}."
                    )
                raise BotorchTensorDimensionError(message)
        else:
            warnings.warn(
                "Non-strict enforcement of botorch tensor conventions. Ensure that "
                f"target tensors Y{' and Yvar have' if Yvar is not None else ' has an'}"
                f" explicit output dimension{'s' if Yvar is not None else ''}.",
                BotorchTensorDimensionWarning,
            )
        # Yvar may not have the same batch dimensions, but the trailing dimensions
        # of Yvar should be the same as the trailing dimensions of Y.
        if Yvar is not None and Y.shape[-(Yvar.dim()) :] != Yvar.shape:
            raise BotorchTensorDimensionError(
                "An explicit output dimension is required for observation noise."
                f" Expected Yvar with shape: {Y.shape[-Yvar.dim() :]} (got"
                f" {Yvar.shape})."
            )

    @property
    def batch_shape(self) -> torch.Size:
        r"""The batch shape of the model.

        This is a batch shape from an I/O perspective, independent of the internal
        representation of the model (as e.g. in BatchedMultiOutputGPyTorchModel).
        For a model with `m` outputs, a `test_batch_shape x q x d`-shaped input `X`
        to the `posterior` method returns a Posterior object over an output of
        shape `broadcast(test_batch_shape, model.batch_shape) x q x m`.
        """
        return self.train_inputs[0].shape[:-2]

    @property
    def num_outputs(self) -> int:
        r"""The number of outputs of the model."""
        return self._num_outputs

    def posterior(
        self,
        X: Tensor,
        observation_noise: Union[bool, Tensor] = False,
        posterior_transform: Optional[PosteriorTransform] = None,
        **kwargs: Any,
    ) -> GPyTorchPosterior:
        r"""Computes the posterior over model outputs at the provided points.

        Args:
            X: A `(batch_shape) x q x d`-dim Tensor, where `d` is the dimension
                of the feature space and `q` is the number of points considered
                jointly.
            observation_noise: If True, add the observation noise from the
                likelihood to the posterior. If a Tensor, use it directly as the
                observation noise (must be of shape `(batch_shape) x q`).
            posterior_transform: An optional PosteriorTransform.

        Returns:
            A `GPyTorchPosterior` object, representing a batch of `b` joint
            distributions over `q` points. Includes observation noise if
            specified.
        """
        self.eval()  # make sure model is in eval mode
        # input transforms are applied at `posterior` in `eval` mode, and at
        # `model.forward()` at the training time
        X = self.transform_inputs(X)
        with gpt_posterior_settings():
            mvn = self(X)
            if observation_noise is not False:
                if torch.is_tensor(observation_noise):
                    # TODO: Make sure observation noise is transformed correctly
                    self._validate_tensor_args(X=X, Y=observation_noise)
                    if observation_noise.size(-1) == 1:
                        observation_noise = observation_noise.squeeze(-1)
                    mvn = self.likelihood(mvn, X, noise=observation_noise)
                else:
                    mvn = self.likelihood(mvn, X)
        posterior = GPyTorchPosterior(mvn=mvn)
        if hasattr(self, "outcome_transform"):
            posterior = self.outcome_transform.untransform_posterior(posterior)
        if posterior_transform is not None:
            return posterior_transform(posterior)
        return posterior

    def condition_on_observations(self, X: Tensor, Y: Tensor, **kwargs: Any) -> Model:
        r"""Condition the model on new observations.

        Args:
            X: A `batch_shape x n' x d`-dim Tensor, where `d` is the dimension of
                the feature space, `n'` is the number of points per batch, and
                `batch_shape` is the batch shape (must be compatible with the
                batch shape of the model).
            Y: A `batch_shape' x n x m`-dim Tensor, where `m` is the number of
                model outputs, `n'` is the number of points per batch, and
                `batch_shape'` is the batch shape of the observations.
                `batch_shape'` must be broadcastable to `batch_shape` using
                standard broadcasting semantics. If `Y` has fewer batch dimensions
                than `X`, its is assumed that the missing batch dimensions are
                the same for all `Y`.

        Returns:
            A `Model` object of the same type, representing the original model
            conditioned on the new observations `(X, Y)` (and possibly noise
            observations passed in via kwargs).

        Example:
            >>> train_X = torch.rand(20, 2)
            >>> train_Y = torch.sin(train_X[:, 0]) + torch.cos(train_X[:, 1])
            >>> model = SingleTaskGP(train_X, train_Y)
            >>> new_X = torch.rand(5, 2)
            >>> new_Y = torch.sin(new_X[:, 0]) + torch.cos(new_X[:, 1])
            >>> model = model.condition_on_observations(X=new_X, Y=new_Y)
        """
        Yvar = kwargs.get("noise", None)
        if hasattr(self, "outcome_transform"):
            # pass the transformed data to get_fantasy_model below
            # (unless we've already trasnformed if BatchedMultiOutputGPyTorchModel)
            if not isinstance(self, BatchedMultiOutputGPyTorchModel):
                Y, Yvar = self.outcome_transform(Y, Yvar)
        # validate using strict=False, since we cannot tell if Y has an explicit
        # output dimension
        self._validate_tensor_args(X=X, Y=Y, Yvar=Yvar, strict=False)
        if Y.size(-1) == 1:
            Y = Y.squeeze(-1)
            if Yvar is not None:
                kwargs.update({"noise": Yvar.squeeze(-1)})
        # get_fantasy_model will properly copy any existing outcome transforms
        # (since it deepcopies the original model)
        return self.get_fantasy_model(inputs=X, targets=Y, **kwargs)


class BatchedMultiOutputGPyTorchModel(GPyTorchModel):
    r"""Base class for batched multi-output GPyTorch models with independent outputs.

    This model should be used when the same training data is used for all outputs.
    Outputs are modeled independently by using a different batch for each output.

    :meta private:
    """

    _num_outputs: int
    _input_batch_shape: torch.Size
    _aug_batch_shape: torch.Size

    @staticmethod
    def get_batch_dimensions(
        train_X: Tensor, train_Y: Tensor
    ) -> Tuple[torch.Size, torch.Size]:
        r"""Get the raw batch shape and output-augmented batch shape of the inputs.

        Args:
            train_X: A `n x d` or `batch_shape x n x d` (batch mode) tensor of training
                features.
            train_Y: A `n x m` or `batch_shape x n x m` (batch mode) tensor of
                training observations.

        Returns:
            2-element tuple containing

            - The `input_batch_shape`
            - The output-augmented batch shape: `input_batch_shape x (m)`
        """
        input_batch_shape = train_X.shape[:-2]
        aug_batch_shape = input_batch_shape
        num_outputs = train_Y.shape[-1]
        if num_outputs > 1:
            aug_batch_shape += torch.Size([num_outputs])
        return input_batch_shape, aug_batch_shape

    def _set_dimensions(self, train_X: Tensor, train_Y: Tensor) -> None:
        r"""Store the number of outputs and the batch shape.

        Args:
            train_X: A `n x d` or `batch_shape x n x d` (batch mode) tensor of training
                features.
            train_Y: A `n x m` or `batch_shape x n x m` (batch mode) tensor of
                training observations.
        """
        self._num_outputs = train_Y.shape[-1]
        self._input_batch_shape, self._aug_batch_shape = self.get_batch_dimensions(
            train_X=train_X, train_Y=train_Y
        )

    @property
    def batch_shape(self) -> torch.Size:
        r"""The batch shape of the model.

        This is a batch shape from an I/O perspective, independent of the internal
        representation of the model (as e.g. in BatchedMultiOutputGPyTorchModel).
        For a model with `m` outputs, a `test_batch_shape x q x d`-shaped input `X`
        to the `posterior` method returns a Posterior object over an output of
        shape `broadcast(test_batch_shape, model.batch_shape) x q x m`.
        """
        return self._input_batch_shape

    def _transform_tensor_args(
        self, X: Tensor, Y: Tensor, Yvar: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        r"""Transforms tensor arguments: for single output models, the output
        dimension is squeezed and for multi-output models, the output dimension is
        transformed into the left-most batch dimension.

        Args:
            X: A `n x d` or `batch_shape x n x d` (batch mode) tensor of training
                features.
            Y: A `n x m` or `batch_shape x n x m` (batch mode) tensor of
                training observations.
            Yvar: A `n x m` or `batch_shape x n x m` (batch mode) tensor of
                observed measurement noise. Note: this will be None when using a model
                that infers the noise level (e.g. a `SingleTaskGP`).

        Returns:
            3-element tuple containing

            - A `input_batch_shape x (m) x n x d` tensor of training features.
            - A `target_batch_shape x (m) x n` tensor of training observations.
            - A `target_batch_shape x (m) x n` tensor observed measurement noise
                (or None).
        """
        if self._num_outputs > 1:
            return multioutput_to_batch_mode_transform(
                train_X=X, train_Y=Y, train_Yvar=Yvar, num_outputs=self._num_outputs
            )
        return X, Y.squeeze(-1), None if Yvar is None else Yvar.squeeze(-1)

    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: Union[bool, Tensor] = False,
        posterior_transform: Optional[PosteriorTransform] = None,
        **kwargs: Any,
    ) -> GPyTorchPosterior:
        r"""Computes the posterior over model outputs at the provided points.

        Args:
            X: A `(batch_shape) x q x d`-dim Tensor, where `d` is the dimension
                of the feature space and `q` is the number of points considered
                jointly.
            output_indices: A list of indices, corresponding to the outputs over
                which to compute the posterior (if the model is multi-output).
                Can be used to speed up computation if only a subset of the
                model's outputs are required for optimization. If omitted,
                computes the posterior over all model outputs.
            observation_noise: If True, add the observation noise from the
                likelihood to the posterior. If a Tensor, use it directly as the
                observation noise (must be of shape `(batch_shape) x q x m`).
            posterior_transform: An optional PosteriorTransform.

        Returns:
            A `GPyTorchPosterior` object, representing `batch_shape` joint
            distributions over `q` points and the outputs selected by
            `output_indices` each. Includes observation noise if specified.
        """
        self.eval()  # make sure model is in eval mode
        # input transforms are applied at `posterior` in `eval` mode, and at
        # `model.forward()` at the training time
        X = self.transform_inputs(X)
        with gpt_posterior_settings():
            # insert a dimension for the output dimension
            if self._num_outputs > 1:
                X, output_dim_idx = add_output_dim(
                    X=X, original_batch_shape=self._input_batch_shape
                )
            mvn = self(X)
            if observation_noise is not False:
                if torch.is_tensor(observation_noise):
                    # TODO: Validate noise shape
                    # make observation_noise `batch_shape x q x n`
                    obs_noise = observation_noise.transpose(-1, -2)
                    mvn = self.likelihood(mvn, X, noise=obs_noise)
                elif isinstance(self.likelihood, FixedNoiseGaussianLikelihood):
                    # Use the mean of the previous noise values (TODO: be smarter here).
                    noise = self.likelihood.noise.mean().expand(X.shape[:-1])
                    mvn = self.likelihood(mvn, X, noise=noise)
                else:
                    mvn = self.likelihood(mvn, X)
            if self._num_outputs > 1:
                mean_x = mvn.mean
                covar_x = mvn.lazy_covariance_matrix
                output_indices = output_indices or range(self._num_outputs)
                mvns = [
                    MultivariateNormal(
                        mean_x.select(dim=output_dim_idx, index=t),
                        covar_x[(slice(None),) * output_dim_idx + (t,)],
                    )
                    for t in output_indices
                ]
                mvn = MultitaskMultivariateNormal.from_independent_mvns(mvns=mvns)

        posterior = GPyTorchPosterior(mvn=mvn)
        if hasattr(self, "outcome_transform"):
            posterior = self.outcome_transform.untransform_posterior(posterior)
        if posterior_transform is not None:
            return posterior_transform(posterior)
        return posterior

    def condition_on_observations(
        self, X: Tensor, Y: Tensor, **kwargs: Any
    ) -> BatchedMultiOutputGPyTorchModel:
        r"""Condition the model on new observations.

        Args:
            X: A `batch_shape x n' x d`-dim Tensor, where `d` is the dimension of
                the feature space, `m` is the number of points per batch, and
                `batch_shape` is the batch shape (must be compatible with the
                batch shape of the model).
            Y: A `batch_shape' x n' x m`-dim Tensor, where `m` is the number of
                model outputs, `n'` is the number of points per batch, and
                `batch_shape'` is the batch shape of the observations.
                `batch_shape'` must be broadcastable to `batch_shape` using
                standard broadcasting semantics. If `Y` has fewer batch dimensions
                than `X`, its is assumed that the missing batch dimensions are
                the same for all `Y`.

        Returns:
            A `BatchedMultiOutputGPyTorchModel` object of the same type with
            `n + n'` training examples, representing the original model
            conditioned on the new observations `(X, Y)` (and possibly noise
            observations passed in via kwargs).

        Example:
            >>> train_X = torch.rand(20, 2)
            >>> train_Y = torch.cat(
            >>>     [torch.sin(train_X[:, 0]), torch.cos(train_X[:, 1])], -1
            >>> )
            >>> model = SingleTaskGP(train_X, train_Y)
            >>> new_X = torch.rand(5, 2)
            >>> new_Y = torch.cat([torch.sin(new_X[:, 0]), torch.cos(new_X[:, 1])], -1)
            >>> model = model.condition_on_observations(X=new_X, Y=new_Y)
        """
        noise = kwargs.get("noise")
        if hasattr(self, "outcome_transform"):
            # we need to apply transforms before shifting batch indices around
            Y, noise = self.outcome_transform(Y, noise)
        self._validate_tensor_args(X=X, Y=Y, Yvar=noise, strict=False)
        inputs = X
        if self._num_outputs > 1:
            inputs, targets, noise = multioutput_to_batch_mode_transform(
                train_X=X, train_Y=Y, num_outputs=self._num_outputs, train_Yvar=noise
            )
            # `multioutput_to_batch_mode_transform` removes the output dimension,
            # which is necessary for `condition_on_observations`
            targets = targets.unsqueeze(-1)
            if noise is not None:
                noise = noise.unsqueeze(-1)
        else:
            inputs = X
            targets = Y
        if noise is not None:
            kwargs.update({"noise": noise})
        fantasy_model = super().condition_on_observations(X=inputs, Y=targets, **kwargs)
        fantasy_model._input_batch_shape = fantasy_model.train_targets.shape[
            : (-1 if self._num_outputs == 1 else -2)
        ]
        fantasy_model._aug_batch_shape = fantasy_model.train_targets.shape[:-1]
        return fantasy_model

    def subset_output(self, idcs: List[int]) -> BatchedMultiOutputGPyTorchModel:
        r"""Subset the model along the output dimension.

        Args:
            idcs: The output indices to subset the model to.

        Returns:
            The current model, subset to the specified output indices.
        """
        try:
            subset_batch_dict = self._subset_batch_dict
        except AttributeError:
            raise NotImplementedError(
                "subset_output requires the model to define a `_subset_dict` attribute"
            )

        m = len(idcs)
        new_model = deepcopy(self)
        tidxr = torch.tensor(idcs, device=new_model.train_targets.device)
        idxr = tidxr if m > 1 else idcs[0]
        new_tail_bs = torch.Size([m]) if m > 1 else torch.Size()

        new_model._num_outputs = m
        new_model._aug_batch_shape = new_model._aug_batch_shape[:-1] + new_tail_bs
        new_model.train_inputs = tuple(
            ti[..., idxr, :, :] for ti in new_model.train_inputs
        )
        new_model.train_targets = new_model.train_targets[..., idxr, :]

        # adjust batch shapes of parameters/buffers if necessary
        for full_name, p in itertools.chain(
            new_model.named_parameters(), new_model.named_buffers()
        ):
            if full_name in subset_batch_dict:
                idx = subset_batch_dict[full_name]
                new_data = p.index_select(dim=idx, index=tidxr)
                if m == 1:
                    new_data = new_data.squeeze(idx)
                p.data = new_data
            mod_name = full_name.split(".")[:-1]
            mod_batch_shape(new_model, mod_name, m if m > 1 else 0)

        # subset outcome transform if present
        try:
            subset_octf = new_model.outcome_transform.subset_output(idcs=idcs)
            new_model.outcome_transform = subset_octf
        except AttributeError:
            pass

        return new_model


class ModelListGPyTorchModel(GPyTorchModel, ModelList, ABC):
    r"""Abstract base class for models based on multi-output GPyTorch models.

    This is meant to be used with a gpytorch ModelList wrapper for independent
    evaluation of submodels.

    :meta private:
    """

    @property
    def batch_shape(self) -> torch.Size:
        r"""The batch shape of the model.

        This is a batch shape from an I/O perspective, independent of the internal
        representation of the model (as e.g. in BatchedMultiOutputGPyTorchModel).
        For a model with `m` outputs, a `test_batch_shape x q x d`-shaped input `X`
        to the `posterior` method returns a Posterior object over an output of
        shape `broadcast(test_batch_shape, model.batch_shape) x q x m`.
        """
        batch_shapes = {ti[0].shape[:-2] for ti in self.train_inputs}
        if len(batch_shapes) > 1:
            msg = (
                f"Component models of {self.__class__.__name__} have different "
                "batch shapes"
            )
            try:
                broadcast_shape = torch.broadcast_shapes(*batch_shapes)
                warnings.warn(msg + ". Broadcasting batch shapes.")
                return broadcast_shape
            except RuntimeError:
                raise NotImplementedError(msg + " that are not broadcastble.")
        return next(iter(batch_shapes))

    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: Union[bool, Tensor] = False,
        posterior_transform: Optional[PosteriorTransform] = None,
        **kwargs: Any,
    ) -> GPyTorchPosterior:
        r"""Computes the posterior over model outputs at the provided points.

        Args:
            X: A `b x q x d`-dim Tensor, where `d` is the dimension of the
                feature space, `q` is the number of points considered jointly,
                and `b` is the batch dimension.
            output_indices: A list of indices, corresponding to the outputs over
                which to compute the posterior (if the model is multi-output).
                Can be used to speed up computation if only a subset of the
                model's outputs are required for optimization. If omitted,
                computes the posterior over all model outputs.
            observation_noise: If True, add the observation noise from the
                respective likelihoods to the posterior. If a Tensor of shape
                `(batch_shape) x q x m`, use it directly as the observation
                noise (with `observation_noise[...,i]` added to the posterior
                of the `i`-th model).
            posterior_transform: An optional PosteriorTransform.

        Returns:
            A `GPyTorchPosterior` or `FullyBayesianPosterior` object, representing
            `batch_shape` joint distributions over `q` points and the outputs selected
            by `output_indices` each. Includes measurement noise if
            `observation_noise` is specified.
        """
        self.eval()  # make sure model is in eval mode
        # input transforms are applied at `posterior` in `eval` mode, and at
        # `model.forward()` at the training time
        transformed_X = self.transform_inputs(X)
        mvn_gen: Iterator
        with gpt_posterior_settings():
            # only compute what's necessary
            if output_indices is not None:
                mvns = [self.forward_i(i, transformed_X[i]) for i in output_indices]
                if observation_noise is not False:
                    if torch.is_tensor(observation_noise):
                        lh_kwargs = [
                            {"noise": observation_noise[..., i]}
                            for i, lh in enumerate(self.likelihood.likelihoods)
                        ]
                    else:
                        lh_kwargs = [
                            {"noise": lh.noise.mean().expand(t_X.shape[:-1])}
                            if isinstance(lh, FixedNoiseGaussianLikelihood)
                            else {}
                            for t_X, lh in zip(
                                transformed_X, self.likelihood.likelihoods
                            )
                        ]
                    mvns = [
                        self.likelihood_i(i, mvn, transformed_X[i], **lkws)
                        for i, mvn, lkws in zip(output_indices, mvns, lh_kwargs)
                    ]
                mvn_gen = zip(output_indices, mvns)
            else:
                mvns = self(*transformed_X)
                if observation_noise is not False:
                    mvnX = [(mvn, transformed_X[i]) for i, mvn in enumerate(mvns)]
                    if torch.is_tensor(observation_noise):
                        mvns = self.likelihood(*mvnX, noise=observation_noise)
                    else:
                        mvns = self.likelihood(*mvnX)
                mvn_gen = enumerate(mvns)
        # apply output transforms of individual models if present
        mvns = []
        for i, mvn in mvn_gen:
            try:
                oct = self.models[i].outcome_transform
                tf_mvn = oct.untransform_posterior(GPyTorchPosterior(mvn)).mvn
            except AttributeError:
                tf_mvn = mvn
            mvns.append(tf_mvn)
        # return result as a GPyTorchPosteriors/FullyBayesianPosterior
        mvn = (
            mvns[0]
            if len(mvns) == 1
            else MultitaskMultivariateNormal.from_independent_mvns(mvns=mvns)
        )
        if any(is_fully_bayesian(m) for m in self.models):
            # mixing fully Bayesian and other GP models is currently not supported
            posterior = FullyBayesianPosterior(mvn=mvn)
        else:
            posterior = GPyTorchPosterior(mvn=mvn)
        if posterior_transform is not None:
            return posterior_transform(posterior)
        return posterior

    def condition_on_observations(self, X: Tensor, Y: Tensor, **kwargs: Any) -> Model:
        raise NotImplementedError()


class MultiTaskGPyTorchModel(GPyTorchModel, ABC):
    r"""Abstract base class for multi-task models based on GPyTorch models.

    This class provides the `posterior` method to models that implement a
    "long-format" multi-task GP in the style of `MultiTaskGP`.

    :meta private:
    """

    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: Union[bool, Tensor] = False,
        posterior_transform: Optional[PosteriorTransform] = None,
        **kwargs: Any,
    ) -> GPyTorchPosterior:
        r"""Computes the posterior over model outputs at the provided points.

        Args:
            X: A `q x d` or `batch_shape x q x d` (batch mode) tensor, where `d` is the
                dimension of the feature space (not including task indices) and
                `q` is the number of points considered jointly.
            output_indices: A list of indices, corresponding to the outputs over
                which to compute the posterior (if the model is multi-output).
                Can be used to speed up computation if only a subset of the
                model's outputs are required for optimization. If omitted,
                computes the posterior over all model outputs.
            observation_noise: If True, add observation noise from the respective
                likelihoods. If a Tensor, specifies the observation noise levels
                to add.
            posterior_transform: An optional PosteriorTransform.

        Returns:
            A `GPyTorchPosterior` object, representing `batch_shape` joint
            distributions over `q` points and the outputs selected by
            `output_indices`. Includes measurement noise if
            `observation_noise` is specified.
        """
        if output_indices is None:
            output_indices = self._output_tasks
        num_outputs = len(output_indices)
        if any(i not in self._output_tasks for i in output_indices):
            raise ValueError("Too many output indices")
        cls_name = self.__class__.__name__

        # construct evaluation X
        X_full = _make_X_full(X=X, output_indices=output_indices, tf=self._task_feature)

        self.eval()  # make sure model is in eval mode
        # input transforms are applied at `posterior` in `eval` mode, and at
        # `model.forward()` at the training time
        X_full = self.transform_inputs(X_full)
        with gpt_posterior_settings():
            mvn = self(X_full)
            if observation_noise is not False:
                raise NotImplementedError(
                    f"Specifying observation noise is not yet supported by {cls_name}"
                )
        # If single-output, return the posterior of a single-output model
        if num_outputs == 1:
            posterior = GPyTorchPosterior(mvn=mvn)
        else:
            # Otherwise, make a MultitaskMultivariateNormal out of this
            mtmvn = MultitaskMultivariateNormal(
                mean=mvn.mean.view(*mvn.mean.shape[:-1], num_outputs, -1).transpose(
                    -1, -2
                ),
                covariance_matrix=mvn.lazy_covariance_matrix,
                interleaved=False,
            )
            posterior = GPyTorchPosterior(mvn=mtmvn)
        if hasattr(self, "outcome_transform"):
            posterior = self.outcome_transform.untransform_posterior(posterior)
        if posterior_transform is not None:
            return posterior_transform(posterior)
        return posterior
