#! /usr/bin/env python3

r"""
Abstract model class for all GPyTorch-based botorch models.

To implement your own, simply inherit from both the provided classes and a
GPyTorch Model class such as an ExactGP.
"""

from abc import ABC, abstractproperty
from contextlib import ExitStack
from typing import Any, List, Optional

import torch
from gpytorch import settings
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from gpytorch.lazy import lazify
from torch import Tensor

from ..posteriors.gpytorch import GPyTorchPosterior
from .model import Model
from .utils import _make_X_full, add_output_dim


class GPyTorchModel(Model, ABC):
    r"""Abstract base class for models based on GPyTorch models.

    The easiest way to use this is to subclass a model from a GPyTorch model
    class (e.g. an `ExactGP`) and this `GPyTorchModel`. See e.g. `SingleTaskGP`.
    """

    def posterior(
        self, X: Tensor, observation_noise: bool = False, **kwargs: Any
    ) -> GPyTorchPosterior:
        r"""Computes the posterior over model outputs at the provided points.

        Args:
            X: A `(batch_shape) x q x d`-dim Tensor, where `d` is the dimension of the
                feature space and `q` is the number of points considered jointly.
            observation_noise: If True, add observation noise to the posterior.
            detach_test_caches: If True, detach GPyTorch test caches during
                computation of the posterior. Required for being able to compute
                derivatives with respect to training inputs at test time (used
                e.g. by qNoisyExpectedImprovement). Defaults to `True`.

        Returns:
            A `GPyTorchPosterior` object, representing a batch of `b` joint
            distributions over `q` points. Includes observation noise if
            `observation_noise=True`.
        """
        self.eval()  # make sure model is in eval mode
        detach_test_caches = kwargs.get("detach_test_caches", True)
        with ExitStack() as es:
            es.enter_context(settings.debug(False))
            es.enter_context(settings.fast_pred_var())
            es.enter_context(settings.detach_test_caches(detach_test_caches))
            mvn = self(X)
            if observation_noise:
                # TODO: Allow passing in observation noise via kwarg
                mvn = self.likelihood(mvn, X)
            return GPyTorchPosterior(mvn=mvn)


class BatchedMultiOutputGPyTorchModel(GPyTorchModel):
    r"""Base class for batched multi-output GPyTorch models with independent outputs.

    This model should be used when the same training data is used for all outputs.
    Outputs are modeled independently by using a different batch for each output.
    """

    _num_outputs: int
    _input_batch_shape: torch.Size
    _aug_batch_shape: torch.Size

    def _set_dimensions(self, train_X: Tensor, train_Y: Tensor) -> None:
        r"""Store the number of outputs and the batch shape.

        Args:
            train_X: A `n x d` or `batch_shape x n x d` (batch mode) tensor of training
                features.
            train_Y: A `n x (o)` or `batch_shape x n x (o)` (batch mode) tensor of
                training observations.
        """
        self._num_outputs = train_Y.shape[-1] if train_Y.dim() == train_X.dim() else 1
        self._input_batch_shape = train_X.shape[:-2]
        if self._num_outputs > 1:
            self._aug_batch_shape = (
                torch.Size([self._num_outputs]) + self._input_batch_shape
            )
        else:
            self._aug_batch_shape = self._input_batch_shape

    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: bool = False,
        **kwargs: Any,
    ) -> GPyTorchPosterior:
        r"""Computes the posterior over model outputs at the provided points.

        Args:
            X: A `(batch_shape) x q x d`-dim Tensor, where `d` is the dimension of the
                feature space and `q` is the number of points considered jointly.
            output_indices: A list of indices, corresponding to the outputs over
                which to compute the posterior (if the model is multi-output).
                Can be used to speed up computation if only a subset of the
                model's outputs are required for optimization. If omitted,
                computes the posterior over all model outputs.
            observation_noise: If True, add observation noise to the posterior.
            detach_test_caches: If True, detach GPyTorch test caches during
                computation of the posterior. Required for being able to compute
                derivatives with respect to training inputs at test time (used
                e.g. by qNoisyExpectedImprovement). Defaults to `True`.

        Returns:
            A `GPyTorchPosterior` object, representing `batch_shape` joint
            distributions over `q` points and the outputs selected by
            `output_indices` each. Includes observation noise if
            `observation_noise=True`.
        """
        self.eval()  # make sure model is in eval mode
        detach_test_caches = kwargs.get("detach_test_caches", True)
        with ExitStack() as es:
            es.enter_context(settings.debug(False))
            es.enter_context(settings.fast_pred_var())
            es.enter_context(settings.detach_test_caches(detach_test_caches))
            # insert a dimension for the output dimension
            if self._num_outputs > 1:
                X, output_dim_idx = add_output_dim(
                    X=X, original_batch_shape=self._input_batch_shape
                )
            mvn = self(X)
            mean_x = mvn.mean
            covar_x = mvn.covariance_matrix
            if self._num_outputs > 1:
                output_indices = output_indices or range(self._num_outputs)
                mvns = [
                    MultivariateNormal(
                        mean_x.select(dim=output_dim_idx, index=t),
                        lazify(covar_x.select(dim=output_dim_idx, index=t)),
                    )
                    for t in output_indices
                ]
                mvn = MultitaskMultivariateNormal.from_independent_mvns(mvns=mvns)
        return GPyTorchPosterior(mvn=mvn)


class ModelListGPyTorchModel(GPyTorchModel, ABC):
    r"""Abstract base class for models based on multi-output GPyTorch models.

    This is meant to be used with a gpytorch ModelList wrapper for independent
    evaluation of submodels.
    """

    @abstractproperty
    def num_outputs(self) -> int:
        r"""The number of outputs of the model."""
        pass  # pragma: no cover

    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: bool = False,
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
            observation_noise: If True, add observation noise to the posterior.
            detach_test_caches: If True, detach GPyTorch test caches during
                computation of the posterior. Required for being able to compute
                derivatives with respect to training inputs at test time (used
                e.g. by qNoisyExpectedImprovement).

        Returns:
            A `GPyTorchPosterior` object, representing `batch_shape` joint
            distributions over `q` points and the outputs selected by
            `output_indices` each. Includes measurement noise if
            `observation_noise=True`.
        """
        detach_test_caches = kwargs.get("detach_test_caches", True)
        self.eval()  # make sure model is in eval mode
        with ExitStack() as es:
            es.enter_context(settings.debug(False))
            es.enter_context(settings.fast_pred_var())
            es.enter_context(settings.detach_test_caches(detach_test_caches))
            if output_indices is not None:
                mvns = [self.forward_i(i, X) for i in output_indices]
                if observation_noise:
                    mvns = [
                        self.likelihood_i(i, mvn, X)
                        for i, mvn in zip(output_indices, mvns)
                    ]
            else:
                mvns = self(*[X for _ in range(self.num_outputs)])
                if observation_noise:
                    # TODO: Allow passing in observation noise via kwarg
                    mvns = self.likelihood(*[(mvn, X) for mvn in mvns])
        if len(mvns) == 1:
            return GPyTorchPosterior(mvn=mvns[0])
        else:
            return GPyTorchPosterior(
                mvn=MultitaskMultivariateNormal.from_independent_mvns(mvns=mvns)
            )


class MultiTaskGPyTorchModel(GPyTorchModel, ABC):
    r"""Abstract base class for multi-task models baed on GPyTorch models.

    This class provides the `posterior` method to models that implement a
    "long-format" multi-task GP in the style of `MultiTaskGP`.
    """

    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: bool = False,
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
            observation_noise: If True, add observation noise to the posterior.
            detach_test_caches: If True, detach GPyTorch test caches during
                computation of the posterior. Required for being able to compute
                derivatives with respect to training inputs at test time (used
                e.g. by qNoisyExpectedImprovement).

        Returns:
            A `GPyTorchPosterior` object, representing `batch_shape` joint
            distributions over `q` points and the outputs selected by
            `output_indices`. Includes measurement noise if
            `observation_noise=True`.
        """
        if output_indices is None:
            output_indices = self._output_tasks
        if any(i not in self._output_tasks for i in output_indices):
            raise ValueError("Too many output indices")

        # construct evaluation X
        X_full = _make_X_full(X=X, output_indices=output_indices, tf=self._task_feature)

        self.eval()  # make sure model is in eval mode
        detach_test_caches = kwargs.get("detach_test_caches", True)
        with ExitStack() as es:
            es.enter_context(settings.debug(False))
            es.enter_context(settings.fast_pred_var())
            es.enter_context(settings.detach_test_caches(detach_test_caches))
            mvn = self(X_full)
            if observation_noise:
                # TODO: Allow passing in observation noise via kwarg
                mvn = self.likelihood(mvn, X_full)
        # If single-output, return the posterior of a single-output model
        if len(output_indices) == 1:
            return GPyTorchPosterior(mvn=mvn)
        # Otherwise, make a MultitaskMultivariateNormal out of this
        mtmvn = MultitaskMultivariateNormal(
            mean=mvn.mean.view(*X.shape[:-1], len(output_indices)),
            covariance_matrix=mvn.lazy_covariance_matrix,
            interleaved=False,
        )
        return GPyTorchPosterior(mvn=mtmvn)
