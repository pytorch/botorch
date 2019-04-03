#! /usr/bin/env python3

r"""
Abstract class for all GPyTorch-based botorch models.

To implement your own, simply inherit from both the provided classes and a
GPyTorch Model class such as an ExactGP.
"""

from abc import ABC, abstractproperty
from contextlib import ExitStack
from typing import List, Optional

import torch
from gpytorch import settings
from gpytorch.distributions import MultitaskMultivariateNormal
from torch import Tensor

from ..posteriors.gpytorch import GPyTorchPosterior
from .model import Model


class GPyTorchModel(Model, ABC):
    r"""Abstract base class for models based on GPyTorch models.

    The easiest way to use this is to subclass a model from a GPyTorch model
    class (e.g. an `ExactGP`) and this `GPyTorchModel`. See e.g. `SingleTaskGP`.
    """

    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: bool = False,
        detach_test_caches: bool = True,
        **kwargs,
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
            A `Posterior` object, representing a batch of `b` joint distributions
                over `q` points and the outputs selected by `output_indices` each.
                Includes measurement noise if `observation_noise=True`.
        """
        if output_indices is not None and output_indices != [0]:
            raise RuntimeError(
                "Cannot pass more than one output index to single-output model"
            )
        self.eval()
        with ExitStack() as es:
            es.enter_context(settings.debug(False))
            es.enter_context(settings.fast_pred_var())
            es.enter_context(settings.detach_test_caches(detach_test_caches))
            mvn = self(X)
            if observation_noise:
                mvn = self.likelihood(mvn, X)
        return GPyTorchPosterior(mvn=mvn)


class MultiOutputGPyTorchModel(GPyTorchModel, ABC):
    r"""Abstract base class for models based on multi-output GPyTorch models.

    This is meant to be used with a gpytorch ModelList wrapper for independent
    evaluation of submodels.
    """

    @abstractproperty
    def num_outputs(self) -> int:
        """The number of outputs of the model."""
        pass

    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: bool = False,
        detach_test_caches: bool = True,
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
            A `Posterior` object, representing a batch of `b` joint distributions
                over `q` points and the outputs selected by `output_indices` each.
                Includes measurement noise if `observation_noise=True`.
        """
        self.eval()
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
                    mvns = self.likelihood(*[(mvn, X) for mvn in mvns])
        if len(mvns) == 1:
            mvn = mvns[0]
        else:
            mvn = MultitaskMultivariateNormal.from_independent_mvns(mvns=mvns)
        return GPyTorchPosterior(mvn=mvn)


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
        detach_test_caches: bool = True,
    ) -> GPyTorchPosterior:
        r"""Computes the posterior over model outputs at the provided points.

        Args:
            X: A `q x d` or `b x q x d` (batch mode) tensor, where `d` is the
                dimension of the feature space (not including task indices),
                `q` is the number of points considered jointly, and `b` is the
                (optional) batch dimension.
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
            A `Posterior` object, representing a batch of `b` joint distributions
                over `q` points and the outputs selected by `output_indices`.
                Includes measurement noise if `observation_noise=True`.
        """
        if output_indices is None:
            output_indices = self._output_tasks
        elif isinstance(output_indices, int):
            output_indices = [output_indices]
        if any(i not in self._output_tasks for i in output_indices):
            raise ValueError("Too many output indices")

        # construct evaluation X
        X_full = _make_X_full(X=X, output_indices=output_indices, tf=self._task_feature)

        self.eval()
        with ExitStack() as es:
            es.enter_context(settings.debug(False))
            es.enter_context(settings.fast_pred_var())
            es.enter_context(settings.detach_test_caches(detach_test_caches))
            mvn = self(X_full)
            if observation_noise:
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


def _make_X_full(X: Tensor, output_indices: List[int], tf: int) -> Tensor:
    index_shape = X.shape[:-1] + torch.Size([1])
    indexers = (
        torch.full(index_shape, fill_value=i, device=X.device, dtype=X.dtype)
        for i in output_indices
    )
    X_l, X_r = X[..., :tf], X[..., tf:]
    return torch.cat(
        [torch.cat([X_l, indexer, X_r], dim=-1) for indexer in indexers], dim=0
    )
