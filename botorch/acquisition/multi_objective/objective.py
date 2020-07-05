#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import abstractmethod
from typing import List, Optional

import torch
from botorch.acquisition.objective import AcquisitionObjective
from botorch.exceptions.errors import BotorchTensorDimensionError
from botorch.models.transforms.outcome import Standardize
from botorch.posteriors import GPyTorchPosterior
from botorch.utils.transforms import normalize_indices
from torch import Tensor


class MCMultiOutputObjective(AcquisitionObjective):
    r"""Abstract base class for multi-output objectives."""

    @abstractmethod
    def forward(self, samples: Tensor) -> Tensor:
        r"""Evaluate the multi-output objective on the samples.

        Args:
            samples: A `sample_shape x batch_shape x q x m`-dim Tensors of samples from
                a model posterior.

        Returns:
            A `sample_shape x batch_shape x q x m'`-dim Tensor of objective values with
            `m'` the output dimension. assuming maximization in each output dimension).

        This method is usually not called directly, but via the objectives

        Example:
            >>> # `__call__` method:
            >>> samples = sampler(posterior)
            >>> outcomes = multi_obj(samples)
        """
        pass  # pragma: no cover


class IdentityMCMultiOutputObjective(MCMultiOutputObjective):
    r"""Trivial objective extracting the last dimension.

    Example:
        >>> identity_objective = IdentityMCObjective()
        >>> samples = sampler(posterior)
        >>> objective = identity_objective(samples)
    """

    def forward(self, samples: Tensor) -> Tensor:
        return samples


class UnstandardizeMCMultiOutputObjective(MCMultiOutputObjective):
    r"""Objective that unstandardizes the samples.

    TODO: remove this when MultiTask models support outcome transforms.

    Example:
        >>> unstd_objective = UnstandardizeMCMultiOutputObjective(Y_mean, Y_std)
        >>> samples = sampler(posterior)
        >>> objective = unstd_objective(samples)
    """

    def __init__(
        self, Y_mean: Tensor, Y_std: Tensor, outcomes: Optional[List[int]] = None
    ) -> None:
        r"""Initialize objective.

        Args:
            Y_mean: `m`-dim tensor of outcome means.
            Y_std: `m`-dim tensor of outcome standard deviations.
            outcomes: A list of `m' <= m` indices that specifies which of the `m` model
                outputs should be considered as the outcomes for MOO. If omitted, use
                all model outcomes. Typically used for constrained optimization.
        """
        if Y_mean.ndim > 1 or Y_std.ndim > 1:
            raise BotorchTensorDimensionError(
                "Y_mean and Y_std must both be 1-dimensional, but got "
                f"{Y_mean.ndim} and {Y_std.ndim}"
            )
        super().__init__()
        if outcomes is not None:
            if len(outcomes) < 2:
                raise BotorchTensorDimensionError(
                    "Must specify at least two outcomes for MOO."
                )
            elif len(outcomes) > Y_mean.shape[-1]:
                raise BotorchTensorDimensionError(
                    f"Cannot specify more ({len(outcomes)}) outcomes that present in "
                    f"the normalization inputs ({Y_mean.shape[-1]})."
                )
            nlzd_idcs = normalize_indices(outcomes, Y_mean.shape[-1])
            self.register_buffer(
                "outcomes",
                torch.tensor(nlzd_idcs, dtype=torch.long).to(device=Y_mean.device),
            )
            Y_mean = Y_mean.index_select(-1, self.outcomes)
            Y_std = Y_std.index_select(-1, self.outcomes)

        self.register_buffer("Y_mean", Y_mean)
        self.register_buffer("Y_std", Y_std)

    def forward(self, samples: Tensor) -> Tensor:
        if hasattr(self, "outcomes"):
            samples = samples.index_select(-1, self.outcomes)
        return samples * self.Y_std + self.Y_mean


class AnalyticMultiOutputObjective(AcquisitionObjective):
    r"""Abstract base class for multi-output analyic objectives."""

    @abstractmethod
    def forward(self, posterior: GPyTorchPosterior) -> GPyTorchPosterior:
        r"""Transform the posterior

        Args:
            posterior: A posterior.

        Returns:
            A transformed posterior.
        """
        pass  # pragma: no cover


class IdentityAnalyticMultiOutputObjective(AnalyticMultiOutputObjective):
    def forward(self, posterior: GPyTorchPosterior) -> GPyTorchPosterior:
        return posterior


class UnstandardizeAnalyticMultiOutputObjective(AnalyticMultiOutputObjective):
    r"""Objective that unstandardizes the posterior.

    TODO: remove this when MultiTask models support outcome transforms.

    Example:
        >>> unstd_objective = UnstandardizeAnalyticMultiOutputObjective(Y_mean, Y_std)
        >>> unstd_posterior = unstd_objective(posterior)
    """

    def __init__(
        self, Y_mean: Tensor, Y_std: Tensor, outcomes: Optional[List[int]] = None
    ) -> None:
        r"""Initialize objective.

        Args:
            Y_mean: `m`-dim tensor of outcome means
            Y_std: `m`-dim tensor of outcome standard deviations
            outcomes: A list of `m' <= m` indices that specifies which of the `m` model
                outputs should be considered as the outcomes for MOO. If omitted, use
                all model outcomes. Typically used for constrained optimization.

        """
        if Y_mean.ndim > 1 or Y_std.ndim > 1:
            raise BotorchTensorDimensionError(
                "Y_mean and Y_std must both be 1-dimensional, but got "
                f"{Y_mean.ndim} and {Y_std.ndim}"
            )
        if outcomes is not None:
            if len(outcomes) < 2:
                raise BotorchTensorDimensionError(
                    "Must specify at least two outcomes for MOO."
                )
            elif len(outcomes) > Y_mean.shape[-1]:
                raise BotorchTensorDimensionError(
                    f"Cannot specify more ({len(outcomes)}) outcomes that present in "
                    f"the normalization inputs ({Y_mean.shape[-1]})."
                )
        super().__init__()
        self.outcome_transform = Standardize(m=Y_mean.shape[0], outputs=outcomes).to(
            Y_mean
        )
        Y_std_unsqueezed = Y_std.unsqueeze(0)
        self.outcome_transform.means = Y_mean.unsqueeze(0)
        self.outcome_transform.stdvs = Y_std_unsqueezed
        self.outcome_transform._stdvs_sq = Y_std_unsqueezed.pow(2)
        self.outcome_transform.eval()

    def forward(self, posterior: GPyTorchPosterior) -> Tensor:
        return self.outcome_transform.untransform_posterior(posterior)
