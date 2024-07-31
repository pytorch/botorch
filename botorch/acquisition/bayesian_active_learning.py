# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Acquisition functions for Bayesian active learning. This includes:
BALD [Houlsby2011bald]_ and its batch version [kirsch2019batchbald]_.

References

.. [kirsch2019batchbald]
    Andreas Kirsch, Joost van Amersfoort, Yarin Gal.
    BatchBALD: Efficient and Diverse Batch Acquisition for Deep Bayesian
    Active Learning.
    In Proceedings of the Annual Conference on Neural Information
    Processing Systems (NeurIPS), 2019.

"""

from __future__ import annotations

from typing import Optional

import torch
from botorch.acquisition.acquisition import AcquisitionFunction, MCSamplerMixin
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.model import Model
from botorch.utils.transforms import concatenate_pending_points, t_batch_mode_transform
from torch import Tensor


class FullyBayesianAcquisitionFunction(AcquisitionFunction):
    def __init__(self, model: Model):
        """Base class for acquisition functions which require a Fully Bayesian
        model treatment.

        Args:
            model: A fully bayesian single-outcome model.
        """
        if model._is_fully_bayesian:
            super().__init__(model)

        else:
            raise ValueError(
                "Fully Bayesian acquisition functions require "
                "a SaasFullyBayesianSingleTaskGP to run."
            )


class qBayesianActiveLearningByDisagreement(
    FullyBayesianAcquisitionFunction, MCSamplerMixin
):
    def __init__(
        self,
        model: SaasFullyBayesianSingleTaskGP,
        X_pending: Optional[Tensor] = None,
    ) -> None:
        """
        Batch implementation [kirsch2019batchbald]_ of BALD [Houlsby2011bald]_,
        which maximizes the mutual information between the next observation and the
        hyperparameters of the model. Computed by informational lower bound.

        Args:
            model: A fully bayesian single-outcome model.
            X_pending: A `batch_shape, m x d`-dim Tensor of `m` design points.
        """
        super().__init__(model)
        self.set_X_pending(X_pending)

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate qBayesianActiveLearningByDisagreement on the candidate set `X`.

        Args:
            X: `batch_shape x q x D`-dim Tensor of input points.

        Returns:
            A `batch_shape x num_models`-dim Tensor of BALD values.
        """
        return self._compute_lower_bound_information_gain(X)

    def _compute_lower_bound_information_gain(self, X: Tensor) -> Tensor:
        r"""Evaluates the lower bounded information gain on the candidate set `X`.

        Args:
            X: `batch_shape x q x D`-dim Tensor of input points.

        Returns:
            A `batch_shape x num_models`-dim Tensor of information gains.
        """
        posterior = self.model.posterior(X, observation_noise=True)
        marg_covar = posterior.mixture_covariance_matrix
        cond_variances = posterior.variance

        prev_entropy = torch.logdet(marg_covar).unsqueeze(-1)
        # squeeze excess dim and mean over q-batch
        post_ub_entropy = torch.log(cond_variances).squeeze(-1).mean(-1)

        return prev_entropy - post_ub_entropy
