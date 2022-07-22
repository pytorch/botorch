#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Preference acquisition functions. This includes:
Analytical EUBO acquisition function as introduced in [Lin2022preference]_.

.. [Lin2022preference]
    Lin, Z.J., Astudillo, R., Frazier, P.I. and Bakshy, E. Preference Exploration
    for Efficient Bayesian Optimization with Multiple Outcomes. International
    Conference on Artificial Intelligence and Statistics (AISTATS), 2022.
"""

from __future__ import annotations

from typing import Optional

import torch
from botorch.acquisition import AnalyticAcquisitionFunction
from botorch.exceptions.errors import UnsupportedError
from botorch.models.deterministic import DeterministicModel
from botorch.models.model import Model
from botorch.utils.transforms import match_batch_shape, t_batch_mode_transform
from torch import Tensor


class AnalyticExpectedUtilityOfBestOption(AnalyticAcquisitionFunction):
    r"""Analytic Prefential Expected Utility of Best Options, i.e., Analytical EUBO"""

    def __init__(
        self,
        pref_model: Model,
        outcome_model: Optional[DeterministicModel] = None,
        previous_winner: Optional[Tensor] = None,
    ) -> None:
        r"""Analytic implementation of Expected Utility of the Best Option under the
        Laplace model (assumes a PairwiseGP is used as the preference model) as
        proposed in [Lin2022preference]_.

        Args:
            pref_model: The preference model that maps the outcomes (i.e., Y) to
                scalar-valued utility.
            model: A deterministic model that maps parameters (i.e., X) to outcomes
                (i.e., Y). The outcome model f defines the search space of Y = f(X).
                If model is None, we are directly calculating EUBO on the parameter
                space. When used with `OneSamplePosteriorDrawModel`, we are obtaining
                EUBO-zeta as described in [Lin2022preference].
            previous_winner: Tensor representing the previous winner in the Y space.
        """
        pref_model.eval()
        super().__init__(model=pref_model)
        # ensure the model is in eval mode
        self.add_module("outcome_model", outcome_model)
        self.register_buffer("previous_winner", previous_winner)

        tkwargs = {
            "dtype": pref_model.datapoints.dtype,
            "device": pref_model.datapoints.device,
        }
        std_norm = torch.distributions.normal.Normal(
            torch.zeros(1, **tkwargs),
            torch.ones(1, **tkwargs),
        )
        self.std_norm = std_norm

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate analytical EUBO on the candidate set X.

        Args:
            X: A `batch_shape x q x d`-dim Tensor, where `q = 2` if `previous_winner`
                is not `None`, and `q = 1` otherwise.

        Returns:
            The acquisition value for each batch as a tensor of shape `batch_shape`.
        """
        if not (
            (X.shape[-2] == 2)
            or ((X.shape[-2] == 1) and (self.previous_winner is not None))
        ):
            raise UnsupportedError(
                f"{self.__class__.__name__} only support q=2 or q=1"
                "with a previous winner specified"
            )

        Y = X if self.outcome_model is None else self.outcome_model(X)

        if self.previous_winner is not None:
            Y = torch.cat([Y, match_batch_shape(self.previous_winner, Y)], dim=-2)

        # Calling forward directly instead of posterior here to
        # obtain the full covariance matrix
        pref_posterior = self.model(Y)
        pref_mean = pref_posterior.mean
        pref_cov = pref_posterior.covariance_matrix
        delta = pref_mean[..., 0] - pref_mean[..., 1]
        sigma = torch.sqrt(
            pref_cov[..., 0, 0]
            + pref_cov[..., 1, 1]
            - pref_cov[..., 0, 1]
            - pref_cov[..., 1, 0]
        )
        u = delta / sigma

        ucdf = self.std_norm.cdf(u)
        updf = torch.exp(self.std_norm.log_prob(u))
        acqf_val = sigma * (updf + u * ucdf)
        if self.previous_winner is None:
            acqf_val = acqf_val + pref_mean[..., 1]
        return acqf_val
