#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Latent Information Gain Acquisition Function for Neural Process Models.

References:

.. [Wu2023arxiv]
   Wu, D., Niu, R., Chinazzi, M., Vespignani, A., Ma, Y.-A., & Yu, R. (2023).
   Deep Bayesian Active Learning for Accelerating Stochastic Simulation.
   arXiv preprint arXiv:2106.02770. Retrieved from https://arxiv.org/abs/2106.02770

Contributor: eibarolle
"""

from __future__ import annotations

import warnings
from typing import Optional

import torch
from botorch import settings
from botorch_community.models.np_regression import NeuralProcessModel
from torch import Tensor

import torch
#reference: https://arxiv.org/abs/2106.02770 

class LatentInformationGain:
    def __init__(
        self, 
        model: NeuralProcessModel, 
        num_samples: int = 10,
        min_std: float = 0.1,
        scaler: float = 0.9
    ) -> None:
        """
        Latent Information Gain (LIG) Acquisition Function, designed for the
        NeuralProcessModel.

        Args:
            model: Trained NeuralProcessModel.
            num_samples (int): Number of samples for calculation, defaults to 10.
            min_std: Float representing the minimum possible standardized std, defaults to 0.1.
            scaler: Float scaling the std, defaults to 0.9.
        """
        self.model = model
        self.num_samples = num_samples
        self.min_std = min_std
        self.scaler = scaler

    def acquisition(self, candidate_x, context_x, context_y):
        """
        Conduct the Latent Information Gain acquisition function for the inputs.

        Args:
            candidate_x: Candidate input points, as a Tensor.
            context_x: Context input points, as a Tensor.
            context_y: Context target points, as a Tensor.

        Returns:
            torch.Tensor: The LIG score of computed KLDs.
        """

        # Encoding and Scaling the context data
        z_mu_context, z_logvar_context = self.model.data_to_z_params(context_x, context_y)
        kl = 0.0
        for _ in range(self.num_samples):
            # Taking reparameterized samples
            samples = self.model.sample_z(z_mu_context, z_logvar_context)

            # Using the Decoder to take predicted values
            y_pred = self.model.decoder(candidate_x, samples)

            # Combining context and candidate data
            combined_x = torch.cat([context_x, candidate_x], dim=0)
            combined_y = torch.cat([context_y, y_pred], dim=0)

            # Computing posterior variables
            z_mu_posterior, z_logvar_posterior = self.model.data_to_z_params(combined_x, combined_y)
            std_prior = self.min_std + self.scaler * torch.sigmoid(z_logvar_context) 
            std_posterior = self.min_std + self.scaler * torch.sigmoid(z_logvar_posterior)

            p = torch.distributions.Normal(z_mu_posterior, std_posterior)
            q = torch.distributions.Normal(z_mu_context, std_prior)

            kl_divergence = torch.distributions.kl_divergence(p, q).sum()
            kl += kl_divergence

        # Average KLD
        return kl / self.num_samples
