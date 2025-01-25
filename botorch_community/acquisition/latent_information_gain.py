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
from botorch.acquisition import AcquisitionFunction
from botorch_community.models.np_regression import NeuralProcessModel
from torch import Tensor

import torch
#reference: https://arxiv.org/abs/2106.02770 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LatentInformationGain(AcquisitionFunction):
    def __init__(
        self, 
        context_x: torch.Tensor, 
        context_y: torch.Tensor,
        model: NeuralProcessModel, 
        num_samples: int = 10,
        min_std: float = 0.01,
        scaler: float = 0.5
    ) -> None:
        """
        Latent Information Gain (LIG) Acquisition Function, designed for the
        NeuralProcessModel. This is a subclass of AcquisitionFunction.

        Args:
            model: Trained NeuralProcessModel.
            context_x: Context input points, as a Tensor.
            context_y: Context target points, as a Tensor.
            num_samples (int): Number of samples for calculation, defaults to 10.
            min_std: Float representing the minimum possible standardized std, defaults to 0.1.
            scaler: Float scaling the std, defaults to 0.9.
        """
        super().__init__(model=model)
        self.model = model.to(device)
        self.num_samples = num_samples
        self.min_std = min_std
        self.scaler = scaler
        self.context_x = context_x.to(device)
        self.context_y = context_y.to(device)

    def forward(self, candidate_x):
        """
        Conduct the Latent Information Gain acquisition function for the inputs.

        Args:
            candidate_x: Candidate input points, as a Tensor.

        Returns:
            torch.Tensor: The LIG score of computed KLDs.
        """

        candidate_x = candidate_x.to(device)
        
        # Encoding and Scaling the context data
        z_mu_context, z_logvar_context = self.model.data_to_z_params(self.context_x, self.context_y)
        kl = 0.0
        for _ in range(self.num_samples):
            # Taking reparameterized samples
            samples = self.model.sample_z(z_mu_context, z_logvar_context)

            # Using the Decoder to take predicted values
            y_pred = self.model.decoder(candidate_x, samples)

            # Combining context and candidate data
            combined_x = torch.cat([self.context_x, candidate_x], dim=0).to(device)
            combined_y = torch.cat([self.context_y, y_pred], dim=0).to(device)

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
