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

from typing import Any, Type

import torch
from botorch.acquisition import AcquisitionFunction
from botorch_community.models.np_regression import NeuralProcessModel
from torch import Tensor
# reference: https://arxiv.org/abs/2106.02770

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LatentInformationGain(AcquisitionFunction):
    def __init__(
        self,
        model: Type[Any] = NeuralProcessModel,
        num_samples: int = 10,
        min_std: float = 0.01,
        scaler: float = 0.5,
    ) -> None:
        """
        Latent Information Gain (LIG) Acquisition Function.
        Uses the model's built-in posterior function to generalize KL computation.

        Args:
            model: The model class to be used, defaults to NeuralProcessModel.
            num_samples: Int showing the # of samples for calculation, defaults to 10.
            min_std: Float representing the minimum possible standardized std,
                defaults to 0.01.
            scaler: Float scaling the std, defaults to 0.5.
        """
        super().__init__(model)
        self.model = model
        self.num_samples = num_samples
        self.min_std = min_std
        self.scaler = scaler

    def forward(self, candidate_x: Tensor) -> Tensor:
        """
        Conduct the Latent Information Gain acquisition function using the model's
            posterior.

        Args:
            candidate_x: Candidate input points, as a Tensor. Ideally in the shape
                (N, q, D).

        Returns:
            torch.Tensor: The LIG scores of computed KLDs, in the shape (N, q).
        """
        candidate_x = candidate_x.to(device)
        if candidate_x.dim() == 2:
            candidate_x = candidate_x.unsqueeze(0)  # Ensure (N, q, D) format
        N, q, D = candidate_x.shape

        kl = torch.zeros(N, q, device=device)

        if isinstance(self.model, NeuralProcessModel):
            x_c, y_c, x_t, y_t = self.model.random_split_context_target(
                self.model.train_X,
                self.model.train_Y,
                self.model.n_context
            )
            z_mu_context, z_logvar_context = self.model.data_to_z_params(x_c, y_c)
            for _ in range(self.num_samples):
                # Taking Samples/Predictions
                samples = self.model.sample_z(z_mu_context, z_logvar_context)
                y_pred = self.model.decoder(candidate_x.view(-1, D), samples)
                # Combining the data
                combined_x = torch.cat([x_c, candidate_x.view(-1, D)], dim=0).to(device)
                combined_y = torch.cat([y_c, y_pred], dim=0).to(device)
                # Computing posterior variables
                z_mu_posterior, z_logvar_posterior = self.model.data_to_z_params(
                    combined_x, combined_y
                )
                std_prior = self.min_std + self.scaler * torch.sigmoid(z_logvar_context)
                std_posterior = self.min_std + self.scaler * torch.sigmoid(
                    z_logvar_posterior
                )
                p = torch.distributions.Normal(z_mu_posterior, std_posterior)
                q = torch.distributions.Normal(z_mu_context, std_prior)
                kl_divergence = torch.distributions.kl_divergence(p, q).sum(dim=-1)
                kl += kl_divergence
        else:
            for _ in range(self.num_samples):
                posterior_prior = self.model.posterior(self.model.train_X)
                posterior_candidate = self.model.posterior(candidate_x.view(-1, D))

                kl_divergence = torch.distributions.kl_divergence(
                    posterior_candidate.mvn, posterior_prior.mvn
                ).sum(dim=-1)
                kl += kl_divergence

        return kl / self.num_samples
