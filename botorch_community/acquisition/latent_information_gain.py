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


class LatentInformationGain(AcquisitionFunction):
    def __init__(
        self,
        model: Type[Any],
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
        device = candidate_x.device
        candidate_x = candidate_x.to(device)
        N, q, D = candidate_x.shape
        kl = torch.zeros(N, device=device)
        if isinstance(self.model, NeuralProcessModel):
            x_c, y_c, _, _ = self.model.random_split_context_target(
                self.model.train_X, self.model.train_Y, self.model.n_context
            )
            z_mu_context, z_logvar_context = self.model.data_to_z_params(x_c, y_c)

            for i in range(N):
                x_i = candidate_x[i]
                kl_i = 0.0

                for _ in range(self.num_samples):
                    sample_z = self.model.sample_z(z_mu_context, z_logvar_context)
                    if sample_z.dim() == 1:
                        sample_z = sample_z.unsqueeze(0)

                    y_pred = self.model.decoder(x_i, sample_z)

                    combined_x = torch.cat([x_c, x_i], dim=0)
                    combined_y = torch.cat([y_c, y_pred], dim=0)

                    z_mu_post, z_logvar_post = self.model.data_to_z_params(
                        combined_x, combined_y
                    )

                    std_prior = self.min_std + self.scaler * torch.sigmoid(
                        z_logvar_context
                    )
                    std_post = self.min_std + self.scaler * torch.sigmoid(z_logvar_post)

                    p = torch.distributions.Normal(z_mu_post, std_post)
                    q = torch.distributions.Normal(z_mu_context, std_prior)
                    kl_sample = torch.distributions.kl_divergence(p, q).sum()

                    kl_i += kl_sample

                kl[i] = kl_i / self.num_samples

        else:
            for i in range(N):
                x_i = candidate_x[i]
                kl_i = 0.0
                for _ in range(self.num_samples):
                    posterior_prior = self.model.posterior(self.model.train_X)
                    posterior_candidate = self.model.posterior(x_i)

                    kl_i += torch.distributions.kl_divergence(
                        posterior_candidate.mvn, posterior_prior.mvn
                    ).sum()

                kl[i] = kl_i / self.num_samples
        return kl
