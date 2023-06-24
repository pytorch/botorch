#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
This file has an implementation of Batch H-Entropy Search (qHES) via
one-shot optimization as introduced in [Neiswanger2022]_.

The authors adopt a generalized definition of entropy from past work
in Bayesian decision theory, which proposes a family of decision-theoretic
entropies parameterized by a problem-specific loss function and
action set. Therefore, the method allow the development of a common
acquisition optimization procedure, which applies generically
to many members of this family (where each member is induced by a
specific loss function and action set).

.. [Neiswanger2022]
    W. Neiswanger, L. Yu, S. Zhao, C. Meng, S. Ermon. Generalizing Bayesian
    Optimization with Decision-theoretic Entropies. Appears in Proceedings
    of the 36th Conference on Neural Information Processing Systems
    (NeurIPS 2022)
"""

from typing import Any, Dict, Optional, Type

import torch
import torch.nn as nn
from botorch.acquisition import MCAcquisitionObjective
from botorch.acquisition.acquisition import OneShotAcquisitionFunction
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.models.model import Model
from botorch.sampling.base import MCSampler
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor


class qHEntropySearch(MCAcquisitionFunction, OneShotAcquisitionFunction):
    def __init__(
        self,
        model: Model,
        loss_function_class: nn.Module,
        loss_function_hyperparameters: Dict[str, Any],
        n_fantasy_at_design_pts: Optional[int] = 64,
        n_fantasy_at_action_pts: Optional[int] = 64,
        design_sampler: Optional[MCSampler] = None,
        action_sampler: Optional[MCSampler] = None,
    ) -> None:
        r"""Batch H-Entropy Search using one-shot optimization.

        Args:
            model: A fitted model. Must support fantasizing.
            loss_function_class: The loss function class that is used to compute
                the expected loss of the fantasized actions.
            loss_function_hyperparameters: The hyperparameters for the loss
                function class.
            n_fantasy_at_design_pts: Number of fantasized outcomes for each
                design point. Must match the sample shape of `design_sampler`
                if specified.
            n_fantasy_at_action_pts: Number of fantasized outcomes for each
                action point. Must match the sample shape of `action_sampler`
                if specified.
            design_sampler: The sampler used to sample fantasized outcomes at each
                design point. Optional if `n_fantasy_at_design_pts` is specified.
            action_sampler: The sampler used to sample fantasized outcomes at each
                action point. Optional if `n_fantasy_at_design_pts` is specified.
        """

        super(MCAcquisitionFunction, self).__init__(model=model)

        if design_sampler is None:
            if n_fantasy_at_design_pts is None:
                raise ValueError(
                    "Must specify `n_fantasy_at_design_pts` if no `design_sampler` "
                    "is provided."
                )
            # base samples should be fixed for joint optimization over X, A
            design_sampler = SobolQMCNormalSampler(
                sample_shape=n_fantasy_at_design_pts,
                resample=False,
                collapse_batch_dims=True,
            )
        elif n_fantasy_at_design_pts is not None:
            if design_sampler.sample_shape != torch.Size([n_fantasy_at_design_pts]):
                raise ValueError(
                    "The design_sampler shape must match n_fantasy_at_design_pts="
                    f"{n_fantasy_at_design_pts}."
                )
        else:
            n_fantasy_at_design_pts = design_sampler.sample_shape[0]

        if action_sampler is None:
            if n_fantasy_at_action_pts is None:
                raise ValueError(
                    "Must specify `n_fantasy_at_action_pts` if no `action_sampler` "
                    "is provided."
                )
            # base samples should be fixed for joint optimization over X, A
            action_sampler = SobolQMCNormalSampler(
                sample_shape=n_fantasy_at_action_pts,
                resample=False,
                collapse_batch_dims=True,
            )
        elif n_fantasy_at_action_pts is not None:
            if action_sampler.sample_shape != torch.Size([n_fantasy_at_action_pts]):
                raise ValueError(
                    "The sampler shape must match n_fantasy_at_action_pts="
                    f"{n_fantasy_at_action_pts}."
                )
        else:
            n_fantasy_at_action_pts = action_sampler.sample_shape[0]

        self.design_sampler = design_sampler
        self.action_sampler = action_sampler
        self.n_fantasy_at_design_pts = n_fantasy_at_design_pts
        self.loss_function_hyperparameters = loss_function_hyperparameters
        self.loss_function = loss_function_class(
            **self.loss_function_hyperparameters,
        )

    @t_batch_mode_transform()
    def forward(self, X: Tensor, A: Tensor) -> Tensor:
        r"""Evaluate qHEntropySearch objective (q-HES) on the candidate set `X`.

        Args:
            X: Design tensor of shape `(batch) x q x num_dim_design`.
            A: Action tensor of shape `(batch) x n_fantasy_at_design_pts
                x num_actions x num_dim_action`.

        Returns:
            A Tensor of shape `(batch)`.
        """

        # construct the fantasy model of shape `n_fantasy_at_design_pts x b`
        fantasy_model = self.model.fantasize(
            X=X, sampler=self.design_sampler, observation_noise=True
        )

        # Permute shape of A to work with self.model.posterior correctly
        A = A.permute(1, 0, 2, 3)

        fantasized_outcome = self.action_sampler(fantasy_model.posterior(A))
        # >>> n_fantasy_at_action_pts x n_fantasy_at_design_pts
        # ... x batch_size x num_actions x 1

        fantasized_outcome = fantasized_outcome.squeeze(dim=-1)
        # >>> n_fantasy_at_action_pts x n_fantasy_at_design_pts
        # ... x batch_size x num_actions

        values = self.loss_function(A=A, Y=fantasized_outcome)
        # >>> n_fantasy_at_design_pts x batch_size

        # return average over the fantasy samples
        return values.mean(dim=0)

    def get_augmented_q_batch_size(self, q: int) -> int:
        return q + self.cfg.n_dim_action

    def extract_candidates(self, batch_as_full: Tensor) -> Tensor:
        assert len(batch_as_full.shape) == 2
        n_restarts = batch_as_full.size(0)
        split_sizes = [
            batch_as_full.size(1) - self.cfg.n_actions * self.cfg.n_dim_action,
            self.cfg.n_actions * self.cfg.n_dim_action,
        ]
        batch_xs, batch_as_full = torch.split(batch_as_full, split_sizes, dim=1)
        batch_xs = batch_xs.reshape(n_restarts, -1, self.cfg.n_dim_design)
        batch_as_full = batch_as_full.reshape(
            n_restarts, self.cfg.n_actions, self.cfg.n_dim_action
        )

        return batch_xs


class qLossFunctionTopK(nn.Module):
    def __init__(self, dist_weight=1.0, dist_threshold=0.5) -> None:
        r"""Batch loss function for the task of finding top-K.

        Args:
            loss_function_hyperparameters: hyperparameters for the loss function class.
        """

        super().__init__()
        self.register_buffer("dist_weight", torch.as_tensor(dist_weight))
        self.register_buffer(
            "dist_threshold",
            torch.as_tensor(dist_threshold),
        )

    def forward(self, A: Tensor, Y: Tensor) -> Tensor:
        r"""Evaluate batch loss function on a tensor of actions.

        Args:
            A: Actor tensor with shape `batch_size x n_fantasy_at_design_pts
                x num_actions x action_dim`.
            Y: Fantasized sample with shape `n_fantasy_at_action_pts x
                n_fantasy_at_design_pts x batch_size x num_actions`.

        Returns:
            A Tensor of shape `n_fantasy_at_action_pts x batch`.
        """

        Y = Y.sum(dim=-1).mean(dim=0)
        # >>> n_fantasy_at_design_pts x batch_size

        num_actions = A.shape[-2]

        dist_reward = 0
        if num_actions >= 2:
            dists = torch.nn.functional.pdist(A.contiguous(), p=1.0).clamp_min(
                self.dist_threshold
            )
            denominator = num_actions * (num_actions - 1) / 2.0
            dist_reward = dists.sum(-1) / denominator
            # >>> n_fantasy_at_design_pts x batch_size

        q_hes = Y + self.dist_weight * dist_reward
        # >>> n_fantasy_at_design_pts x batch_size

        return q_hes


class qLossFunctionMinMax(nn.Module):
    def __init__(self) -> None:
        r"""Loss function for task of finding min and max.

        Args:
            loss_function_hyperparameters: hyperparameters for the loss function class
        """
        super().__init__()

    def forward(self, A: Tensor, Y: Tensor) -> Tensor:
        r"""Evaluate batch loss function on a tensor of actions.

        Args:
            A: Actor tensor with shape `batch_size x n_fantasy_at_design_pts
                x num_actions x action_dim`.
            Y: Fantasized sample with shape `n_fantasy_at_action_pts x
                n_fantasy_at_design_pts x batch_size x num_actions`.

        Returns:
            A Tensor of shape `n_fantasy_at_action_pts x batch`.
        """

        if A.shape[-2] != 2:
            raise RuntimeError("qLossFunctionMinMax only supports 2 actions.")

        q_hes = (Y[..., 1:].sum(dim=-1) - Y[..., 0]).mean(dim=0)
        # >>> n_fantasy_at_design_pts x batch_size

        return q_hes
