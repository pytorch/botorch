#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Acquisition functions for max-value entropy search for multi-objective
Bayesian optimization (MESMO).

References

.. [Belakaria2019]
    S. Belakaria, A. Deshwal, J. R. Doppa. Max-value Entropy Search
    for Multi-Objective Bayesian Optimization. Advances in Neural
    Information Processing Systems, 32. 2019.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch
from botorch.acquisition.max_value_entropy_search import qMaxValueEntropy
from botorch.acquisition.multi_objective.monte_carlo import (
    MultiObjectiveMCAcquisitionFunction,
)
from botorch.models.converter import (
    batched_multi_output_to_single_output,
    model_list_to_batched,
)
from botorch.models.model import Model
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.sampling.samplers import MCSampler, SobolQMCNormalSampler
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor


class qMultiObjectiveMaxValueEntropy(
    qMaxValueEntropy, MultiObjectiveMCAcquisitionFunction
):
    r"""The acquisition function for MESMO.

    This acquisition function computes the mutual information of
    Pareto frontier and a candidate point. See [Belakaria2019]_ for
    a detailed discussion.

    q > 1 is supported through cyclic optimization and fantasies.

    Noisy observations are support by computing information gain with
    observation noise as in Appendix C in [Takeno2020mfmves]_.

    Note: this only supports maximization.

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> MESMO = qMultiObjectiveMaxValueEntropy(model, sample_pfs)
        >>> mesmo = MESMO(test_X)
    """

    def __init__(
        self,
        model: Model,
        sample_pareto_frontiers: Callable[[Model], Tensor],
        num_fantasies: int = 16,
        X_pending: Optional[Tensor] = None,
        sampler: Optional[MCSampler] = None,
        **kwargs: Any,
    ) -> None:
        r"""Multi-objective max-value entropy search acquisition function.

        Args:
            model: A fitted multi-output model.
            sample_pareto_frontiers: A callable that takes a model and returns a
                `num_samples x n' x m`-dim tensor of outcomes to use for constructing
                `num_samples` sampled Pareto frontiers.
            num_fantasies: Number of fantasies to generate. The higher this
                number the more accurate the model (at the expense of model
                complexity, wall time and memory). Ignored if `X_pending` is `None`.
            X_pending: A `m x d`-dim Tensor of `m` design points that have been
                submitted for function evaluation but have not yet been evaluated.
        """
        MultiObjectiveMCAcquisitionFunction.__init__(self, model=model, sampler=sampler)

        # Batch GP models (e.g. fantasized models) are not currently supported
        if isinstance(model, ModelListGP):
            train_X = model.models[0].train_inputs[0]
        else:
            train_X = model.train_inputs[0]
        if train_X.ndim > 3:
            raise NotImplementedError(
                "Batch GP models (e.g. fantasized models) "
                "are not yet supported by qMultiObjectiveMaxValueEntropy"
            )
        # convert to batched MO model
        batched_mo_model = (
            model_list_to_batched(model) if isinstance(model, ModelListGP) else model
        )
        self._init_model = batched_mo_model
        self.mo_model = batched_mo_model
        self.model = batched_multi_output_to_single_output(
            batch_mo_model=batched_mo_model
        )
        self.fantasies_sampler = SobolQMCNormalSampler(num_fantasies)
        self.num_fantasies = num_fantasies
        # weight is used in _compute_information_gain
        self.maximize = True
        self.weight = 1.0
        self.sample_pareto_frontiers = sample_pareto_frontiers

        # this avoids unnecessary model conversion if X_pending is None
        if X_pending is None:
            self._sample_max_values()
        else:
            self.set_X_pending(X_pending)
        # This avoids attribute errors in qMaxValueEntropy code.
        self.posterior_transform = None

    def set_X_pending(self, X_pending: Optional[Tensor] = None) -> None:
        r"""Set pending points.

        Informs the acquisition function about pending design points,
        fantasizes the model on the pending points and draws max-value samples
        from the fantasized model posterior.

        Args:
            X_pending: `m x d` Tensor with `m` `d`-dim design points that have
                been submitted for evaluation but have not yet been evaluated.
        """
        MultiObjectiveMCAcquisitionFunction.set_X_pending(self, X_pending=X_pending)
        if X_pending is not None:
            # fantasize the model
            fantasy_model = self._init_model.fantasize(
                X=X_pending, sampler=self.fantasies_sampler, observation_noise=True
            )
            self.mo_model = fantasy_model
            # convert model to batched single outcome model.
            self.model = batched_multi_output_to_single_output(
                batch_mo_model=self.mo_model
            )
            self._sample_max_values()
        else:
            # This is mainly for setting the model to the original model
            # after the sequential optimization at q > 1
            self.mo_model = self._init_model
            self.model = batched_multi_output_to_single_output(
                batch_mo_model=self.mo_model
            )
            self._sample_max_values()

    def _sample_max_values(self) -> None:
        r"""Sample max values for MC approximation of the expectation in MES"""
        with torch.no_grad():
            # num_samples x (num_fantasies) x n_pareto_points x m
            sampled_pfs = self.sample_pareto_frontiers(self.mo_model)
            if sampled_pfs.ndim == 3:
                # add fantasy dim
                sampled_pfs = sampled_pfs.unsqueeze(-3)
            # take component-wise max value
            self.posterior_max_values = sampled_pfs.max(dim=-2).values

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        r"""Compute max-value entropy at the design points `X`.

        Args:
            X: A `batch_shape x 1 x d`-dim Tensor of `batch_shape` t-batches
                with `1` `d`-dim design points each.

        Returns:
            A `batch_shape`-dim Tensor of MVE values at the given design points `X`.
        """
        # `m` dim tensor of information gains
        # unsqueeze X to add a batch-dim for the batched model
        igs = qMaxValueEntropy.forward(self, X=X.unsqueeze(-3))
        # sum over objectives
        return igs.sum(dim=-1)
