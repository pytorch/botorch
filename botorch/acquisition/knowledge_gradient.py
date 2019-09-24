#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

r"""
Batch Knowledge Gradient (KG) via one-shot optimization as introduced in
[Balandat2019botorch]_. For broader discussion of KG see also
[Frazier2008knowledge]_, [Wu2016parallelkg]_.

.. [Balandat2019botorch]
    M. Balandat, B. Karrer, D. R. Jiang, S. Daulton, B. Letham, A. G. Wilson,
    and E. Bakshy. BoTorch: Programmable Bayesian Optimziation in PyTorch.
    ArXiv 2019.

.. [Frazier2008knowledge]
    P. Frazier, W. Powell, and S. Dayanik. A Knowledge-Gradient policy for
    sequential information collection. SIAM Journal on Control and Optimization,
    2008.

.. [Wu2016parallelkg]
    J. Wu and P. Frazier. The parallel knowledge gradient method for batch
    bayesian optimization. NIPS 2016.
"""

from typing import Optional, Union

import torch
from torch import Tensor

from .. import settings
from ..models.model import Model
from ..sampling.samplers import MCSampler, SobolQMCNormalSampler
from ..utils.transforms import match_batch_shape
from .acquisition import AcquisitionFunction, OneShotAcquisitionFunction
from .analytic import PosteriorMean
from .monte_carlo import MCAcquisitionFunction, qSimpleRegret
from .objective import AcquisitionObjective, MCAcquisitionObjective, ScalarizedObjective


class qKnowledgeGradient(MCAcquisitionFunction, OneShotAcquisitionFunction):
    r"""Batch Knowledge Gradient using one-shot optimization.

    This computes the batch Knowledge Gradient using fantasies for the outer
    expectation and either the model posterior mean or MC-sampling for the inner
    expectation.

    In addition to the design variables, the input `X` also includes variables
    for the optimal designs for each of the fantasy models. For a fixed number
    of fantasies, all parts of `X` can be optimized in a "one-shot" fashion.
    """

    def __init__(
        self,
        model: Model,
        num_fantasies: Optional[int] = 64,
        sampler: Optional[MCSampler] = None,
        objective: Optional[AcquisitionObjective] = None,
        inner_sampler: Optional[MCSampler] = None,
        X_pending: Optional[Tensor] = None,
        current_value: Optional[Tensor] = None,
    ) -> None:
        r"""q-Knowledge Gradient (one-shot optimization).

        Args:
            model: A fitted model. Must support fantasizing.
            num_fantasies: The number of fantasy points to use. More fantasy
                points result in a better approximation, at the expense of
                memory and wall time. Unused if `sampler` is specified.
            sampler: The sampler used to sample fantasy observations. Optional
                if `num_fantasies` is specified.
            objective: The objective under which the samples are evaluated. If
                `None` or a ScalarizedObjective, then the analytic posterior mean
                is used, otherwise the objective is MC-evaluated (using
                inner_sampler).
            inner_sampler: The sampler used for inner sampling. Ignored if the
                objective is `None` or a ScalarizedObjective.
            X_pending: A `m x d`-dim Tensor of `m` design points that have
                points that have been submitted for function evaluation
                but have not yet been evaluated.
            current_value: The current value, i.e. the expected best objective
                given the observed points `D`. If omitted, forward will not
                return the actual KG value, but the expected best objective
                given the data set `D u X`.
        """
        if sampler is None:
            if num_fantasies is None:
                raise ValueError(
                    "Must specify `num_fantasies` if no `sampler` is provided."
                )
            # base samples should be fixed for joint optimization over X, X_fantasies
            sampler = SobolQMCNormalSampler(
                num_samples=num_fantasies, resample=False, collapse_batch_dims=True
            )
        elif num_fantasies is not None:
            if sampler.sample_shape != torch.Size([num_fantasies]):
                raise ValueError(
                    f"The sampler shape must match num_fantasies={num_fantasies}."
                )
        else:
            num_fantasies = sampler.sample_shape[0]
        super().__init__(model=model, sampler=sampler, X_pending=X_pending)
        # if not explicitly specified, we use the posterior mean for linear objs
        if isinstance(objective, MCAcquisitionObjective) and inner_sampler is None:
            inner_sampler = SobolQMCNormalSampler(
                num_samples=128, resample=False, collapse_batch_dims=True
            )
        self.inner_sampler = inner_sampler
        self.objective = objective
        self.num_fantasies = num_fantasies
        self.current_value = current_value

    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate qKnowledgeGradient on the candidate set `X`.

        Args:
            X: A `b x (q + num_fantasies) x d` Tensor with `b` t-batches of
                `q + num_fantasies` design points each. We split this X tensor
                into two parts in the `q` dimension (`dim=-2`). The first `q`
                are the q-batch of design points and the last num_fantasies are
                the current solutions of the inner optimization problem.

                `X_fantasies = X[..., -num_fantasies:, :]`
                `X_fantasies.shape = b x num_fantasies x d`

                `X_actual = X[..., :-num_fantasies, :]`
                `X_actual.shape = b x q x d`

        Returns:
            A Tensor of shape `b`. For t-batch b, the q-KG value of the design
                `X_actual[b]` is averaged across the fantasy models, where
                `X_fantasies[b, i]` is chosen as the final selection for the
                `i`-th fantasy model.
                NOTE: If `current_value` is not provided, then this is not the
                true KG value of `X_actual[b]`, and `X_fantasies[b, : ]` must be
                maximized at fixed `X_actual[b]`.
        """
        split_sizes = [X.size(-2) - self.num_fantasies, self.num_fantasies]
        X_actual, X_fantasies = torch.split(X, split_sizes, dim=-2)

        # X_fantasies is b x num_fantasies x d, needs to be num_fantasies x b x 1 x d
        # for batch mode evaluation with batch shape num_fantasies x b.
        # b x num_fantasies x d --> num_fantasies x b x d
        X_fantasies = X_fantasies.permute(-2, *range(X_fantasies.dim() - 2), -1)
        # num_fantasies x b x 1 x d
        X_fantasies = X_fantasies.unsqueeze(dim=-2)

        # We only concatenate X_pending into the X part after splitting
        if self.X_pending is not None:
            X_actual = torch.cat(
                [X_actual, match_batch_shape(self.X_pending, X_actual)], dim=-2
            )

        # construct the fantasy model of shape `num_fantasies x b`
        fantasy_model = self.model.fantasize(
            X=X_actual, sampler=self.sampler, observation_noise=True
        )
        value_function = _get_value_function(
            model=fantasy_model, objective=self.objective, sampler=self.inner_sampler
        )
        # we need to make sure to propagate gradients to the fantasy model train inputs
        with settings.propagate_grads(True):
            values = value_function(X=X_fantasies)  # num_fantasies x b

        # average over the fantasy samples
        result = values.mean(dim=0)

        if self.current_value is not None:
            result = result - self.current_value

        return result

    def get_augmented_q_batch_size(self, q: int) -> int:
        r"""Get augmented q batch size for one-shot optimzation.

        Args:
            q: The number of candidates to consider jointly.

        Returns:
            The augmented size for one-shot optimzation (including variables
            parameterizing the fantasy solutions).
        """
        return q + self.num_fantasies

    def extract_candidates(self, X_full: Tensor) -> Tensor:
        r"""We only return X as the set of candidates post-optimization.

        Args:
            X_full: A `b x (q + num_fantasies) x d`-dim Tensor with `b`
                t-batches of `q + num_fantasies` design points each.

        Returns:
            A `b x q x d`-dim Tensor with `b` t-batches of `q` design points each.
        """
        return X_full[..., : -self.num_fantasies, :]


def _get_value_function(
    model: Model,
    objective: Optional[Union[MCAcquisitionObjective, ScalarizedObjective]] = None,
    sampler: Optional[MCSampler] = None,
) -> AcquisitionFunction:
    r"""Construct value function (i.e. inner acquisition function)."""
    if isinstance(objective, MCAcquisitionObjective):
        return qSimpleRegret(model=model, sampler=sampler, objective=objective)
    else:
        return PosteriorMean(model=model, objective=objective)
