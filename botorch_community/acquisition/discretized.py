#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Abstract base module for all botorch acquisition functions."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from botorch.acquisition import AcquisitionFunction
from botorch.models.model import Model
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor


class DiscretizedAcquistionFunction(AcquisitionFunction, ABC):
    r"""DiscretizedAcquistionFunction is an abstract base class for acquisition
    functions that are defined on discrete distributions. It wraps a model and
    implements a forward method that computes the acquisition function value at
    a given set of points.
    This class can be subclassed to define acquisiton functions for Riemann-
    distributed posteriors.
    The acquisition function must have the form $$acq(x) = \int p(y|x) ag(x)$$,
    where $$ag$$ is defined differently for each acquisition function.
    The ag_integrate method, which computes the integral of ag between two points, must
    be implemented by subclasses to define the specific acquisition functions.
    """

    def __init__(self, model: Model) -> None:
        r"""
        Initialize the DiscretizedAcquistionFunction

        Args:
            model: A fitted model that is used to compute the posterior
                distribution over the outcomes of interest.
                The model should be a `PFNModel`.
        """

        super().__init__(model=model)

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the acquisition function on the candidate set X.

        Args:
            X: A `(b) x q x d`-dim Tensor of `(b)` t-batches with `q` `d`-dim
                design points each.

        Returns:
            A `(b)`-dim Tensor of the acquisition function at the given
            design points `X`.
        """
        self.to(device=X.device)

        discrete_posterior = self.model.posterior(X)
        result = discrete_posterior.integrate(self.ag_integrate)
        # remove q dimension
        return result.squeeze(-1)

    @abstractmethod
    def ag_integrate(self, lower_bound: Tensor, upper_bound: Tensor) -> Tensor:
        r"""
        This function calculates the integral that computes the acquisition function
        without the posterior factor from lower_bound to upper_bound.
        That is, our acquisition function is assumed to have the form
        \int ag(x) * p(x) dx,
        and this function calculates \int_{lower_bound}^{upper_bound} ag(x) dx.
        The `integrate` method of the posterior (`BoundedRiemannPosterior`)
        then computes the final acquisition value.

        Args:
            lower_bound: lower bound of integral
            upper_bound: upper bound of integral

        Returns:
            A `(b)`-dim Tensor of acquisition function derivatives at the given
            design points `X`.
        """
        pass  # pragma: no cover

    r"""DiscretizedExpectedImprovement is an acquisition function that computes
    the expected improvement over the current best observed value for a Riemann
    distribution."""


class DiscretizedExpectedImprovement(DiscretizedAcquistionFunction):
    r"""DiscretizedExpectedImprovement is an acquisition function that
    computes the expected improvement over the current best observed value
    for a Riemann distribution.
    """

    def __init__(self, model: Model, best_f: Tensor) -> None:
        r"""
        Initialize the DiscretizedExpectedImprovement

        Args:
            model: A fitted model that is used to compute the posterior
                distribution over the outcomes of interest.
                The model should be a `PFNModel`.
            best_f: A tensor representing the current best observed value.
        """
        super().__init__(model)
        self.register_buffer("best_f", torch.as_tensor(best_f))

    def ag_integrate(self, lower_bound: Tensor, upper_bound: Tensor) -> Tensor:
        r"""
        As Expected improvement has ag(y) = (y - best_f).clamp(min=0), and
        is defined as \int ag(y) * p(y) dy, we can calculate the integral
        of ag(y) like so:
        We just calculate ag(y) at beginning and end, and since the function has
        a gradient of 1 or 0, we can just take the average of the two.

        Args:
            lower_bound: lower bound of integral
            upper_bound: upper bound of integral

        Returns:
            A `(b)`-dim Tensor of acquisition function derivatives at the given
            design points `X`.
        """
        max_lower_bound_and_f = torch.max(self.best_f, lower_bound)
        bucket_average = (upper_bound + max_lower_bound_and_f) / 2
        improvement = bucket_average - self.best_f

        return improvement.clamp_min(0)


class DiscretizedProbabilityOfImprovement(DiscretizedAcquistionFunction):
    r"""DiscretizedProbabilityOfImprovement is an acquisition function that
    computes the probability of improvement over the current best observed value
    for a Riemann distribution.
    """

    def __init__(self, model: Model, best_f: Tensor) -> None:
        r"""
        Initialize the DiscretizedProbabilityOfImprovement

        Args:
            model: A fitted model that is used to compute the posterior
                distribution over the outcomes of interest.
                The model should be a `PFNModel`.
            best_f: A tensor representing the current best observed value.
        """

        super().__init__(model)
        self.register_buffer("best_f", torch.as_tensor(best_f))

    def ag_integrate(self, lower_bound: Tensor, upper_bound: Tensor) -> Tensor:
        r"""
        PI is defined as \int ag(y) * p(y) dy, where ag(y) = I(y - best_f)
        and I being the indicator function.

        So all we need to do is calculate the portion between the bounds
        that is larger than best_f.
        We do this by comparing how much higher the upper bound is than best_f,
        compared to the size of the bucket.
        Then we clamp at one if best_f is below lower_bound and at zero if
        best_f is above upper_bound.

        Args:
            lower_bound: lower bound of integral
            upper_bound: upper bound of integral

        Returns:
            A `(b)`-dim Tensor of acquisition function derivatives at the given
            design points `X`.
        """
        proportion = (upper_bound - self.best_f) / (upper_bound - lower_bound)
        return proportion.clamp(0, 1)
