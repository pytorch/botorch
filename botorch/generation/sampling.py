#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Sampling-based generation strategies.

A SamplingStrategy returns samples from the input points (i.e. Tensors in feature
space), rather than the value for a set of tensors, as acquisition functions do.
The q-batch dimension has similar semantics as for acquisition functions in that the
points across the q-batch are considered jointly for sampling (where as for
q-acquisition functions we evaluate the joint value of the q-batch).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Union

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.objective import (
    IdentityMCObjective,
    MCAcquisitionObjective,
    PosteriorTransform,
    ScalarizedPosteriorTransform,
)
from botorch.generation.utils import _flip_sub_unique
from botorch.models.model import Model

from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.multitask import MultiTaskGP
from botorch.utils.sampling import batched_multinomial
from botorch.utils.transforms import standardize
from torch import Tensor
from torch.nn import Module


class SamplingStrategy(Module, ABC):
    r"""
    Abstract base class for sampling-based generation strategies.

    :meta private:
    """

    @abstractmethod
    def forward(self, X: Tensor, num_samples: int = 1, **kwargs: Any) -> Tensor:
        r"""Sample according to the SamplingStrategy.

        Args:
            X: A `batch_shape x N x d`-dim Tensor from which to sample (in the `N`
                dimension).
            num_samples: The number of samples to draw.
            kwargs: Additional implementation-specific kwargs.

        Returns:
            A `batch_shape x num_samples x d`-dim Tensor of samples from `X`, where
            `X[..., i, :]` is the `i`-th sample.
        """

        pass  # pragma: no cover


class MaxPosteriorSampling(SamplingStrategy):
    r"""Sample from a set of points according to their max posterior value.

    Example:
        >>> MPS = MaxPosteriorSampling(model)  # model w/ feature dim d=3
        >>> X = torch.rand(2, 100, 3)
        >>> sampled_X = MPS(X, num_samples=5)
    """

    def __init__(
        self,
        model: Model,
        objective: Optional[MCAcquisitionObjective] = None,
        posterior_transform: Optional[PosteriorTransform] = None,
        replacement: bool = True,
    ) -> None:
        r"""Constructor for the SamplingStrategy base class.

        Args:
            model: A fitted model.
            objective: The MCAcquisitionObjective under which the samples are
                evaluated. Defaults to `IdentityMCObjective()`.
            posterior_transform: An optional PosteriorTransform.
            replacement: If True, sample with replacement.
        """
        super().__init__()
        self.model = model
        if objective is None:
            objective = IdentityMCObjective()
        elif not isinstance(objective, MCAcquisitionObjective):
            # TODO: Clean up once ScalarizedObjective is removed.
            if posterior_transform is not None:
                raise RuntimeError(
                    "A ScalarizedObjective (DEPRECATED) and a posterior transform "
                    "are not supported at the same time. Use only a posterior "
                    "transform instead."
                )
            else:
                posterior_transform = ScalarizedPosteriorTransform(
                    weights=objective.weights, offset=objective.offset
                )
                objective = IdentityMCObjective()
        self.objective = objective
        self.posterior_transform = posterior_transform
        self.replacement = replacement

    def forward(
        self, X: Tensor, num_samples: int = 1, observation_noise: bool = False
    ) -> Tensor:
        r"""Sample from the model posterior.

        Args:
            X: A `batch_shape x N x d`-dim Tensor from which to sample (in the `N`
                dimension) according to the maximum posterior value under the objective.
            num_samples: The number of samples to draw.
            observation_noise: If True, sample with observation noise.

        Returns:
            A `batch_shape x num_samples x d`-dim Tensor of samples from `X`, where
            `X[..., i, :]` is the `i`-th sample.
        """
        posterior = self.model.posterior(
            X,
            observation_noise=observation_noise,
            posterior_transform=self.posterior_transform,
        )
        # num_samples x batch_shape x N x m
        samples = posterior.rsample(sample_shape=torch.Size([num_samples]))
        return self.maximize_samples(X, samples, num_samples)

    def maximize_samples(self, X: Tensor, samples: Tensor, num_samples: int = 1):
        obj = self.objective(samples, X=X)  # num_samples x batch_shape x N
        if self.replacement:
            # if we allow replacement then things are simple(r)
            idcs = torch.argmax(obj, dim=-1)
        else:
            # if we need to deduplicate we have to do some tensor acrobatics
            # first we get the indices associated w/ the num_samples top samples
            _, idcs_full = torch.topk(obj, num_samples, dim=-1)
            # generate some indices to smartly index into the lower triangle of
            # idcs_full (broadcasting across batch dimensions)
            ridx, cindx = torch.tril_indices(num_samples, num_samples)
            # pick the unique indices in order - since we look at the lower triangle
            # of the index matrix and we don't sort, this achieves deduplication
            sub_idcs = idcs_full[ridx, ..., cindx]
            if sub_idcs.ndim == 1:
                idcs = _flip_sub_unique(sub_idcs, num_samples)
            elif sub_idcs.ndim == 2:
                # TODO: Find a better way to do this
                n_b = sub_idcs.size(-1)
                idcs = torch.stack(
                    [_flip_sub_unique(sub_idcs[:, i], num_samples) for i in range(n_b)],
                    dim=-1,
                )
            else:
                # TODO: Find a general way to do this efficiently.
                raise NotImplementedError(
                    "MaxPosteriorSampling without replacement for more than a single "
                    "batch dimension is not yet implemented."
                )
        # idcs is num_samples x batch_shape, to index into X we need to permute for it
        # to have shape batch_shape x num_samples
        if idcs.ndim > 1:
            idcs = idcs.permute(*range(1, idcs.ndim), 0)
        # in order to use gather, we need to repeat the index tensor d times
        idcs = idcs.unsqueeze(-1).expand(*idcs.shape, X.size(-1))
        # now if the model is batched batch_shape will not necessarily be the
        # batch_shape of X, so we expand X to the proper shape
        Xe = X.expand(*obj.shape[1:], X.size(-1))
        # finally we can gather along the N dimension
        return torch.gather(Xe, -2, idcs)


class BoltzmannSampling(SamplingStrategy):
    r"""Sample from a set of points according to a tempered acquisition value.

    Given an acquisition function `acq_func`, this sampling strategies draws
    samples from a `batch_shape x N x d`-dim tensor `X` according to a multinomial
    distribution over its indices given by

        weight(X[..., i, :]) ~ exp(eta * standardize(acq_func(X[..., i, :])))

    where `standardize(Y)` standardizes `Y` to zero mean and unit variance. As the
    temperature parameter `eta -> 0`, this approaches uniform sampling, while as
    `eta -> infty`, this approaches selecting the maximizer(s) of the acquisition
    function `acq_func`.

    Example:
        >>> UCB = UpperConfidenceBound(model, beta=0.1)
        >>> BMUCB = BoltzmannSampling(UCB, eta=0.5)
        >>> X = torch.rand(2, 100, 3)
        >>> sampled_X = BMUCB(X, num_samples=5)
    """

    def __init__(
        self, acq_func: AcquisitionFunction, eta: float = 1.0, replacement: bool = True
    ) -> None:
        r"""Boltzmann Acquisition Value Sampling.

        Args:
            acq_func: The acquisition function; to be evaluated in batch at the
                individual points of a q-batch (not jointly, as is the case for
                acquisition functions). Can be analytic or Monte-Carlo.
            eta: The temperature parameter in the softmax.
            replacement: If True, sample with replacement.
        """
        super().__init__()
        self.acq_func = acq_func
        self.eta = eta
        self.replacement = replacement

    def forward(self, X: Tensor, num_samples: int = 1) -> Tensor:
        r"""Sample from a tempered value of the acquisition function value.

        Args:
            X: A `batch_shape x N x d`-dim Tensor from which to sample (in the `N`
                dimension) according to the maximum posterior value under the objective.
                Note that if a batched model is used in the underlying acquisition
                function, then its batch shape must be broadcastable to `batch_shape`.
            num_samples: The number of samples to draw.

        Returns:
            A `batch_shape x num_samples x d`-dim Tensor of samples from `X`, where
            `X[..., i, :]` is the `i`-th sample.
        """
        # TODO: Can we get the model batch shape property from the model?
        # we move the `N` dimension to the front for evaluating the acquisition function
        # so that X_eval has shape `N x batch_shape x 1 x d`
        X_eval = X.permute(-2, *range(X.ndim - 2), -1).unsqueeze(-2)
        acqval = self.acq_func(X_eval)  # N x batch_shape
        # now move the `N` dimension back (this is the number of categories)
        acqval = acqval.permute(*range(1, X.ndim - 1), 0)  # batch_shape x N
        weights = torch.exp(self.eta * standardize(acqval))  # batch_shape x N
        idcs = batched_multinomial(
            weights=weights, num_samples=num_samples, replacement=self.replacement
        )
        # now do some gathering acrobatics to select the right elements from X
        return torch.gather(X, -2, idcs.unsqueeze(-1).expand(*idcs.shape, X.size(-1)))


class ConstrainedMaxPosteriorSampling(MaxPosteriorSampling):
    r"""Sample from a set of points according to
    their max posterior value,
    which also likely meet a set of constraints
    c1(x) <= 0, c2(x) <= 0, ..., cm(x) <= 0
    c1, c2, ..., cm are black-box constraint functions
    Each constraint function is modeled by a seperate
    surrogate GP constraint model
    We sample points for which the posterior value
    for each constraint model <= 0,
    as described in https://doi.org/10.48550/arxiv.2002.08526

    Example:
        >>> CMPS = ConstrainedMaxPosteriorSampling(model,
                    constraint_model=ModelListGP(cmodel1, cmodel2,
                    ..., cmodelm)  # models w/ feature dim d=3
        >>> X = torch.rand(2, 100, 3)
        >>> sampled_X = CMPS(X, num_samples=5)
    """

    def __init__(
        self,
        model: Model,
        constraint_model: Union[ModelListGP, MultiTaskGP],
        objective: Optional[MCAcquisitionObjective] = None,
        posterior_transform: Optional[PosteriorTransform] = None,
        replacement: bool = True,
        minimize_constraints_only: bool = False,
    ) -> None:
        r"""Constructor for the SamplingStrategy base class.

        Args:
            model: A fitted model.
            objective: The MCAcquisitionObjective under
                which the samples are evaluated.
                Defaults to `IdentityMCObjective()`.
            posterior_transform: An optional PosteriorTransform.
            replacement: If True, sample with replacement.
            constraint_model: either a ModelListGP where each submodel
                is a GP model for one constraint function,
                or a MultiTaskGP model where each task is one
                constraint function
                All constraints are of the form c(x) <= 0.
                In the case when the constraint model predicts
                that all candidates violate constraints,
                we pick the candidates with minimum violation.
            minimize_constraints_only: False by default, if true,
                we will automatically return the candidates
                with minimum posterior constraint values,
                (minimum predicted c(x) summed over all constraints)
                reguardless of predicted objective values.
        """
        super().__init__(
            model=model,
            objective=objective,
            posterior_transform=posterior_transform,
            replacement=replacement,
        )
        self.constraint_model = constraint_model
        self.minimize_constraints_only = minimize_constraints_only

    def forward(
        self, X: Tensor, num_samples: int = 1, observation_noise: bool = False
    ) -> Tensor:
        r"""Sample from the model posterior.

        Args:
            X: A `batch_shape x N x d`-dim Tensor
                from which to sample (in the `N`
                dimension) according to the maximum
                posterior value under the objective.
            num_samples: The number of samples to draw.
            observation_noise: If True, sample with observation noise.

        Returns:
            A `batch_shape x num_samples x d`-dim
            Tensor of samples from `X`, where
            `X[..., i, :]` is the `i`-th sample.
        """
        posterior = self.model.posterior(X, observation_noise=observation_noise)
        samples = posterior.rsample(sample_shape=torch.Size([num_samples]))

        c_posterior = self.constraint_model.posterior(
            X, observation_noise=observation_noise
        )
        constraint_samples = c_posterior.rsample(sample_shape=torch.Size([num_samples]))
        valid_samples = constraint_samples <= 0
        if valid_samples.shape[-1] > 1:  # if more than one constraint
            valid_samples = torch.all(valid_samples, dim=-1).unsqueeze(-1)
        if (valid_samples.sum() == 0) or self.minimize_constraints_only:
            # if none of the samples meet the constraints
            # we pick the one that minimizes total violation
            constraint_samples = constraint_samples.sum(dim=-1)
            idcs = torch.argmin(constraint_samples, dim=-1)
            if idcs.ndim > 1:
                idcs = idcs.permute(*range(1, idcs.ndim), 0)
            idcs = idcs.unsqueeze(-1).expand(*idcs.shape, X.size(-1))
            Xe = X.expand(*constraint_samples.shape[1:], X.size(-1))
            return torch.gather(Xe, -2, idcs)
        # replace all violators with -infinty so it will never choose them
        replacement_infs = -torch.inf * torch.ones(samples.shape).to(X.device).to(
            X.dtype
        )
        samples = torch.where(valid_samples, samples, replacement_infs)

        return self.maximize_samples(X, samples, num_samples)
