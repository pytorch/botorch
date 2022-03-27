#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Objective Modules to be used with acquisition functions.
"""

from __future__ import annotations

import inspect
import warnings
from abc import ABC, abstractmethod
from typing import Callable, List, Optional

import torch
from botorch.models.model import Model
from botorch.posteriors.gpytorch import GPyTorchPosterior, scalarize_posterior
from botorch.posteriors.posterior import Posterior
from botorch.sampling import IIDNormalSampler, MCSampler
from botorch.utils import apply_constraints
from torch import Tensor
from torch.nn import Module


class AcquisitionObjective(Module, ABC):
    r"""Abstract base class for objectives.

    DEPRECATED - This will be removed in the next version.
    """
    ...


class PosteriorTransform(Module, ABC):
    r"""Abstract base class for objectives that transform the posterior."""

    scalarize: bool  # True if the transform reduces to single-output

    @abstractmethod
    def evaluate(self, Y: Tensor) -> Tensor:
        r"""Evaluate the transform on a set of outcomes.

        Args:
            Y: A `batch_shape x q x m`-dim tensor of outcomes.

        Returns:
            A `batch_shape x q' [x m']`-dim tensor of transformed outcomes.
        """
        pass  # pragma: no cover

    @abstractmethod
    def forward(self, posterior: Posterior) -> Posterior:
        r"""Compute the transformed posterior.

        Args:
            posterior: The posterior to be transformed.

        Returns:
            The transformed posterior object.
        """
        pass  # pragma: no cover


# import DeterministicModel after PosteriorTransform to avoid circular import
from botorch.models.deterministic import DeterministicModel  # noqa


class ScalarizedPosteriorTransform(PosteriorTransform):
    r"""An affine posterior transform for scalarizing multi-output posteriors.

    For a Gaussian posterior at a single point (`q=1`) with mean `mu` and
    covariance matrix `Sigma`, this yields a single-output posterior with mean
    `weights^T * mu` and variance `weights^T Sigma w`.

    Example:
        Example for a model with two outcomes:

        >>> weights = torch.tensor([0.5, 0.25])
        >>> posterior_transform = ScalarizedPosteriorTransform(weights)
        >>> EI = ExpectedImprovement(
        ... model, best_f=0.1, posterior_transform=posterior_transform
        ... )
    """

    scalarize: bool = True

    def __init__(self, weights: Tensor, offset: float = 0.0) -> None:
        r"""Affine posterior transform.

        Args:
            weights: A one-dimensional tensor with `m` elements representing the
                linear weights on the outputs.
            offset: An offset to be added to posterior mean.
        """
        if weights.dim() != 1:
            raise ValueError("weights must be a one-dimensional tensor.")
        super().__init__()
        self.register_buffer("weights", weights)
        self.offset = offset

    def evaluate(self, Y: Tensor) -> Tensor:
        r"""Evaluate the transform on a set of outcomes.

        Args:
            Y: A `batch_shape x q x m`-dim tensor of outcomes.

        Returns:
            A `batch_shape x q`-dim tensor of transformed outcomes.
        """
        return self.offset + Y @ self.weights

    def forward(self, posterior: GPyTorchPosterior) -> GPyTorchPosterior:
        r"""Compute the posterior of the affine transformation.

        Args:
            posterior: A posterior with the same number of outputs as the
                elements in `self.weights`.

        Returns:
            A single-output posterior.
        """
        return scalarize_posterior(
            posterior=posterior, weights=self.weights, offset=self.offset
        )


class ScalarizedObjective(ScalarizedPosteriorTransform, AcquisitionObjective):
    """DEPRECATED - Use ScalarizedPosteriorTransform instead."""

    def __init__(self, weights: Tensor, offset: float = 0.0) -> None:
        warnings.warn(
            "ScalarizedObjective is deprecated and will be removed in the next "
            "version. Use ScalarizedPosteriorTransform instead."
        )
        super().__init__(weights=weights, offset=offset)


class MCAcquisitionObjective(Module, ABC):
    r"""Abstract base class for MC-based objectives.

    Args:
        _verify_output_shape: If True and `X` is given, check that the q-batch
            shape of the objectives agrees with that of X.
        _is_mo: A boolean denoting whether the objectives are multi-output.
    """

    _verify_output_shape: bool = True
    _is_mo: bool = False

    @abstractmethod
    def forward(self, samples: Tensor, X: Optional[Tensor] = None) -> Tensor:
        r"""Evaluate the objective on the samples.

        Args:
            samples: A `sample_shape x batch_shape x q x m`-dim Tensors of
                samples from a model posterior.
            X: A `batch_shape x q x d`-dim tensor of inputs. Relevant only if
                the objective depends on the inputs explicitly.

        Returns:
            Tensor: A `sample_shape x batch_shape x q`-dim Tensor of objective
            values (assuming maximization).

        This method is usually not called directly, but via the objectives.

        Example:
            >>> # `__call__` method:
            >>> samples = sampler(posterior)
            >>> outcome = mc_obj(samples)
        """
        pass  # pragma: no cover

    def __call__(
        self, samples: Tensor, X: Optional[Tensor] = None, *args, **kwargs
    ) -> Tensor:
        output = super().__call__(samples=samples, X=X, *args, **kwargs)
        # q-batch dimension is at -1 for single-output objectives and at
        # -2 for multi-output objectives.
        q_batch_idx = -2 if self._is_mo else -1
        if (
            X is not None
            and self._verify_output_shape
            and output.shape[q_batch_idx] != X.shape[-2]
        ):
            raise RuntimeError(
                "The q-batch shape of the objective values does not agree with "
                f"the q-batch shape of X. Got {output.shape[q_batch_idx]} and "
                f"{X.shape[-2]}. This may happen if you used a one-to-many input "
                "transform but forgot to use a corresponding objective."
            )
        return output


class IdentityMCObjective(MCAcquisitionObjective):
    r"""Trivial objective extracting the last dimension.

    Example:
        >>> identity_objective = IdentityMCObjective()
        >>> samples = sampler(posterior)
        >>> objective = identity_objective(samples)
    """

    def forward(self, samples: Tensor, X: Optional[Tensor] = None) -> Tensor:
        return samples.squeeze(-1)


class LinearMCObjective(MCAcquisitionObjective):
    r"""Linear objective constructed from a weight tensor.

    For input `samples` and `mc_obj = LinearMCObjective(weights)`, this produces
    `mc_obj(samples) = sum_{i} weights[i] * samples[..., i]`

    Example:
        Example for a model with two outcomes:

        >>> weights = torch.tensor([0.75, 0.25])
        >>> linear_objective = LinearMCObjective(weights)
        >>> samples = sampler(posterior)
        >>> objective = linear_objective(samples)
    """

    def __init__(self, weights: Tensor) -> None:
        r"""Linear Objective.

        Args:
            weights: A one-dimensional tensor with `m` elements representing the
                linear weights on the outputs.
        """
        super().__init__()
        if weights.dim() != 1:
            raise ValueError("weights must be a one-dimensional tensor.")
        self.register_buffer("weights", weights)

    def forward(self, samples: Tensor, X: Optional[Tensor] = None) -> Tensor:
        r"""Evaluate the linear objective on the samples.

        Args:
            samples: A `sample_shape x batch_shape x q x m`-dim tensors of
                samples from a model posterior.
            X: A `batch_shape x q x d`-dim tensor of inputs. Relevant only if
                the objective depends on the inputs explicitly.

        Returns:
            A `sample_shape x batch_shape x q`-dim tensor of objective values.
        """
        if samples.shape[-1] != self.weights.shape[-1]:
            raise RuntimeError("Output shape of samples not equal to that of weights")
        return torch.einsum("...m, m", [samples, self.weights])


class GenericMCObjective(MCAcquisitionObjective):
    r"""Objective generated from a generic callable.

    Allows to construct arbitrary MC-objective functions from a generic
    callable. In order to be able to use gradient-based acquisition function
    optimization it should be possible to backpropagate through the callable.

    Example:
        >>> generic_objective = GenericMCObjective(
                lambda Y, X: torch.sqrt(Y).sum(dim=-1),
            )
        >>> samples = sampler(posterior)
        >>> objective = generic_objective(samples)
    """

    def __init__(self, objective: Callable[[Tensor, Optional[Tensor]], Tensor]) -> None:
        r"""Objective generated from a generic callable.

        Args:
            objective: A callable `f(samples, X)` mapping a
                `sample_shape x batch-shape x q x m`-dim Tensor `samples` and
                an optional `batch-shape x q x d`-dim Tensor `X` to a
                `sample_shape x batch-shape x q`-dim Tensor of objective values.
        """
        super().__init__()
        if len(inspect.signature(objective).parameters) == 1:
            warnings.warn(
                "The `objective` callable of `GenericMCObjective` is expected to "
                "take two arguments. Passing a callable that expects a single "
                "argument will result in an error in future versions.",
                DeprecationWarning,
            )

            def obj(samples: Tensor, X: Optional[Tensor] = None) -> Tensor:
                return objective(samples)

            self.objective = obj
        else:
            self.objective = objective

    def forward(self, samples: Tensor, X: Optional[Tensor] = None) -> Tensor:
        r"""Evaluate the feasibility-weigthed objective on the samples.

        Args:
            samples: A `sample_shape x batch_shape x q x m`-dim Tensors of
                samples from a model posterior.
            X: A `batch_shape x q x d`-dim tensor of inputs. Relevant only if
                the objective depends on the inputs explicitly.

        Returns:
            A `sample_shape x batch_shape x q`-dim Tensor of objective values
            weighted by feasibility (assuming maximization).
        """
        return self.objective(samples, X=X)


class ConstrainedMCObjective(GenericMCObjective):
    r"""Feasibility-weighted objective.

    An Objective allowing to maximize some scalable objective on the model
    outputs subject to a number of constraints. Constraint feasibilty is
    approximated by a sigmoid function.

        mc_acq(X) = (
        (objective(X) + infeasible_cost) * \prod_i (1  - sigmoid(constraint_i(X)))
        ) - infeasible_cost

    See `botorch.utils.objective.apply_constraints` for details on the constraint
    handling.

    Example:
        >>> bound = 0.0
        >>> objective = lambda Y: Y[..., 0]
        >>> # apply non-negativity constraint on f(x)[1]
        >>> constraint = lambda Y: bound - Y[..., 1]
        >>> constrained_objective = ConstrainedMCObjective(objective, [constraint])
        >>> samples = sampler(posterior)
        >>> objective = constrained_objective(samples)
    """

    def __init__(
        self,
        objective: Callable[[Tensor, Optional[Tensor]], Tensor],
        constraints: List[Callable[[Tensor], Tensor]],
        infeasible_cost: float = 0.0,
        eta: float = 1e-3,
    ) -> None:
        r"""Feasibility-weighted objective.

        Args:
            objective: A callable `f(samples, X)` mapping a
                `sample_shape x batch-shape x q x m`-dim Tensor `samples` and
                an optional `batch-shape x q x d`-dim Tensor `X` to a
                `sample_shape x batch-shape x q`-dim Tensor of objective values.
            constraints: A list of callables, each mapping a Tensor of dimension
                `sample_shape x batch-shape x q x m` to a Tensor of dimension
                `sample_shape x batch-shape x q`, where negative values imply
                feasibility.
            infeasible_cost: The cost of a design if all associated samples are
                infeasible.
            eta: The temperature parameter of the sigmoid function approximating
                the constraint.
        """
        super().__init__(objective=objective)
        self.constraints = constraints
        self.eta = eta
        self.register_buffer("infeasible_cost", torch.as_tensor(infeasible_cost))

    def forward(self, samples: Tensor, X: Optional[Tensor] = None) -> Tensor:
        r"""Evaluate the feasibility-weighted objective on the samples.

        Args:
            samples: A `sample_shape x batch_shape x q x m`-dim Tensors of
                samples from a model posterior.
            X: A `batch_shape x q x d`-dim tensor of inputs. Relevant only if
                the objective depends on the inputs explicitly.

        Returns:
            A `sample_shape x batch_shape x q`-dim Tensor of objective values
            weighted by feasibility (assuming maximization).
        """
        obj = super().forward(samples=samples)
        return apply_constraints(
            obj=obj,
            constraints=self.constraints,
            samples=samples,
            infeasible_cost=self.infeasible_cost,
            eta=self.eta,
        )


class LearnedObjective(MCAcquisitionObjective):
    r"""Learned preference objective constructed from a preference model.

    For input `samples`, it samples each individual sample again from the latent
    preference posterior distribution using `pref_model` and return the posterior mean.

    Example:
        >>> train_X = torch.rand(2, 2)
        >>> train_comps = torch.LongTensor([[0, 1]])
        >>> pref_model = PairwiseGP(train_X, train_comps)
        >>> learned_pref_obj = LearnedObjective(pref_model)
        >>> samples = sampler(posterior)
        >>> objective = learned_pref_obj(samples)
    """

    def __init__(
        self,
        pref_model: Model,
        sampler: Optional[MCSampler] = None,
    ):
        r"""Learned preference objective constructed from a preference model.

        Args:
            pref_model: A BoTorch model, which models the latent preference/utility
                function. Given an input tensor of size
                `sample_size x batch_shape x N x d`, its `posterior` method should
                return a `Posterior` object with single outcome representing the
                utility values of the input.
            sampler: Sampler for the preference model to account for uncertainty in
                preferece when calculating the objective; it's not the one used
                in MC acquisition functions. If None,
                it uses `IIDNormalSampler(num_samples=1)`.
        """
        super().__init__()
        self.pref_model = pref_model
        if isinstance(pref_model, DeterministicModel):
            assert sampler is None
            self.sampler = None
        else:
            if sampler is None:
                self.sampler = IIDNormalSampler(num_samples=1)
            else:
                self.sampler = sampler

    def forward(self, samples: Tensor, X: Optional[Tensor] = None) -> Tensor:
        r"""Sample each element of samples.

        Args:
            samples: A `sample_size x batch_shape x N x d`-dim Tensors of
                samples from a model posterior.

        Returns:
            A `(sample_size * num_samples) x batch_shape x N`-dim Tensor of
            objective values sampled from utility posterior using `pref_model`.
        """
        post = self.pref_model.posterior(samples)
        if isinstance(self.pref_model, DeterministicModel):
            # return preference posterior mean
            return post.mean.squeeze(-1)
        else:
            # return preference posterior sample mean
            samples = self.sampler(post).squeeze(-1)
            return samples.reshape(-1, *samples.shape[2:])  # batch_shape x N
