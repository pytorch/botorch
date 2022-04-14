#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Input Augmentation Transformations.

These classes implement a variety of transformations for
input parameters that are applied only to the test inputs
at the `posterior` call.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Callable, Union

import torch
from torch import Tensor
from torch.nn import Module


class InputAugmentationTransform(Module, ABC):
    r"""Abstract base class for input augmentation transforms."""

    @abstractmethod
    def forward(self, X: Tensor) -> Tensor:
        r"""Transform the inputs to a model.

        Args:
            X: A `batch_shape x q x d`-dim tensor of inputs.

        Returns:
            A `batch_shape x q' x d'`-dim tensor of transformed inputs, where `q'`
                is generally an integer multiple of `q` and `d' > d`, both determined
                by the transform arguments.
        """
        pass  # pragma: no cover

    def equals(self, other: InputAugmentationTransform) -> bool:
        r"""Check if another input augmentation transform is equivalent.

        Note: The reason that a custom equals method is defined rather than
        defining an __eq__ method is because defining an __eq__ method sets
        the __hash__ method to None. Hashing modules is currently used in
        pytorch. See https://github.com/pytorch/pytorch/issues/7733.

        Args:
            other: Another input augmentation transform.

        Returns:
            A boolean indicating if the other transform is equivalent.
        """
        other_state_dict = other.state_dict()
        return type(self) == type(other) and all(
            torch.allclose(v, other_state_dict[k].to(v))
            for k, v in self.state_dict().items()
        )


class AppendFeatures(InputAugmentationTransform):
    r"""A transform that appends the input with a given set of features.

    As an example, this can be used with `RiskMeasureMCObjective` to optimize risk
    measures as described in [Cakmak2020risk]_. A tutorial notebook implementing the
    rhoKG acqusition function introduced in [Cakmak2020risk]_ can be found at
    https://botorch.org/tutorials/risk_averse_bo_with_environmental_variables.

    The steps for using this to obtain samples of a risk measure are as follows:

    -   Train a model on `(x, w)` inputs and the corresponding observations;

    -   Pass in an instance of `AppendFeatures` with the `feature_set` denoting the
        samples of `W` as the `input_transform` to the trained model;

    -   Call `posterior(...).rsample(...)` on the model with `x` inputs only to
        get the joint posterior samples over `(x, w)`s, where the `w`s come
        from the `feature_set`;

    -   Pass these posterior samples through the `RiskMeasureMCObjective` of choice to
        get the samples of the risk measure.

    Note: The samples of the risk measure obtained this way are in general biased
    since the `feature_set` does not fully represent the distribution of the
    environmental variable.

    Example:
        >>> # We consider 1D `x` and 1D `w`, with `W` having a
        >>> # uniform distribution over [0, 1]
        >>> model = SingleTaskGP(
        ...     train_X=torch.rand(10, 2),
        ...     train_Y=torch.randn(10, 1),
        ...     input_augmentation_transform=AppendFeatures(feature_set=torch.rand(10, 1))
        ... )
        >>> mll = ExactMarginalLogLikelihood(model.likelihood, model)
        >>> fit_gpytorch_model(mll)
        >>> test_x = torch.rand(3, 1)
        >>> # `posterior_samples` is a `10 x 30 x 1`-dim tensor
        >>> posterior_samples = model.posterior(test_x).rsamples(torch.size([10]))
        >>> risk_measure = VaR(alpha=0.8, n_w=10)
        >>> # `risk_measure_samples` is a `10 x 3`-dim tensor of samples of the
        >>> # risk measure VaR
        >>> risk_measure_samples = risk_measure(posterior_samples)
    """

    def __init__(
        self,
        feature_set: Tensor,
    ) -> None:
        r"""Append `feature_set` to each input.

        Args:
            feature_set: An `n_f x d_f`-dim tensor denoting the features to be
                appended to the inputs.
        """
        super().__init__()
        if feature_set.dim() != 2:
            raise ValueError("`feature_set` must be an `n_f x d_f`-dim tensor!")
        self.register_buffer("feature_set", feature_set)

    def forward(self, X: Tensor) -> Tensor:
        r"""Transform the inputs by appending `feature_set` to each input.

        For each `1 x d`-dim element in the input tensor, this will produce
        an `n_f x (d + d_f)`-dim tensor with `feature_set` appended as the last `d_f`
        dimensions. For a generic `batch_shape x q x d`-dim `X`, this translates to a
        `batch_shape x (q * n_f) x (d + d_f)`-dim output, where the values corresponding
        to `X[..., i, :]` are found in `output[..., i * n_f: (i + 1) * n_f, :]`.

        Note: Adding the `feature_set` on the `q-batch` dimension is necessary to avoid
        introducing additional bias by evaluating the inputs on independent GP
        sample paths.

        Args:
            X: A `batch_shape x q x d`-dim tensor of inputs.

        Returns:
            A `batch_shape x (q * n_f) x (d + d_f)`-dim tensor of appended inputs.
        """
        expanded_X = X.unsqueeze(dim=-2).expand(
            *X.shape[:-1], self.feature_set.shape[0], -1
        )
        expanded_features = self.feature_set.expand(*expanded_X.shape[:-1], -1)
        appended_X = torch.cat([expanded_X, expanded_features], dim=-1)
        return appended_X.view(*X.shape[:-2], -1, appended_X.shape[-1])


class InputPerturbation(InputAugmentationTransform):
    r"""A transform that adds the set of perturbations to the given input.

    Similar to `AppendFeatures`, this can be used with `RiskMeasureMCObjective`
    to optimize risk measures. See `AppendFeatures` for additional discussion
    on optimizing risk measures.

    A tutorial notebook using this with `qNoisyExpectedImprovement` can be found at
    https://botorch.org/tutorials/risk_averse_bo_with_input_perturbations.
    """

    def __init__(
        self,
        perturbation_set: Union[Tensor, Callable[[Tensor], Tensor]],
        bounds: Optional[Tensor] = None,
        multiplicative: bool = False,
    ) -> None:
        r"""Add `perturbation_set` to each input.

        Args:
            perturbation_set: An `n_p x d`-dim tensor denoting the perturbations
                to be added to the inputs. Alternatively, this can be a callable that
                returns `batch x n_p x d`-dim tensor of perturbations for input of
                shape `batch x d`. This is useful for heteroscedastic perturbations.
            bounds: A `2 x d`-dim tensor of lower and upper bounds for each
                column of the input. If given, the perturbed inputs will be
                clamped to these bounds.
            multiplicative: A boolean indicating whether the input perturbations
                are additive or multiplicative. If True, inputs will be multiplied
                with the perturbations.
        """
        super().__init__()
        if isinstance(perturbation_set, Tensor):
            if perturbation_set.dim() != 2:
                raise ValueError("`perturbation_set` must be an `n_p x d`-dim tensor!")
            self.register_buffer("perturbation_set", perturbation_set)
        else:
            self.perturbation_set = perturbation_set
        if bounds is not None:
            if (
                isinstance(perturbation_set, Tensor)
                and bounds.shape[-1] != perturbation_set.shape[-1]
            ):
                raise ValueError(
                    "`bounds` must have the same number of columns (last dimension) as "
                    f"the `perturbation_set`! Got {bounds.shape[-1]} and "
                    f"{perturbation_set.shape[-1]}."
                )
            self.register_buffer("bounds", bounds)
        else:
            self.bounds = None
        self.multiplicative = multiplicative

    def forward(self, X: Tensor) -> Tensor:
        r"""Transform the inputs by adding `perturbation_set` to each input.

        For each `1 x d`-dim element in the input tensor, this will produce
        an `n_p x d`-dim tensor with the `perturbation_set` added to the input.
        For a generic `batch_shape x q x d`-dim `X`, this translates to a
        `batch_shape x (q * n_p) x d`-dim output, where the values corresponding
        to `X[..., i, :]` are found in `output[..., i * n_w: (i + 1) * n_w, :]`.

        Note: Adding the `perturbation_set` on the `q-batch` dimension is necessary
        to avoid introducing additional bias by evaluating the inputs on independent
        GP sample paths.

        Args:
            X: A `batch_shape x q x d`-dim tensor of inputs.

        Returns:
            A `batch_shape x (q * n_p) x d`-dim tensor of perturbed inputs.
        """
        if isinstance(self.perturbation_set, Tensor):
            perturbations = self.perturbation_set
        else:
            perturbations = self.perturbation_set(X)
        expanded_X = X.unsqueeze(dim=-2).expand(
            *X.shape[:-1], perturbations.shape[-2], -1
        )
        expanded_perturbations = perturbations.expand(*expanded_X.shape[:-1], -1)
        if self.multiplicative:
            perturbed_inputs = expanded_X * expanded_perturbations
        else:
            perturbed_inputs = expanded_X + expanded_perturbations
        perturbed_inputs = perturbed_inputs.reshape(*X.shape[:-2], -1, X.shape[-1])
        if self.bounds is not None:
            perturbed_inputs = torch.maximum(
                torch.minimum(perturbed_inputs, self.bounds[1]), self.bounds[0]
            )
        return perturbed_inputs
