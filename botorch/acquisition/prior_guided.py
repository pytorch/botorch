#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


"""
Prior-Guided Acquisition Functions

References

.. [Hvarfner2022]
    C. Hvarfner, D. Stoll, A. Souza, M. Lindauer, F. Hutter, L. Nardi. PiBO:
    Augmenting Acquisition Functions with User Beliefs for Bayesian Optimization.
    ICLR 2022.
"""
from __future__ import annotations

from typing import Optional

from botorch.acquisition.acquisition import AcquisitionFunction
from torch import Tensor

from torch.nn import Module


class PriorGuidedAcquisitionFunction(AcquisitionFunction):
    r"""Class for weighting acquisition functions by a prior distribution.

    See [Hvarfner2022]_ for details.
    """

    def __init__(
        self,
        acq_function: AcquisitionFunction,
        prior_module: Module,
        log: bool = False,
        prior_exponent: float = 1.0,
    ) -> None:
        r"""Initialize the prior-guided acquisition function.

        Args:
            acq_function: The base acquisition function.
            prior_module: A Module that computes the probability
                (or log probability) for the provided inputs.
            log: A boolean that should be true if the acquisition function emits a
                log-transformed value and the prior module emits a log probability.
            prior_exponent: The exponent applied to the prior. This can be used
                for example  to decay the effect the prior over time as in
                [Hvarfner2022]_.
        """
        Module.__init__(self)
        self.acq_func = acq_function
        self.prior_module = prior_module
        self._log = log
        self._prior_exponent = prior_exponent

    @property
    def X_pending(self):
        r"""Return the `X_pending` of the base acquisition function."""
        try:
            return self.acq_func.X_pending
        except (ValueError, AttributeError):
            raise ValueError(
                f"Base acquisition function {type(self.acq_func).__name__} "
                "does not have an `X_pending` attribute."
            )

    @X_pending.setter
    def X_pending(self, X_pending: Optional[Tensor]):
        r"""Sets the `X_pending` of the base acquisition function."""
        self.acq_func.X_pending = X_pending

    def forward(self, X: Tensor) -> Tensor:
        r"""Compute the acquisition function weighted by the prior."""
        if self._log:
            return self.acq_func(X) + self.prior_module(X) * self._prior_exponent
        return self.acq_func(X) * self.prior_module(X).pow(self._prior_exponent)
