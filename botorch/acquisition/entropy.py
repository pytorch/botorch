#!/usr/bin/env python3

r"""
Analytic Acquisition Functions that evaluate the posterior without performing
Monte-Carlo sampling.
"""

from abc import ABC
from typing import Dict, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.distributions import Normal
from ..models.model import Model
from ..utils.transforms import q_batch_mode_transform
from .analytic import AnalyticAcquisitionFunction
from .sampler import MCSampler, IIDNormalSampler


class EntropyAcquisitionFunction(AnalyticAcquisitionFunction):
    r"""
    """

    def __init__(
        self,
        model: Model,
        bounds: Tensor,
        candidate_set_size: int,
        sampler: Optional[MCSampler] = None,
    ) -> None:
        r"""
        """
        if sampler is None:
            sampler = IIDNormalSampler(num_samples=16)

        super().__init__(model=model)
        self.sampler = sampler
        self.register_buffer(
            "candidate_set",
            torch.rand(candidate_set_size, bounds.size(1), device=bounds.device, dtype=bounds.dtype)
        )
        self.candidate_set.mul_(bounds[1] - bounds[0]).add_(bounds[0])

        self._candidate_set_argmax = None
        self._candidate_set_max_values = None

    def _sample_candidate_set(self):
        with torch.no_grad():
            samples = self.sampler(self.model.posterior(self.candidate_set)).squeeze(-1)
            max_values, indices = samples.max(dim=-1)
            argmax = self.candidate_set[indices]
            self._candidate_set_argmax = argmax
            self._candidate_set_max_values = max_values

    def candidate_set_argmax(self):
        r"""
        """
        if self._candidate_set_argmax is None:
            self._sample_candidate_set()
        return self._candidate_set_argmax

    def candidate_set_max_values(self):
        r"""
        """
        if self._candidate_set_max_values is None:
            self._sample_candidate_set()
        return self._candidate_set_max_values


class MaxValueEntropySearch(EntropyAcquisitionFunction):
    r"""
    """

    def forward(self, X: Tensor) -> Tensor:
        posterior = self.model.posterior(X)
        mean = posterior.mean
        stdv = posterior.variance.sqrt()
        
        normalized_mvs = (self.candidate_set_max_values() - mean) / stdv
        normal = torch.distributions.Normal(torch.zeros_like(normalized_mvs), torch.ones_like(normalized_mvs))
        pdf = normal.log_prob(normalized_mvs).exp()
        cdf = normal.cdf(normalized_mvs)
        res = (normalized_mvs.mul(pdf).div(2 * cdf) - cdf.log()).mean(-1)
        return res
