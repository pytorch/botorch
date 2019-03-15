#!/usr/bin/env python3

"""
Sampler modules to be used with MC acquisition functions.
"""

from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import Tensor
from torch.nn import Module

from ..exceptions import UnsupportedError
from ..posteriors import Posterior
from ..qmc.sobol import SobolEngine
from ..utils.sampling import draw_sobol_normal_samples, manual_seed


class MCSampler(Module, ABC):
    """Abstract base class for Samplers

    Subclasses must implement the `_construct_base_samples` method.

    Attributes:
        sample_shape: The shape of each sample.
        collapse_batch_dims: If True, collapse the t-batch dimensions of the
            produced samples to size 1. This is useful for preventing sampling
            variance across t-batches.
    """

    def forward(self, posterior: Posterior) -> Tensor:
        """Draws MC samples from the posterior.

        Args:
            posterior: The Posterior to sample from.

        Returns:
            Tensor: The samples drawn from the posterior
        """
        base_sample_shape = self._get_base_sample_shape(posterior=posterior)
        self._construct_base_samples(posterior=posterior, shape=base_sample_shape)
        samples = posterior.rsample(
            sample_shape=self.sample_shape, base_samples=self.base_samples
        )
        return samples

    def _get_base_sample_shape(self, posterior: Posterior) -> torch.Size:
        """Get the shape of the base samples.

        Args:
            posterior: The Posterior to sample from.

        Returns:
            torch.Size: The shape of the base samples expected by the posterior.
                If `collapse_batch_dims=True`, the t-batch dimensions of the
                base samples are collapsed to size 1. This is useful to prevent
                sampling variance across t-batches.
        """
        event_shape = posterior.event_shape
        if self.collapse_batch_dims:
            event_shape = torch.Size([1 for _ in event_shape[:-2]]) + event_shape[-2:]
        return self.sample_shape + event_shape

    @property
    def sample_shape(self) -> torch.Size:
        return self._sample_shape

    @sample_shape.setter
    def sample_shape(self, shape: torch.Size) -> None:
        self._sample_shape = shape

    @abstractmethod
    def _construct_base_samples(self, posterior: Posterior, shape: torch.Size) -> None:
        """Generate base samples if necessary.

        This function will generate a new set of base samples and register the
        `base_samples` buffer if one of the following is true:
          - the MCSampler has no `base_samples` attribute.
          - `shape` is different than `self.base_samples.shape`.
          - device and/or dtype of posterior are different than those of
            `self.base_samples`.
          - the MCSampler does not use a fixed seed.

        Args:
            posterior: The Posterior for which to generate base samples.
            shape: The shape of the base samples to construct.
        """
        pass


class IIDNormalSampler(MCSampler):
    """Sampler for MC base samples using iid N(0,1) samples."""

    def __init__(
        self,
        num_samples: int,
        seed: Optional[int] = None,
        collapse_batch_dims: bool = True,
    ) -> None:
        """Sampler for MC base samples using iid N(0,1) samples.

        Args:
            num_samples: The number of samples to use.
            seed: The seed for the RNG. If omitted, use a random seed.
            collapse_batch_dims: If True, collapse the t-batch dimensions to
                size 1. This is useful for preventing sampling variance across
                t-batches.
        """
        super().__init__()
        self._sample_shape = torch.Size([num_samples])
        self.collapse_batch_dims = collapse_batch_dims
        self.seed = seed

    def _construct_base_samples(self, posterior: Posterior, shape: torch.Size) -> None:
        """Generate iid N(0,1) base samples if necessary.

        This function will generate a new set of base samples and set the
        `base_samples` buffer if one of the following is true:
          - the MCSampler has no `base_samples` attribute.
          - `shape` is different than `self.base_samples.shape`.
          - device and/or dtype of posterior ar different than those of
            `self.base_samples`.
          - the MCSampler does not use a fixed seed.

        Args:
            posterior: The Posterior for which to generate base samples.
            shape: The shape of the base samples to construct.
        """
        if (
            not hasattr(self, "base_samples")
            or self.seed is None
            or self.base_samples.shape != shape
            or self.base_samples.device != posterior.device
            or self.base_samples.dtype != posterior.dtype
        ):
            with manual_seed(seed=self.seed):
                base_samples = torch.randn(
                    shape, device=posterior.device, dtype=posterior.dtype
                )
            self.register_buffer("base_samples", base_samples)


class SobolQMCNormalSampler(MCSampler):
    """Sampler for QMC base samples using Sobol sequences."""

    def __init__(
        self,
        num_samples: int,
        seed: Optional[int] = None,
        collapse_batch_dims: bool = True,
    ) -> None:
        """Sampler for QMC base samples using Sobol sequences.

        Args:
            num_samples: The number of samples to use.
            seed: The seed for the RNG. If omitted, use a random seed.
            collapse_batch_dims: If True, collapse the t-batch dimensions to
                size 1. This is useful for preventing sampling variance across
                t-batches.
        """
        super().__init__()
        self._sample_shape = torch.Size([num_samples])
        self.collapse_batch_dims = collapse_batch_dims
        self.seed = seed

    def _construct_base_samples(self, posterior: Posterior, shape: torch.Size) -> None:
        """Generate quasi-random Normal base samples if necessary.

        This function will generate a new set of base samples and set the
        `base_samples` buffer if one of the following is true:
          - the MCSampler has no `base_samples` attribute.
          - `self.sample_shape` is different than `self.base_samples.shape`.
          - device and/or dtype of posterior ar different than those of
            `self.base_samples`.
          - the MCSampler does not use a fixed seed.

        Args:
            posterior: The Posterior for which to generate base samples.
            shape: The shape of the base samples to construct.
        """
        if (
            not hasattr(self, "base_samples")
            or self.seed is None
            or self.base_samples.shape != shape
            or self.base_samples.device != posterior.device
            or self.base_samples.dtype != posterior.dtype
        ):
            output_dim = shape[-2:].numel()
            if output_dim > SobolEngine.MAXDIM:
                raise UnsupportedError(
                    "SobolQMCSampler only supports dimensions "
                    f"`qt <= {SobolEngine.MAXDIM}`. Requested: {output_dim}"
                )
            base_samples = draw_sobol_normal_samples(
                d=output_dim,
                n=shape[:-2].numel(),
                device=posterior.device,
                dtype=posterior.dtype,
                seed=self.seed,
            )
            base_samples = base_samples.view(shape)
            self.register_buffer("base_samples", base_samples)
