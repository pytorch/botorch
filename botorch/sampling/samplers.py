#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Sampler modules to be used with MC-evaluated acquisition functions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
from botorch.exceptions import UnsupportedError
from botorch.posteriors import Posterior
from botorch.utils.sampling import draw_sobol_normal_samples, manual_seed
from torch import Tensor
from torch.nn import Module
from torch.quasirandom import SobolEngine


class MCSampler(Module, ABC):
    r"""Abstract base class for Samplers.

    Subclasses must implement the `_construct_base_samples` method.

    Attributes:
        resample: If `True`, re-draw samples in each `forward` evaluation -
            this results in stochastic acquisition functions (and thus should
            not be used with deterministic optimization algorithms).
        collapse_batch_dims: If True, collapse the t-batch dimensions of the
            produced samples to size 1. This is useful for preventing sampling
            variance across t-batches.

    Example:
        This method is usually not called directly, but via the sampler's
        `__call__` method:
        >>> posterior = model.posterior(test_X)
        >>> samples = sampler(posterior)

    :meta private:
    """

    def __init__(self, batch_range: Tuple[int, int] = (0, -2)) -> None:
        r"""Abstract base class for Samplers.

        Args:
            batch_range: The range of t-batch dimensions in the `base_sample_shape`
                used by `collapse_batch_dims`. The t-batch dims are
                batch_range[0]:batch_range[1]. By default, this is (0, -2),
                for the case where the non-batch dimensions are -2 (q) and
                -1 (d) and all dims in the front are t-batch dims.
        """
        super().__init__()
        self.batch_range = batch_range
        self.register_buffer("base_samples", None)

    @property
    def batch_range(self) -> Tuple[int, int]:
        r"""The t-batch range."""
        return tuple(self._batch_range.tolist())

    @batch_range.setter
    def batch_range(self, batch_range: Tuple[int, int]):
        r"""Set the t-batch range and clear base samples.

        Args:
            batch_range: The range of t-batch dimensions in the `base_sample_shape`
                used by `collapse_batch_dims`. The t-batch dims are
                batch_range[0]:batch_range[1]. By default, this is (0, -2),
                for the case where the non-batch dimensions are -2 (q) and
                -1 (d) and all dims in the front are t-batch dims.
        """
        # set t-batch range if different; trigger resample & set base_samples to None
        if not hasattr(self, "_batch_range") or self.batch_range != batch_range:
            self.register_buffer(
                "_batch_range", torch.tensor(batch_range, dtype=torch.long)
            )
            self.register_buffer("base_samples", None)

    def forward(self, posterior: Posterior) -> Tensor:
        r"""Draws MC samples from the posterior.

        Args:
            posterior: The Posterior to sample from.

        Returns:
            The samples drawn from the posterior.
        """
        base_sample_shape = self._get_base_sample_shape(posterior=posterior)
        self._construct_base_samples(posterior=posterior, shape=base_sample_shape)
        samples = posterior.rsample(
            sample_shape=self.sample_shape, base_samples=self.base_samples
        )
        return samples

    def _get_base_sample_shape(self, posterior: Posterior) -> torch.Size:
        r"""Get the shape of the base samples.

        Args:
            posterior: The Posterior to sample from.

        Returns:
            The shape of the base samples expected by the posterior. If
            `collapse_batch_dims=True`, the t-batch dimensions of the base
            samples are collapsed to size 1. This is useful to prevent sampling
            variance across t-batches.
        """
        base_sample_shape = posterior.base_sample_shape
        if self.collapse_batch_dims:
            batch_start, batch_end = self.batch_range
            base_sample_shape = (
                base_sample_shape[:batch_start]
                + torch.Size([1 for _ in base_sample_shape[batch_start:batch_end]])
                + base_sample_shape[batch_end:]
            )
        return self.sample_shape + base_sample_shape

    @property
    def sample_shape(self) -> torch.Size:
        r"""The shape of a single sample."""
        return self._sample_shape

    @abstractmethod
    def _construct_base_samples(self, posterior: Posterior, shape: torch.Size) -> None:
        r"""Generate base samples (if necessary).

        This function will generate a new set of base samples and register the
        `base_samples` buffer if one of the following is true:

        - `resample=True`
        - the MCSampler has no `base_samples` attribute.
        - `shape` is different than `self.base_samples.shape` (if
            `collapse_batch_dims=True`, then batch dimensions of will be
            automatically broadcasted as necessary). This shape is expected to
            be `sample_shape + base_sample_shape`, where `base_sample_shape` has
            been adjusted to account for `collapse_batch_dims` (i.e., the output
            of the function `_get_base_sample_shape`).

        Args:
            posterior: The Posterior for which to generate base samples.
            shape: The shape of the base samples to construct.
        """
        pass  # pragma: no cover


class IIDNormalSampler(MCSampler):
    r"""Sampler for MC base samples using iid N(0,1) samples.

    Example:
        >>> sampler = IIDNormalSampler(1000, seed=1234)
        >>> posterior = model.posterior(test_X)
        >>> samples = sampler(posterior)
    """

    def __init__(
        self,
        num_samples: int,
        resample: bool = False,
        seed: Optional[int] = None,
        collapse_batch_dims: bool = True,
        batch_range: Tuple[int, int] = (0, -2),
    ) -> None:
        r"""Sampler for MC base samples using iid `N(0,1)` samples.

        Args:
            num_samples: The number of samples to use.
            resample: If `True`, re-draw samples in each `forward` evaluation -
                this results in stochastic acquisition functions (and thus should
                not be used with deterministic optimization algorithms).
            seed: The seed for the RNG. If omitted, use a random seed.
            collapse_batch_dims: If True, collapse the t-batch dimensions to
                size 1. This is useful for preventing sampling variance across
                t-batches.
            batch_range: The range of t-batch dimensions in the `base_sample_shape`
                used by `collapse_batch_dims`. The t-batch dims are
                batch_range[0]:batch_range[1]. By default, this is (0, -2),
                for the case where the non-batch dimensions are -2 (q) and
                -1 (d) and all dims in the front are t-batch dims.
        """
        super().__init__(batch_range=batch_range)
        self._sample_shape = torch.Size([num_samples])
        self.collapse_batch_dims = collapse_batch_dims
        self.resample = resample
        self.seed = seed if seed is not None else torch.randint(0, 1000000, (1,)).item()

    def _construct_base_samples(self, posterior: Posterior, shape: torch.Size) -> None:
        r"""Generate iid `N(0,1)` base samples (if necessary).

        This function will generate a new set of base samples and set the
        `base_samples` buffer if one of the following is true:

        - `resample=True`
        - the MCSampler has no `base_samples` attribute.
        - `shape` is different than `self.base_samples.shape` (if
            `collapse_batch_dims=True`, then batch dimensions of will be
            automatically broadcasted as necessary). This shape is expected to
            be `sample_shape + base_sample_shape`, where `base_sample_shape` has been
            adjusted to account for `collapse_batch_dims` (i.e., the output
            of the function `_get_base_sample_shape`).

        Args:
            posterior: The Posterior for which to generate base samples.
            shape: The shape of the base samples to construct.
        """
        if (
            self.resample
            or _check_shape_changed(self.base_samples, self.batch_range, shape)
            or (not self.collapse_batch_dims and shape != self.base_samples.shape)
        ):
            with manual_seed(seed=self.seed):
                base_samples = torch.randn(
                    shape, device=posterior.device, dtype=posterior.dtype
                )
            self.seed += 1
            self.register_buffer("base_samples", base_samples)
        elif self.collapse_batch_dims and shape != self.base_samples.shape:
            self.base_samples = self.base_samples.view(shape)
        if self.base_samples.device != posterior.device:
            self.to(device=posterior.device)  # pragma: nocover
        if self.base_samples.dtype != posterior.dtype:
            self.to(dtype=posterior.dtype)


class SobolQMCNormalSampler(MCSampler):
    r"""Sampler for quasi-MC base samples using Sobol sequences.

    Example:
        >>> sampler = SobolQMCNormalSampler(1024, seed=1234)
        >>> posterior = model.posterior(test_X)
        >>> samples = sampler(posterior)
    """

    def __init__(
        self,
        num_samples: int,
        resample: bool = False,
        seed: Optional[int] = None,
        collapse_batch_dims: bool = True,
        batch_range: Tuple[int, int] = (0, -2),
    ) -> None:
        r"""Sampler for quasi-MC base samples using Sobol sequences.

        Args:
            num_samples: The number of samples to use. As a best practice,
                use powers of 2.
            resample: If `True`, re-draw samples in each `forward` evaluation -
                this results in stochastic acquisition functions (and thus should
                not be used with deterministic optimization algorithms).
            seed: The seed for the RNG. If omitted, use a random seed.
            collapse_batch_dims: If True, collapse the t-batch dimensions to
                size 1. This is useful for preventing sampling variance across
                t-batches.
            batch_range: The range of t-batch dimensions in the `base_sample_shape`
                used by `collapse_batch_dims`. The t-batch dims are
                batch_range[0]:batch_range[1]. By default, this is (0, -2),
                for the case where the non-batch dimensions are -2 (q) and
                -1 (d) and all dims in the front are t-batch dims.
        """
        super().__init__(batch_range=batch_range)
        self._sample_shape = torch.Size([num_samples])
        self.collapse_batch_dims = collapse_batch_dims
        self.resample = resample
        self.seed = seed if seed is not None else torch.randint(0, 1000000, (1,)).item()

    def _construct_base_samples(self, posterior: Posterior, shape: torch.Size) -> None:
        r"""Generate quasi-random Normal base samples (if necessary).

        This function will generate a new set of base samples and set the
        `base_samples` buffer if one of the following is true:

        - `resample=True`
        - the MCSampler has no `base_samples` attribute.
        - `shape` is different than `self.base_samples.shape` (if
            `collapse_batch_dims=True`, then batch dimensions of will be
            automatically broadcasted as necessary). This shape is expected to
            be `sample_shape + base_sample_shape`, where `base_sample_shape` has been
            adjusted to account for `collapse_batch_dims` (i.e., the output
            of the function `_get_base_sample_shape`).

        Args:
            posterior: The Posterior for which to generate base samples.
            shape: The shape of the base samples to construct.
        """
        if (
            self.resample
            or _check_shape_changed(self.base_samples, self.batch_range, shape)
            or (not self.collapse_batch_dims and shape != self.base_samples.shape)
        ):
            batch_start, batch_end = self.batch_range
            sample_shape, base_sample_shape = split_shapes(shape)
            output_dim = (
                base_sample_shape[:batch_start] + base_sample_shape[batch_end:]
            ).numel()
            if output_dim > SobolEngine.MAXDIM:
                raise UnsupportedError(
                    "SobolQMCSampler only supports dimensions "
                    f"`q * o <= {SobolEngine.MAXDIM}`. Requested: {output_dim}"
                )
            base_samples = draw_sobol_normal_samples(
                d=output_dim,
                n=(sample_shape + base_sample_shape[batch_start:batch_end]).numel(),
                device=posterior.device,
                dtype=posterior.dtype,
                seed=self.seed,
            )
            self.seed += 1
            base_samples = base_samples.view(shape)
            self.register_buffer("base_samples", base_samples)
        elif self.collapse_batch_dims and shape != posterior.base_sample_shape:
            self.base_samples = self.base_samples.view(shape)
        if self.base_samples.device != posterior.device:
            self.to(device=posterior.device)  # pragma: nocover
        if self.base_samples.dtype != posterior.dtype:
            self.to(dtype=posterior.dtype)


def _check_shape_changed(
    base_samples: Optional[Tensor], batch_range: Tuple[int, int], shape: torch.Size
) -> bool:
    r"""Check if the base samples shape matches a given shape in non batch dims.

    Args:
        base_samples: The Posterior for which to generate base samples.
        batch_range: The range t-batch dimensions to ignore for shape check.
        shape: The base sample shape to compare.

    Returns:
        A bool indicating whether the shape changed.
    """
    if base_samples is None:
        return True
    batch_start, batch_end = batch_range
    b_sample_shape, b_base_sample_shape = split_shapes(base_samples.shape)
    sample_shape, base_sample_shape = split_shapes(shape)
    return (
        b_sample_shape != sample_shape
        or b_base_sample_shape[batch_end:] != base_sample_shape[batch_end:]
        or b_base_sample_shape[:batch_start] != base_sample_shape[:batch_start]
    )


def split_shapes(
    base_sample_shape: torch.Size,
) -> Tuple[torch.Size, torch.Size]:
    r"""Split a base sample shape into sample and base sample shapes.

    Args:
        base_sample_shape: The base sample shape.

    Returns:
        A tuple containing the sample and base sample shape.
    """
    return base_sample_shape[:1], base_sample_shape[1:]
