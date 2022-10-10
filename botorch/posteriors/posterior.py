#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Abstract base module for all botorch posteriors.
"""

from __future__ import annotations

from abc import ABC, abstractmethod, abstractproperty
from typing import List, Optional

import torch
from torch import Tensor


class Posterior(ABC):
    r"""
    Abstract base class for botorch posteriors.

    :meta private:
    """

    @property
    def base_sample_shape(self) -> torch.Size:
        r"""The shape of a base sample used for constructing posterior samples.

        This function may be overwritten by subclasses in case `base_sample_shape`
        and `event_shape` do not agree (e.g. if the posterior is a Multivariate
        Gaussian that is not full rank).
        """
        return self.event_shape

    @abstractproperty
    def device(self) -> torch.device:
        r"""The torch device of the posterior."""
        pass  # pragma: no cover

    @abstractproperty
    def dtype(self) -> torch.dtype:
        r"""The torch dtype of the posterior."""
        pass  # pragma: no cover

    @abstractproperty
    def event_shape(self) -> torch.Size:
        r"""The event shape (i.e. the shape of a single sample)."""
        pass  # pragma: no cover

    @property
    def mean(self) -> Tensor:
        r"""The mean of the posterior as a `(b) x n x m`-dim Tensor."""
        raise NotImplementedError(
            f"Property `mean` not implemented for {self.__class__.__name__}"
        )

    @property
    def variance(self) -> Tensor:
        r"""The variance of the posterior as a `(b) x n x m`-dim Tensor."""
        raise NotImplementedError(
            f"Property `variance` not implemented for {self.__class__.__name__}"
        )

    @abstractmethod
    def rsample(
        self,
        sample_shape: Optional[torch.Size] = None,
        base_samples: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Sample from the posterior (with gradients).

        Args:
            sample_shape: A `torch.Size` object specifying the sample shape. To
                draw `n` samples, set to `torch.Size([n])`. To draw `b` batches
                of `n` samples each, set to `torch.Size([b, n])`.
            base_samples: An (optional) Tensor of `N(0, I)` base samples of
                appropriate dimension, typically obtained from a `Sampler`.
                This is used for deterministic optimization.

        Returns:
            A `sample_shape x event`-dim Tensor of samples from the posterior.
        """
        pass  # pragma: no cover

    def sample(
        self,
        sample_shape: Optional[torch.Size] = None,
        base_samples: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Sample from the posterior (without gradients).

        This is a simple wrapper calling `rsample` using `with torch.no_grad()`.

        Args:
            sample_shape: A `torch.Size` object specifying the sample shape. To
                draw `n` samples, set to `torch.Size([n])`. To draw `b` batches
                of `n` samples each, set to `torch.Size([b, n])`.
            base_samples: An (optional) Tensor of `N(0, I)` base samples of
                appropriate dimension, typically obtained from a `Sampler` object.
                This is used for deterministic optimization.

        Returns:
            A `sample_shape x event_shape`-dim Tensor of samples from the posterior.
        """
        with torch.no_grad():
            return self.rsample(sample_shape=sample_shape, base_samples=base_samples)


class PosteriorList(Posterior):
    r"""A Posterior represented by a list of independent Posteriors."""

    def __init__(self, *posteriors: Posterior) -> None:
        r"""A Posterior represented by a list of independent Posteriors.

        Args:
            *posteriors: A variable number of single-outcome posteriors.

        Example:
            >>> p_1 = model_1.posterior(test_X)
            >>> p_2 = model_2.posterior(test_X)
            >>> p_12 = PosteriorList(p_1, p_2)

        Note: This is typically produced automatically in `ModelList`; it should
        generally not be necessary for the end user to invoke it manually.
        """
        self.posteriors = list(posteriors)

    @property
    def base_sample_shape(self) -> torch.Size:
        r"""The shape of a base sample used for constructing posterior samples."""
        base_sample_shapes = [
            p.base_sample_shape
            for p in self.posteriors
            if p.base_sample_shape  # ignore empty sample shapes
        ]
        batch_shapes = [bss[:-1] for bss in base_sample_shapes]
        if len(set(batch_shapes)) > 1:
            raise NotImplementedError(
                "`PosteriorList` only supported if the constituent posteriors "
                f"all have the same `batch_shape`. Batch shapes: {batch_shapes}."
            )
        elif len(set(batch_shapes)) == 0:
            # batch shapes are all zero if and only if the models
            # are determinisitic
            return torch.Size([])
        return batch_shapes[0] + torch.Size(
            [sum(bss[-1] for bss in base_sample_shapes)]
        )

    @property
    def device(self) -> torch.device:
        r"""The torch device of the posterior."""
        devices = {p.device for p in self.posteriors}
        if len(devices) > 1:
            raise NotImplementedError(  # pragma: no cover
                "Multi-device posteriors are currently not supported. "
                "The devices of the constituent posteriors are: {devices}."
            )
        return next(iter(devices))

    @property
    def dtype(self) -> torch.dtype:
        r"""The torch dtype of the posterior."""
        dtypes = {p.dtype for p in self.posteriors}
        if len(dtypes) > 1:
            raise NotImplementedError(
                "Multi-dtype posteriors are currently not supported. "
                "The dtypes of the constituent posteriors are: {dtypes}."
            )
        return next(iter(dtypes))

    @property
    def event_shape(self) -> torch.Size:
        r"""The event shape (i.e. the shape of a single sample)."""
        event_shapes = [p.event_shape for p in self.posteriors]
        batch_shapes = [es[:-1] for es in event_shapes]
        if len(set(batch_shapes)) > 1:
            raise NotImplementedError(
                "`PosteriorList` only supported if the constituent posteriors "
                f"all have the same `batch_shape`. Batch shapes: {batch_shapes}."
            )
        # last dimension is the output dimension (concatenation dimension)
        return batch_shapes[0] + torch.Size([sum(es[-1] for es in event_shapes)])

    @property
    def mean(self) -> Tensor:
        r"""The mean of the posterior as a `(b) x n x m`-dim Tensor."""
        return torch.cat([p.mean for p in self.posteriors], dim=-1)

    @property
    def variance(self) -> Tensor:
        r"""The variance of the posterior as a `(b) x n x m`-dim Tensor."""
        return torch.cat([p.variance for p in self.posteriors], dim=-1)

    def _rsample(
        self,
        sample_shape: Optional[torch.Size] = None,
        base_samples: Optional[Tensor] = None,
    ) -> List[Tensor]:
        # handle the case where all posteriors have empty
        # base_sample_shape
        split_bss = False
        base_sample_splits = [None] * len(self.posteriors)
        if base_samples is not None:
            split_sizes = []
            for p in self.posteriors:
                if p.base_sample_shape:
                    split_sizes.append(p.base_sample_shape[-1])
                    split_bss = True
                else:
                    split_sizes.append(0)
            if split_bss:
                base_sample_splits = torch.split(base_samples, split_sizes, dim=-1)
                base_sample_splits = [
                    bss if ss > 0 else None
                    for ss, bss in zip(split_sizes, base_sample_splits)
                ]
        return [
            p.rsample(sample_shape=sample_shape, base_samples=bss)
            for p, bss in zip(self.posteriors, base_sample_splits)
        ]

    def rsample(
        self,
        sample_shape: Optional[torch.Size] = None,
        base_samples: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Sample from the posterior (with gradients).

        Args:
            sample_shape: A `torch.Size` object specifying the sample shape. To
                draw `n` samples, set to `torch.Size([n])`. To draw `b` batches
                of `n` samples each, set to `torch.Size([b, n])`.
            base_samples: An (optional) Tensor of `N(0, I)` base samples of
                appropriate dimension, typically obtained from a `Sampler`.
                This is used for deterministic optimization.

        Returns:
            A `sample_shape x event`-dim Tensor of samples from the posterior.
        """
        samples = self._rsample(sample_shape=sample_shape, base_samples=base_samples)
        return torch.cat(samples, dim=-1)
