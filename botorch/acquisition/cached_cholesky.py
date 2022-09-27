#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Abstract class for acquisition functions leveraging a cached Cholesky
decomposition of the posterior covaiance over f(X_baseline).
"""
from __future__ import annotations

import warnings
from abc import ABC
from typing import Optional

import torch
from botorch.exceptions.errors import UnsupportedError
from botorch.exceptions.warnings import BotorchWarning
from botorch.models import HigherOrderGP
from botorch.models.deterministic import DeterministicModel
from botorch.models.model import Model, ModelList
from botorch.models.multitask import KroneckerMultiTaskGP, MultiTaskGP
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.posteriors.posterior import Posterior
from botorch.sampling.samplers import MCSampler
from botorch.utils.low_rank import extract_batch_covar, sample_cached_cholesky
from gpytorch import settings as gpt_settings
from gpytorch.distributions.multitask_multivariate_normal import (
    MultitaskMultivariateNormal,
)
from linear_operator.utils.errors import NanError, NotPSDError
from torch import Tensor


class CachedCholeskyMCAcquisitionFunction(ABC):
    r"""Abstract class for acquisition functions using a cached Cholesky.

    Specifically, this is for acquisition functions that require sampling from
    the posterior P(f(X_baseline, X) | D). The Cholesky of the posterior
    covariance over f(X_baseline) is cached.

    :meta private:
    """

    def _check_sampler(self) -> None:
        r"""Check compatibility of sampler and model with a cached Cholesky."""
        if not self.sampler.collapse_batch_dims:
            raise UnsupportedError(
                "Expected sampler to use `collapse_batch_dims=True`."
            )
        elif self.sampler.base_samples is not None:
            warnings.warn(
                message=(
                    "sampler.base_samples is not None. The base_samples must be "
                    "initialized to None. Resetting sampler.base_samples to None."
                ),
                category=BotorchWarning,
            )
            self.sampler.base_samples = None
        elif self._uses_matheron and self.sampler.batch_range != (0, -1):
            raise RuntimeError(
                "sampler.batch_range is not (0, -1). This check requires that the "
                "sampler.batch_range is (0, -1) with GPs that use Matheron's rule "
                "for sampling, in order to properly collapse batch dimensions. "
            )

    def _setup(
        self,
        model: Model,
        sampler: Optional[MCSampler] = None,
        cache_root: bool = False,
        check_sampler: bool = False,
    ) -> None:
        r"""Set class attributes and perform compatibility checks.

        Args:
            model: A model.
            sampler: A sampler.
            cache_root: A boolean indicating whether to cache the Cholesky.
                This might be overridden in the model is not compatible.
            check_sampler: A boolean indicating whether to check the sampler.
                The sampler is always checked if cache_root is True.
        """
        models = model.models if isinstance(model, ModelList) else [model]
        self._is_mt = any(
            isinstance(m, (MultiTaskGP, KroneckerMultiTaskGP, HigherOrderGP))
            for m in models
        )
        self._is_deterministic = any(isinstance(m, DeterministicModel) for m in models)
        self._uses_matheron = any(
            isinstance(m, (KroneckerMultiTaskGP, HigherOrderGP)) for m in models
        )
        if check_sampler or cache_root:
            self._check_sampler()
        if self._is_deterministic or self._is_mt:
            cache_root = False
        self._cache_root = cache_root

    def _cache_root_decomposition(
        self,
        posterior: Posterior,
    ) -> None:
        r"""Cache Cholesky of the posterior covariance over f(X_baseline).

        Args:
            posterior: The posterior over f(X_baseline).
        """
        if isinstance(posterior.mvn, MultitaskMultivariateNormal):
            lazy_covar = extract_batch_covar(posterior.mvn)
        else:
            lazy_covar = posterior.mvn.lazy_covariance_matrix
        with gpt_settings.fast_computations.covar_root_decomposition(False):
            lazy_covar_root = lazy_covar.root_decomposition()
            baseline_L = lazy_covar_root.root.to_dense()
        self.register_buffer("_baseline_L", baseline_L)

    def _get_f_X_samples(self, posterior: GPyTorchPosterior, q_in: int) -> Tensor:
        r"""Get posterior samples at the `q_in` new points from the joint posterior.

        Args:
            posterior: The joint posterior is over (X_baseline, X).
            q_in: The number of new points in the posterior. See `_set_sampler` for
                more information.

        Returns:
            A `sample_shape x batch_shape x q x m`-dim tensor of posterior
                samples at the new points.
        """
        # Technically we should make sure that we add a consistent nugget to the
        # cached covariance (and box decompositions) and the new block.
        # But recomputing box decompositions every time the jitter changes would
        # be quite slow.
        if not self._is_mt and self._cache_root and hasattr(self, "_baseline_L"):
            try:
                return sample_cached_cholesky(
                    posterior=posterior,
                    baseline_L=self._baseline_L,
                    q=q_in,
                    base_samples=self.sampler.base_samples,
                    sample_shape=self.sampler.sample_shape,
                )
            except (NanError, NotPSDError):
                warnings.warn(
                    "Low-rank cholesky updates failed due NaNs or due to an "
                    "ill-conditioned covariance matrix. "
                    "Falling back to standard sampling.",
                    BotorchWarning,
                )

        # TODO: improve efficiency for multi-task models
        samples = self.sampler(posterior)
        if isinstance(self.model, HigherOrderGP):
            # Select the correct q-batch dimension for HOGP.
            q_dim = -self.model._num_dimensions
            q_idcs = (
                torch.arange(-q_in, 0, device=samples.device) + samples.shape[q_dim]
            )
            return samples.index_select(q_dim, q_idcs)
        else:
            return samples[..., -q_in:, :]
