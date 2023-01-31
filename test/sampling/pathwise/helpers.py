#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Tuple

from botorch.models.transforms.outcome import Standardize
from torch import Size, Tensor


def get_sample_moments(samples: Tensor, sample_shape: Size) -> Tuple[Tensor, Tensor]:
    sample_dim = len(sample_shape)
    samples = samples.view(-1, *samples.shape[sample_dim:])
    loc = samples.mean(dim=0)
    residuals = (samples - loc).permute(*range(1, samples.ndim), 0)
    return loc, (residuals @ residuals.transpose(-2, -1)) / sample_shape.numel()


def standardize_moments(
    transform: Standardize,
    loc: Tensor,
    covariance_matrix: Tensor,
) -> Tuple[Tensor, Tensor]:

    m = transform.means.squeeze().unsqueeze(-1)
    s = transform.stdvs.squeeze().reciprocal().unsqueeze(-1)
    loc = s * (loc - m)
    correlation_matrix = s.unsqueeze(-1) * covariance_matrix * s.unsqueeze(-2)
    return loc, correlation_matrix
