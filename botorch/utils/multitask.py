#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Helpers for multitask modeling.
"""

from __future__ import annotations

import torch
from gpytorch.distributions import MultitaskMultivariateNormal
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from linear_operator import to_linear_operator


def separate_mtmvn(mvn: MultitaskMultivariateNormal) -> list[MultivariateNormal]:
    """
    Separate a MTMVN into a list of MVNs, where covariance across data within each task
    are preserved, while covariance across task are dropped.
    """
    # T150340766 Upstream as a class method on gpytorch MultitaskMultivariateNormal.
    full_covar = mvn.lazy_covariance_matrix
    num_data, num_tasks = mvn.mean.shape[-2:]
    if mvn._interleaved:
        data_indices = torch.arange(
            0, num_data * num_tasks, num_tasks, device=full_covar.device
        ).view(-1, 1, 1)
        task_indices = torch.arange(num_tasks, device=full_covar.device)
    else:
        data_indices = torch.arange(num_data, device=full_covar.device).view(-1, 1, 1)
        task_indices = torch.arange(
            0, num_data * num_tasks, num_data, device=full_covar.device
        )
    slice_ = (data_indices + task_indices).transpose(-1, -3)
    data_covars = full_covar[..., slice_, slice_.transpose(-1, -2)]
    mvns = []
    for c in range(num_tasks):
        mvns.append(
            MultivariateNormal(
                mvn.mean[..., c], to_linear_operator(data_covars[..., c, :, :])
            )
        )
    return mvns
