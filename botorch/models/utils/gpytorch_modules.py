#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
from gpytorch.constraints.constraints import GreaterThan
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood
from gpytorch.priors.torch_priors import GammaPrior

MIN_INFERRED_NOISE_LEVEL = 1e-4


def get_matern_kernel_with_gamma_prior(
    ard_num_dims: int, batch_shape: Optional[torch.Size] = None
) -> ScaleKernel:
    r"""Constructs the Scale-Matern kernel that is used by default by
    several models. This uses a Gamma(3.0, 6.0) prior for the lengthscale
    and a Gamma(2.0, 0.15) prior for the output scale.
    """
    return ScaleKernel(
        base_kernel=MaternKernel(
            nu=2.5,
            ard_num_dims=ard_num_dims,
            batch_shape=batch_shape,
            lengthscale_prior=GammaPrior(3.0, 6.0),
        ),
        batch_shape=batch_shape,
        outputscale_prior=GammaPrior(2.0, 0.15),
    )


def get_gaussian_likelihood_with_gamma_prior(
    batch_shape: Optional[torch.Size] = None,
) -> GaussianLikelihood:
    r"""Constructs the GaussianLikelihood that is used by default by
    several models. This uses a Gamma(1.1, 0.05) prior and constrains the
    noise level to be greater than MIN_INFERRED_NOISE_LEVEL (=1e-4).
    """
    batch_shape = torch.Size() if batch_shape is None else batch_shape
    noise_prior = GammaPrior(1.1, 0.05)
    noise_prior_mode = (noise_prior.concentration - 1) / noise_prior.rate
    return GaussianLikelihood(
        noise_prior=noise_prior,
        batch_shape=batch_shape,
        noise_constraint=GreaterThan(
            MIN_INFERRED_NOISE_LEVEL,
            transform=None,
            initial_value=noise_prior_mode,
        ),
    )
