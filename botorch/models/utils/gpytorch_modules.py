#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from math import log, sqrt
from typing import Optional, Union

import torch
from gpytorch.constraints.constraints import GreaterThan
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood
from gpytorch.priors.torch_priors import GammaPrior, LogNormalPrior

MIN_INFERRED_NOISE_LEVEL = 1e-4
SQRT2 = sqrt(2)
SQRT3 = sqrt(3)


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


def get_gaussian_likelihood_with_lognormal_prior(
    batch_shape: Optional[torch.Size] = None,
) -> GaussianLikelihood:
    """Return Gaussian likelihood with a LogNormal(-4.0, 1.0) prior.
    This prior is based on Hvarfner et al (2024) "Vanilla Bayesian Optimization
    Performs Great in High Dimensions".
    https://arxiv.org/abs/2402.02229
    https://github.com/hvarfner/vanilla_bo_in_highdim

    Args:
        batch_shape: Batch shape for the likelihood.

    Returns:
        GaussianLikelihood with LogNormal(-4.0, 1.0) prior constrained to [1e-4, 1.0].
    """
    batch_shape = torch.Size() if batch_shape is None else batch_shape
    noise_prior = LogNormalPrior(loc=-4.0, scale=1.0)
    return GaussianLikelihood(
        noise_prior=noise_prior,
        batch_shape=batch_shape,
        noise_constraint=GreaterThan(
            MIN_INFERRED_NOISE_LEVEL,
            transform=None,
            initial_value=noise_prior.mode,
        ),
    )


def get_covar_module_with_dim_scaled_prior(
    dim: int,
    batch_shape: Optional[torch.Size] = None,
    use_matern_kernel: bool = False,
    learn_outputscale: bool = False,
) -> Union[MaternKernel, RBFKernel, ScaleKernel]:
    """Returns an RBF or Matern kernel (with optional output scale) with priors
    from Hvarfner et al (2024) "Vanilla Bayesian Optimization Performs Great in
    High Dimensions".
    https://arxiv.org/abs/2402.02229
    https://github.com/hvarfner/vanilla_bo_in_highdim

    Args:
        dim: Number of feature dimensions. Always uses ARD.
        batch_shape: Batch shape for the covariance module.
        use_matern_kernel: Whether to use a Matern kernel. If False, uses an RBF kernel.
        trainable_outputscale: Whether to add an output scale using a ScaleKernel.

    Returns:
        A Kernel constructed according to the given arguments. d
    """
    base_class = MaternKernel if use_matern_kernel else RBFKernel
    lengthscale_prior = LogNormalPrior(loc=SQRT2 + log(dim) * 0.5, scale=SQRT3)
    base_kernel = base_class(
        ard_num_dims=dim,
        batch_shape=batch_shape,
        lengthscale_prior=lengthscale_prior,
        lengthscale_constraint=GreaterThan(
            2.5e-2, transform=None, initial_value=lengthscale_prior.mode
        ),
    )
    if learn_outputscale:
        return ScaleKernel(
            base_kernel=base_kernel,
            batch_shape=batch_shape,
            outputscale_prior=GammaPrior(2.0, 0.15),
        )
    return base_kernel
