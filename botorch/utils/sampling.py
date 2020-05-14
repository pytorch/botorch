#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Utilities for MC and qMC sampling.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from typing import Generator, Optional

import torch
from botorch.exceptions.warnings import SamplingWarning
from botorch.posteriors.posterior import Posterior
from botorch.sampling.qmc import NormalQMCEngine
from torch import LongTensor, Tensor
from torch.quasirandom import SobolEngine


@contextmanager
def manual_seed(seed: Optional[int] = None) -> Generator[None, None, None]:
    r"""Contextmanager for manual setting the torch.random seed.

    Args:
        seed: The seed to set the random number generator to.

    Returns:
        Generator

    Example:
        >>> with manual_seed(1234):
        >>>     X = torch.rand(3)
    """
    old_state = torch.random.get_rng_state()
    try:
        if seed is not None:
            torch.random.manual_seed(seed)
        yield
    finally:
        if seed is not None:
            torch.random.set_rng_state(old_state)


def construct_base_samples(
    batch_shape: torch.Size,
    output_shape: torch.Size,
    sample_shape: torch.Size,
    qmc: bool = True,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    r"""Construct base samples from a multi-variate standard normal N(0, I_qo).

    Args:
        batch_shape: The batch shape of the base samples to generate. Typically,
            this is used with each dimension of size 1, so as to eliminate
            sampling variance across batches.
        output_shape: The output shape (`q x m`) of the base samples to generate.
        sample_shape: The sample shape of the samples to draw.
        qmc: If True, use quasi-MC sampling (instead of iid draws).
        seed: If provided, use as a seed for the RNG.

    Returns:
        A `sample_shape x batch_shape x mutput_shape` dimensional tensor of base
        samples, drawn from a N(0, I_qm) distribution (using QMC if `qmc=True`).
        Here `output_shape = q x m`.

    Example:
        >>> batch_shape = torch.Size([2])
        >>> output_shape = torch.Size([3])
        >>> sample_shape = torch.Size([10])
        >>> samples = construct_base_samples(batch_shape, output_shape, sample_shape)
    """
    base_sample_shape = batch_shape + output_shape
    output_dim = output_shape.numel()
    if qmc and output_dim <= SobolEngine.MAXDIM:
        n = (sample_shape + batch_shape).numel()
        base_samples = draw_sobol_normal_samples(
            d=output_dim, n=n, device=device, dtype=dtype, seed=seed
        )
        base_samples = base_samples.view(sample_shape + base_sample_shape)
    else:
        if qmc and output_dim > SobolEngine.MAXDIM:
            warnings.warn(
                f"Number of output elements (q*d={output_dim}) greater than "
                f"maximum supported by qmc ({SobolEngine.MAXDIM}). "
                "Using iid sampling instead.",
                SamplingWarning,
            )
        with manual_seed(seed=seed):
            base_samples = torch.randn(
                sample_shape + base_sample_shape, device=device, dtype=dtype
            )
    return base_samples


def construct_base_samples_from_posterior(
    posterior: Posterior,
    sample_shape: torch.Size,
    qmc: bool = True,
    collapse_batch_dims: bool = True,
    seed: Optional[int] = None,
) -> Tensor:
    r"""Construct a tensor of normally distributed base samples.

    Args:
        posterior: A Posterior object.
        sample_shape: The sample shape of the samples to draw.
        qmc: If True, use quasi-MC sampling (instead of iid draws).
        seed: If provided, use as a seed for the RNG.

    Returns:
        A `num_samples x 1 x q x m` dimensional Tensor of base samples, drawn
        from a N(0, I_qm) distribution (using QMC if `qmc=True`). Here `q` and
        `m` are the same as in the posterior's `event_shape` `b x q x m`.
        Importantly, this only obtain a single t-batch of samples, so as to not
        introduce any sampling variance across t-batches.

    Example:
        >>> sample_shape = torch.Size([10])
        >>> samples = construct_base_samples_from_posterior(posterior, sample_shape)
    """
    output_shape = posterior.event_shape[-2:]  # shape of joint output: q x m
    if collapse_batch_dims:
        batch_shape = torch.Size([1] * len(posterior.event_shape[:-2]))
    else:
        batch_shape = posterior.event_shape[:-2]
    base_samples = construct_base_samples(
        batch_shape=batch_shape,
        output_shape=output_shape,
        sample_shape=sample_shape,
        qmc=qmc,
        seed=seed,
        device=posterior.device,
        dtype=posterior.dtype,
    )
    return base_samples


def draw_sobol_samples(
    bounds: Tensor, n: int, q: int, seed: Optional[int] = None
) -> Tensor:
    r"""Draw qMC samples from the box defined by bounds.

    Args:
        bounds: A `2 x d` dimensional tensor specifying box constraints on a
            `d`-dimensional space, where bounds[0, :] and bounds[1, :] correspond
            to lower and upper bounds, respectively.
        n: The number of (q-batch) samples.
        q: The size of each q-batch.
        seed: The seed used for initializing Owen scrambling. If None (default),
            use a random seed.

    Returns:
        A `n x q x d`-dim tensor of qMC samples from the box defined by bounds.

    Example:
        >>> bounds = torch.stack([torch.zeros(3), torch.ones(3)])
        >>> samples = draw_sobol_samples(bounds, 10, 2)
    """
    d = bounds.shape[-1]
    lower = bounds[0]
    rng = bounds[1] - bounds[0]
    sobol_engine = SobolEngine(d, scramble=True, seed=seed)
    samples_raw = sobol_engine.draw(n * q, dtype=lower.dtype).view(n, q, d)
    samples_raw = samples_raw.to(device=lower.device)
    return lower + rng * samples_raw


def draw_sobol_normal_samples(
    d: int,
    n: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    seed: Optional[int] = None,
) -> Tensor:
    r"""Draw qMC samples from a multi-variate standard normal N(0, I_d)

    A primary use-case for this functionality is to compute an QMC average
    of f(X) over X where each element of X is drawn N(0, 1).

    Args:
        d: The dimension of the normal distribution.
        n: The number of samples to return.
        device: The torch device.
        dtype:  The torch dtype.
        seed: The seed used for initializing Owen scrambling. If None (default),
            use a random seed.

    Returns:
        A tensor of qMC standard normal samples with dimension `n x d` with device
        and dtype specified by the input.

    Example:
        >>> samples = draw_sobol_normal_samples(2, 10)
    """
    normal_qmc_engine = NormalQMCEngine(d=d, seed=seed, inv_transform=True)
    samples = normal_qmc_engine.draw(n, dtype=torch.float if dtype is None else dtype)
    return samples.to(device=device)


def sample_hypersphere(
    d: int,
    n: int = 1,
    qmc: bool = False,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    r"""Sample uniformly from a unit d-sphere.

    Args:
        d: The dimension of the hypersphere.
        n: The number of samples to return.
        qmc: If True, use QMC Sobol sampling (instead of i.i.d. uniform).
        seed: If provided, use as a seed for the RNG.
        device: The torch device.
        dtype:  The torch dtype.

    Returns:
        An  `n x d` tensor of uniform samples from from the d-hypersphere.

    Example:
        >>> sample_hypersphere(d=5, n=10)
    """
    dtype = torch.float if dtype is None else dtype
    if d == 1:
        rnd = torch.randint(0, 2, (n, 1), device=device, dtype=dtype)
        return 2 * rnd - 1
    if qmc:
        rnd = draw_sobol_normal_samples(d=d, n=n, device=device, dtype=dtype, seed=seed)
    else:
        with manual_seed(seed=seed):
            rnd = torch.randn(n, d, dtype=dtype)
    samples = rnd / torch.norm(rnd, dim=-1, keepdim=True)
    if device is not None:
        samples = samples.to(device)
    return samples


def sample_simplex(
    d: int,
    n: int = 1,
    qmc: bool = False,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    r"""Sample uniformly from a d-simplex.

    Args:
        d: The dimension of the simplex.
        n: The number of samples to return.
        qmc: If True, use QMC Sobol sampling (instead of i.i.d. uniform).
        seed: If provided, use as a seed for the RNG.
        device: The torch device.
        dtype:  The torch dtype.

    Returns:
        An `n x d` tensor of uniform samples from from the d-simplex.

    Example:
        >>> sample_simplex(d=3, n=10)
    """
    dtype = torch.float if dtype is None else dtype
    if d == 1:
        return torch.ones(n, 1, device=device, dtype=dtype)
    if qmc:
        sobol_engine = SobolEngine(d - 1, scramble=True, seed=seed)
        rnd = sobol_engine.draw(n, dtype=dtype)
    else:
        with manual_seed(seed=seed):
            rnd = torch.rand(n, d - 1, dtype=dtype)
    srnd, _ = torch.sort(rnd, dim=-1)
    zeros = torch.zeros(n, 1, dtype=dtype)
    ones = torch.ones(n, 1, dtype=dtype)
    srnd = torch.cat([zeros, srnd, ones], dim=-1)
    if device is not None:
        srnd = srnd.to(device)
    return srnd[..., 1:] - srnd[..., :-1]


def batched_multinomial(
    weights: Tensor,
    num_samples: int,
    replacement: bool = False,
    generator: Optional[torch.Generator] = None,
    out: Optional[Tensor] = None,
) -> LongTensor:
    r"""Sample from multinomial with an arbitrary number of batch dimensions.

    Args:
        weights: A `batch_shape x num_categories` tensor of weights. For each batch
            index `i, j, ...`, this functions samples from a multinomial with `input`
            `weights[i, j, ..., :]`. Note that the weights need not sum to one, but must
            be non-negative, finite and have a non-zero sum.
        num_samples: The number of samples to draw for each batch index. Must be smaller
            than `num_categories` if `replacement=False`.
        replacement: If True, samples are drawn with replacement.
        generator: A a pseudorandom number generator for sampling.
        out: The output tensor (optional). If provided, must be of size
            `batch_shape x num_samples`.

    Returns:
        A `batch_shape x num_samples` tensor of samples.

    This is a thin wrapper around `torch.multinomial` that allows weight (`input`)
    tensors with an arbitrary number of batch dimensions (`torch.multinomial` only
    allows a single batch dimension). The calling signature is the same as for
    `torch.multinomial`.

    Example:
        >>> weights = torch.rand(2, 3, 10)
        >>> samples = batched_multinomial(weights, 4)  # shape is 2 x 3 x 4
    """
    batch_shape, n_categories = weights.shape[:-1], weights.size(-1)
    flat_samples = torch.multinomial(
        input=weights.view(-1, n_categories),
        num_samples=num_samples,
        replacement=replacement,
        generator=generator,
        out=None if out is None else out.view(-1, num_samples),
    )
    return flat_samples.view(*batch_shape, num_samples)
