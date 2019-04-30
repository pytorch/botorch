#!/usr/bin/env python3

r"""
Utilities for MC and qMC sampling.
"""

import warnings
from contextlib import contextmanager
from typing import Generator, Optional

import torch
from torch import Tensor
from torch.quasirandom import SobolEngine

from ..exceptions.warnings import SamplingWarning
from ..posteriors.posterior import Posterior
from ..qmc.normal import NormalQMCEngine


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
        output_shape: The output shape (`q x o`) of the base samples to generate.
        sample_shape: The sample shape of the samples to draw.
        qmc: If True, use quasi-MC sampling (instead of iid draws).
        seed: If provided, use as a seed for the RNG.

    Returns:
        A `sample_shape x batch_shape x output_shape` dimensional tensor of base
        samples, drawn from a N(0, I_qo) distribution (using QMC if `qmc=True`).
        Here `output_shape = q x o`.

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
        A `num_samples x 1 x q x o` dimensional Tensor of base samples, drawn
        from a N(0, I_qo) distribution (using QMC if `qmc=True`). Here `q` and
        `o` are the same as in the posterior's `event_shape` `b x q x o`.
        Importantly, this only obtain a single t-batch of samples, so as to not
        introduce any sampling variance across t-batches.

    Example:
        >>> sample_shape = torch.Size([10])
        >>> samples = construct_base_samples_from_posterior(posterior, sample_shape)
    """
    output_shape = posterior.event_shape[-2:]  # shape of joint output: q x o
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
        >>> samples = draw_sobol_samples(bounds, 10, 2)a
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
        d: The dimension of the normal distribution
        n: The number of samples to return
        device: The torch device
        dtype:  The torch dtype
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
