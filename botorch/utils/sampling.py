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
from typing import Generator, Iterable, Optional, Tuple

import numpy as np
import scipy
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
    bounds: Tensor,
    n: int,
    q: int,
    batch_shape: Optional[Iterable[int], torch.Size] = None,
    seed: Optional[int] = None,
) -> Tensor:
    r"""Draw qMC samples from the box defined by bounds.

    Args:
        bounds: A `2 x d` dimensional tensor specifying box constraints on a
            `d`-dimensional space, where bounds[0, :] and bounds[1, :] correspond
            to lower and upper bounds, respectively.
        n: The number of (q-batch) samples.
        q: The size of each q-batch.
        batch_shape: The batch shape of the samples. If given, returns samples
            of shape `n x batch_shape x q x d`, where each batch is an
            `n x q x d`-dim tensor of qMC samples.
        seed: The seed used for initializing Owen scrambling. If None (default),
            use a random seed.

    Returns:
        A `n x batch_shape x q x d`-dim tensor of qMC samples from the box
        defined by bounds.

    Example:
        >>> bounds = torch.stack([torch.zeros(3), torch.ones(3)])
        >>> samples = draw_sobol_samples(bounds, 10, 2)
    """
    batch_shape = batch_shape or torch.Size()
    batch_size = int(torch.prod(torch.tensor(batch_shape)))
    d = bounds.shape[-1]
    lower = bounds[0]
    rng = bounds[1] - bounds[0]
    sobol_engine = SobolEngine(q * d, scramble=True, seed=seed)
    samples_raw = sobol_engine.draw(batch_size * n, dtype=lower.dtype)
    samples_raw = samples_raw.view(*batch_shape, n, q, d).to(device=lower.device)
    if batch_shape != torch.Size():
        samples_raw = samples_raw.permute(-3, *range(len(batch_shape)), -2, -1)
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


class PolytopeSampler:
    r"""
    Sampling points from a polytope described via a set of inequality and
    equality constraints.
    """

    def __init__(
        self,
        inequality_constraints: Tuple[Tensor, Tensor],
        n_burnin: int = 0,
        equality_constraints: Optional[Tuple[Tensor, Tensor]] = None,
        initial_point: Optional[Tensor] = None,
    ) -> None:
        r"""
        Args:
            inequality_constraints: Tensors (A, b) describing inequality
                constraints: A*x<=b, where A is (n_ineq_con, d_sample)-dim Tensor
                and b is (n_ineq_con, 1)-dim Tensor with n_ineq_con
                being the number of inequalities and d_sample the dimension
                of the sample space.
            n_burnin: The number of burn in samples.
            equality_constraints: Tensors (C, d) describing the equality
                constraints: C*x=d, where C is (n_eq_con, d_sample)-dim Tensor and
                d is (n_eq_con-dim, 1) Tensor with n_eq_con
                being the number of equalities.
            initial_point: An (d_sample, 1)-dim Tensor presenting an inital point of
                the chain satisfying all the conditions.
                Determined automatically (by solving an LP) if not provided.
        """
        self.A, self.b = inequality_constraints
        self.n_burnin = n_burnin
        self.equality_constraints = equality_constraints

        if equality_constraints is not None:
            self.C, self.d = equality_constraints
            U, S, V = torch.svd(self.C, some=False)
            r = torch.nonzero(S).size(0)  # rank of matrix C
            self.nullC = V[:, r:]  # orthonormal null space of C, satisfying
            # C @ nullC = 0 and nullC.T @ nullC = I
            # using the change of variables x=x0+nullC*y,
            # sample y satisfies A*nullC*y<=b-A*x0.
            # the linear constraint is automatically satisfied as x0 satisfies it.
        else:
            self.C = None
            self.d = None
            self.nullC = torch.eye(
                self.A.size(-1), dtype=self.A.dtype, device=self.A.device
            )

        self.new_A = self.A @ self.nullC  # doesn't depend on the initial point

        # initial point for the original, not transformed, problem
        if initial_point is not None:
            if self.feasible(initial_point):
                self.x0 = initial_point
            else:
                raise ValueError("The given input point is not feasible.")
        else:
            self.x0 = self.find_initial_point()

    def feasible(self, x: Tensor) -> bool:
        ineq = (self.A @ x - self.b <= 0).all()
        if self.equality_constraints is not None:
            eq = (self.C @ x - self.d == 0).all()
            return ineq & eq
        return ineq

    def find_initial_point(self):
        r"""
        Finds a feasible point of the original problem.
        Details: Solves the following LP:
        min -s such that Ax<=b-2*s, s>=0, plus equality constraints.
        The number of inequality constrains in LP: nrow(A) + 1.
        LP solution dimension: dim(x) + dim(s) = dim(x) + 1.
        """
        # inequality constraints: A_ub * (x, s) <= b_ub
        ncon = self.A.size(0) + 1
        dim = self.A.size(-1) + 1
        c = np.zeros(dim)
        c[-1] = -1
        b_ub = np.zeros(ncon)
        b_ub[:-1] = self.b.cpu().squeeze(-1).numpy()
        A_ub = np.zeros((ncon, dim))
        A_ub[:-1, :-1] = self.A.cpu().numpy()
        A_ub[:, -1] = 2.0
        A_ub[-1, -1] = -1.0

        if self.equality_constraints:
            # equality constraints: A_eq * (x, s) = b_eq
            A_eq = np.zeros((self.C.size(0), self.C.size(-1) + 1))
            A_eq[:, :-1] = self.C.cpu().numpy()
            b_eq = self.d.cpu().numpy()
        else:
            A_eq = None
            b_eq = None

        result = scipy.optimize.linprog(  # solving LP
            c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq
        )

        if result.status == 2:
            raise ValueError(
                "No feasible point found. Constraint polytope appears empty. "
                + "Check your constraints."
            )
        elif result.status > 0:
            raise ValueError(
                "Problem checking constraint specification. "
                + "linprog status: {}".format(result.message)
            )

        x0 = (
            torch.from_numpy(result.x[:-1])
            .to(dtype=self.A.dtype, device=self.A.device)
            .unsqueeze(-1)
        )

        return x0

    def draw(self, n: int = 1, seed: Optional[int] = None) -> Tensor:

        transformed_samples = sample_polytope(
            A=self.new_A,
            b=self.b - self.A @ self.x0,
            x0=torch.zeros(
                (self.nullC.size(1), 1), dtype=self.A.dtype, device=self.A.device
            ),
            n=n,
            n0=self.n_burnin,
            seed=seed,
        )

        init_shift = self.x0.transpose(-1, -2)
        samples = init_shift + transformed_samples @ self.nullC.transpose(-1, -2)

        # keep the last element of the resulting chain as
        # the beginning of the next chain
        self.x0 = samples[-1].reshape(-1, 1)
        # next time the sampling is called there won't be any burn-in
        self.n_burnin = 0
        return samples


def sample_polytope(
    A: Tensor,
    b: Tensor,
    x0: Tensor,
    n: int = 10000,
    n0: int = 100,
    seed: Optional[int] = None,
) -> Tensor:
    r"""
    Hit and run sampler from uniform sampling points from a polytope,
    described via inequality constraints A*x<=b.

    Args:
        A: A Tensor describing inequality constraints
            so that all samples satisfy Ax<=b.
        b: A Tensor describing the inequality constraints
            so that all samples satisfy Ax<=b.
        x0: d dim Tensor representing a starting point of the chain
            satisfying the constraints.
        n: The number of resulting samples kept in the output.
        n0: The number of burn-in samples. The chain will produce
            n+n0 samples but the first n0 samples are not saved.
        seed: The seed for the sampler. If omitted, use a random seed.

    Returns:
        (n, d) dim Tensor containing the resulting samples.
    """
    n_tot = n + n0
    seed = seed if seed is not None else torch.randint(0, 1000000, (1,)).item()
    with manual_seed(seed=seed):
        rands = torch.rand(n_tot, dtype=A.dtype, device=A.device)

    # pre-sample samples from hypersphere
    d = x0.size(0)
    # uniform samples from unit ball in d dims
    Rs = sample_hypersphere(d=d, n=n_tot, dtype=A.dtype, device=A.device).unsqueeze(-1)

    # compute matprods in batch
    ARs = (A @ Rs).squeeze(-1)
    out = torch.empty(n, A.size(-1), dtype=A.dtype, device=A.device)
    x = x0.clone()
    for i, (ar, r, rnd) in enumerate(zip(ARs, Rs, rands)):
        # given x, the next point in the chain is x+alpha*r
        # it also satisfies A(x+alpha*r)<=b which implies A*alpha*r<=b-Ax
        # so alpha<=(b-Ax)/ar for ar>0, and alpha>=(b-Ax)/ar for ar<0.
        w = (b - A @ x).squeeze() / ar  # b - A @ x is always >= 0
        pos = w >= 0
        alpha_max = w[pos].min()
        # important to include equality here in cases x is at the boundary
        # of the polytope
        neg = w <= 0
        alpha_min = w[neg].max()
        # alpha~Unif[alpha_min, alpha_max]
        alpha = alpha_min + rnd * (alpha_max - alpha_min)
        x = x + alpha * r
        if i >= n0:  # save samples after burn-in period
            out[i - n0] = x.squeeze()
    return out
