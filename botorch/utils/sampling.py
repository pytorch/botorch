#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Utilities for MC and qMC sampling.

References

.. [Trikalinos2014polytope]
    T. A. Trikalinos and G. van Valkenhoef. Efficient sampling from uniform
    density n-polytopes. Technical report, Brown University, 2014.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Generator, Iterable, List, Optional, Tuple

import numpy as np
import scipy
import torch
from botorch.exceptions.errors import BotorchError
from botorch.exceptions.warnings import SamplingWarning
from botorch.posteriors.posterior import Posterior
from botorch.sampling.qmc import NormalQMCEngine
from scipy.spatial import Delaunay, HalfspaceIntersection
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
        n: The number of (q-batch) samples. As a best practice, use powers of 2.
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
        >>> samples = draw_sobol_samples(bounds, 16, 2)
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
    r"""Draw qMC samples from a multi-variate standard normal N(0, I_d).

    A primary use-case for this functionality is to compute an QMC average
    of f(X) over X where each element of X is drawn N(0, 1).

    Args:
        d: The dimension of the normal distribution.
        n: The number of samples to return. As a best practice, use powers of 2.
        device: The torch device.
        dtype:  The torch dtype.
        seed: The seed used for initializing Owen scrambling. If None (default),
            use a random seed.

    Returns:
        A tensor of qMC standard normal samples with dimension `n x d` with device
        and dtype specified by the input.

    Example:
        >>> samples = draw_sobol_normal_samples(2, 16)
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
        dtype: The torch dtype.

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
        x0: A `d`-dim Tensor representing a starting point of the chain
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
        # b - A @ x is always >= 0, clamping for numerical tolerances
        w = (b - A @ x).squeeze().clamp(min=0.0) / ar
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


def _convert_bounds_to_inequality_constraints(bounds: Tensor) -> Tuple[Tensor, Tensor]:
    r"""Convert bounds into inequality constraints of the form Ax <= b.

    Args:
        bounds: A `2 x d`-dim tensor of bounds

    Returns:
        A two-element tuple containing
            - A: A `2d x d`-dim tensor of coefficients
            - b: A `2d x 1`-dim tensor containing the right hand side
    """
    d = bounds.shape[-1]
    eye = torch.eye(d, dtype=bounds.dtype, device=bounds.device)
    lower, upper = bounds
    lower_finite, upper_finite = bounds.isfinite()
    A = torch.cat([-eye[lower_finite], eye[upper_finite]], dim=0)
    b = torch.cat([-lower[lower_finite], upper[upper_finite]], dim=0).unsqueeze(-1)
    return A, b


def find_interior_point(
    A: np.ndarray,
    b: np.ndarray,
    A_eq: Optional[np.ndarray] = None,
    b_eq: Optional[np.ndarray] = None,
) -> np.ndarray:
    r"""Find an interior point of a polytope via linear programming.

    Args:
        A: A `n_ineq x d`-dim numpy array containing the coefficients of the
            constraint inequalities.
        b: A `n_ineq x 1`-dim numpy array containing the right hand sides of
            the constraint inequalities.
        A_eq: A `n_eq x d`-dim numpy array containing the coefficients of the
            constraint equalities.
        b_eq: A `n_eq x 1`-dim numpy array containing the right hand sides of
            the constraint equalities.

    Returns:
        A `d`-dim numpy array containing an interior point of the polytope.
        This function will raise a ValueError if there is no such point.

    This method solves the following Linear Program:

        min -s subject to A @ x <= b - 2 * s, s >= 0, A_eq @ x = b_eq

    In case the polytope is unbounded, then it will also constrain the slack
    variable `s` to `s<=1`.
    """
    # augment inequality constraints: A @ (x, s) <= b
    d = A.shape[-1]
    ncon = A.shape[-2] + 1
    c = np.zeros(d + 1)
    c[-1] = -1
    b_ub = np.zeros(ncon)
    b_ub[:-1] = b.reshape(-1)
    A_ub = np.zeros((ncon, d + 1))
    A_ub[:-1, :-1] = A
    A_ub[:-1, -1] = 2.0
    A_ub[-1, -1] = -1.0

    result = scipy.optimize.linprog(
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=(None, None),
        method="highs",
    )

    if result.status == 3:
        # problem is unbounded - to find a bounded solution we constrain the
        # slack variable `s` to `s <= 1.0`.
        A_s = np.concatenate([np.zeros((1, d)), np.ones((1, 1))], axis=-1)
        A_ub = np.concatenate([A_ub, A_s], axis=0)
        b_ub = np.concatenate([b_ub, np.ones(1)], axis=-1)
        result = scipy.optimize.linprog(
            c=c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=(None, None),
            method="highs",
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
    # the x in the result is really (x, s)
    return result.x[:-1]


class PolytopeSampler(ABC):
    r"""
    Base class for samplers that sample points from a polytope.

    :meta private:
    """

    def __init__(
        self,
        inequality_constraints: Optional[Tuple[Tensor, Tensor]] = None,
        equality_constraints: Optional[Tuple[Tensor, Tensor]] = None,
        bounds: Optional[Tensor] = None,
        interior_point: Optional[Tensor] = None,
    ) -> None:
        r"""
        Args:
            inequality_constraints: Tensors `(A, b)` describing inequality
                constraints `A @ x <= b`, where `A` is a `n_ineq_con x d`-dim
                Tensor and `b` is a `n_ineq_con x 1`-dim Tensor, with `n_ineq_con`
                the number of inequalities and `d` the dimension of the sample space.
            equality_constraints: Tensors `(C, d)` describing the equality constraints
                `C @ x = d`, where `C` is a `n_eq_con x d`-dim Tensor and `d` is a
                `n_eq_con x 1`-dim Tensor with `n_eq_con` the number of equalities.
            bounds: A `2 x d`-dim tensor of box bounds, where `inf` (`-inf`) means
                that the respective dimension is unbounded above (below).
            interior_point: A `d x 1`-dim Tensor presenting a point in the
                (relative) interior of the polytope. If omitted, determined
                automatically by solving a Linear Program.
        """
        if inequality_constraints is None:
            if bounds is None:
                raise BotorchError(
                    "PolytopeSampler requires either inequality constraints or bounds."
                )
            A = torch.empty(
                0, bounds.shape[-1], dtype=bounds.dtype, device=bounds.device
            )
            b = torch.empty(0, 1, dtype=bounds.dtype, device=bounds.device)
        else:
            A, b = inequality_constraints
        if bounds is not None:
            # add inequality constraints for bounds
            # TODO: make sure there are not deduplicate constraints
            A2, b2 = _convert_bounds_to_inequality_constraints(bounds=bounds)
            A = torch.cat([A, A2], dim=0)
            b = torch.cat([b, b2], dim=0)
        self.A = A
        self.b = b
        self.equality_constraints = equality_constraints

        if equality_constraints is not None:
            self.C, self.d = equality_constraints
            U, S, Vh = torch.linalg.svd(self.C)
            r = torch.nonzero(S).size(0)  # rank of matrix C
            self.nullC = Vh[r:, :].transpose(-1, -2)  # orthonormal null space of C,
            # satisfying # C @ nullC = 0 and nullC.T @ nullC = I
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
        if interior_point is not None:
            if self.feasible(interior_point):
                self.x0 = interior_point
            else:
                raise ValueError("The given input point is not feasible.")
        else:
            self.x0 = self.find_interior_point()

    def feasible(self, x: Tensor) -> bool:
        r"""Check whether a point is contained in the polytope.

        Args:
            x: A `d x 1`-dim Tensor.

        Returns:
            True if `x` is contained inside the polytope (incl. its boundary),
            False otherwise.
        """
        ineq = (self.A @ x - self.b <= 0).all()
        if self.equality_constraints is not None:
            eq = (self.C @ x - self.d == 0).all()
            return ineq & eq
        return ineq

    def find_interior_point(self) -> Tensor:
        r"""Find an interior point of the polytope.

        Returns:
            A `d x 1`-dim Tensor representing a point contained in the polytope.
            This function will raise a ValueError if there is no such point.
        """
        if self.equality_constraints:
            # equality constraints: A_eq * (x, s) = b_eq
            A_eq = np.zeros((self.C.size(0), self.C.size(-1) + 1))
            A_eq[:, :-1] = self.C.cpu().numpy()
            b_eq = self.d.cpu().numpy()
        else:
            A_eq = None
            b_eq = None
        x0 = find_interior_point(
            A=self.A.cpu().numpy(), b=self.b.cpu().numpy(), A_eq=A_eq, b_eq=b_eq
        )
        return torch.from_numpy(x0).to(self.A).unsqueeze(-1)

    # -------- Abstract methods to be implemented by subclasses -------- #

    @abstractmethod
    def draw(self, n: int = 1, seed: Optional[int] = None) -> Tensor:
        r"""Draw samples from the polytope.

        Args:
            n: The number of samples.
            seed: The random seed.

        Returns:
            A `n x d` Tensor of samples from the polytope.
        """
        pass  # pragma: no cover


class HitAndRunPolytopeSampler(PolytopeSampler):
    r"""A sampler for sampling from a polyope using a hit-and-run algorithm."""

    def __init__(
        self,
        inequality_constraints: Optional[Tuple[Tensor, Tensor]] = None,
        equality_constraints: Optional[Tuple[Tensor, Tensor]] = None,
        bounds: Optional[Tensor] = None,
        interior_point: Optional[Tensor] = None,
        n_burnin: int = 0,
    ) -> None:
        r"""A sampler for sampling from a polyope using a hit-and-run algorithm.

        Args:
            inequality_constraints: Tensors `(A, b)` describing inequality
                constraints `A @ x <= b`, where `A` is a `n_ineq_con x d`-dim
                Tensor and `b` is a `n_ineq_con x 1`-dim Tensor, with `n_ineq_con`
                the number of inequalities and `d` the dimension of the sample space.
            equality_constraints: Tensors `(C, d)` describing the equality constraints
                `C @ x = d`, where `C` is a `n_eq_con x d`-dim Tensor and `d` is a
                `n_eq_con x 1`-dim Tensor with `n_eq_con` the number of equalities.
            bounds: A `2 x d`-dim tensor of box bounds, where `inf` (`-inf`) means
                that the respective dimension is unbounded from above (below).
            interior_point: A `d x 1`-dim Tensor representing a point in the
                (relative) interior of the polytope. If omitted, determined
                automatically by solving a Linear Program.
            n_burnin: The number of burn in samples.
        """
        super().__init__(
            inequality_constraints=inequality_constraints,
            equality_constraints=equality_constraints,
            bounds=bounds,
            interior_point=interior_point,
        )
        self.n_burnin = n_burnin

    def draw(self, n: int = 1, seed: Optional[int] = None) -> Tensor:
        r"""Draw samples from the polytope.

        Args:
            n: The number of samples.
            seed: The random seed.

        Returns:
            A `n x d` Tensor of samples from the polytope.
        """
        transformed_samples = sample_polytope(
            # run this on the cpu
            A=self.new_A.cpu(),
            b=(self.b - self.A @ self.x0).cpu(),
            x0=torch.zeros((self.nullC.size(1), 1), dtype=self.A.dtype),
            n=n,
            n0=self.n_burnin,
            seed=seed,
        ).to(self.b)
        init_shift = self.x0.transpose(-1, -2)
        samples = init_shift + transformed_samples @ self.nullC.transpose(-1, -2)
        # keep the last element of the resulting chain as
        # the beginning of the next chain
        self.x0 = samples[-1].reshape(-1, 1)
        # reset counter so there is no burn-in for subsequent samples
        self.n_burnin = 0
        return samples


class DelaunayPolytopeSampler(PolytopeSampler):
    r"""A polytope sampler using Delaunay triangulation.

    This sampler first enumerates the vertices of the constraint polytope and
    then uses a Delaunay triangulation to tesselate its convex hull.

    The sampling happens in two stages:
    1. First, we sample from the set of hypertriangles generated by the
    Delaunay triangulation (i.e. which hyper-triangle to draw the sample
    from) with probabilities proportional to the triangle volumes.
    2. Then, we sample uniformly from the chosen hypertriangle by sampling
    uniformly from the unit simplex of the appropriate dimension, and
    then computing the convex combination of the vertices of the
    hypertriangle according to that draw from the simplex.

    The best reference (not exactly the same, but functionally equivalent) is
    [Trikalinos2014polytope]_. A simple R implementation is available at
    https://github.com/gertvv/tesselample.
    """

    def __init__(
        self,
        inequality_constraints: Optional[Tuple[Tensor, Tensor]] = None,
        equality_constraints: Optional[Tuple[Tensor, Tensor]] = None,
        bounds: Optional[Tensor] = None,
        interior_point: Optional[Tensor] = None,
    ) -> None:
        r"""Initialize DelaunayPolytopeSampler.

        Args:
            inequality_constraints: Tensors `(A, b)` describing inequality
                constraints `A @ x <= b`, where `A` is a `n_ineq_con x d`-dim
                Tensor and `b` is a `n_ineq_con x 1`-dim Tensor, with `n_ineq_con`
                the number of inequalities and `d` the dimension of the sample space.
            equality_constraints: Tensors `(C, d)` describing the equality constraints
                `C @ x = d`, where `C` is a `n_eq_con x d`-dim Tensor and `d` is a
                `n_eq_con x 1`-dim Tensor with `n_eq_con` the number of equalities.
            bounds: A `2 x d`-dim tensor of box bounds, where `inf` (`-inf`) means
                that the respective dimension is unbounded from above (below).
            interior_point: A `d x 1`-dim Tensor representing a point in the
                (relative) interior of the polytope. If omitted, determined
                automatically by solving a Linear Program.

        Warning: The vertex enumeration performed in this algorithm can become
        extremely costly if there are a large number of inequalities. Similarly,
        the triangulation can get very expensive in high dimensions. Only use
        this algorithm for moderate dimensions / moderately complex constraint sets.
        An alternative is the `HitAndRunPolytopeSampler`.
        """
        super().__init__(
            inequality_constraints=inequality_constraints,
            equality_constraints=equality_constraints,
            bounds=bounds,
            interior_point=interior_point,
        )
        # shift coordinate system to be anchored at x0
        new_b = self.b - self.A @ self.x0
        if self.new_A.shape[-1] < 2:
            # if the polytope is in dim 1 (i.e. a line segment) Qhull won't work
            tshlds = new_b / self.new_A
            neg = self.new_A < 0
            self.y_min = tshlds[neg].max()
            self.y_max = tshlds[~neg].min()
            self.dim = 1
        else:
            # Qhull expects inputs of the form A @ x + b <= 0, so we need to negate here
            halfspaces = torch.cat([self.new_A, -new_b], dim=-1).cpu().numpy()
            vertices = HalfspaceIntersection(
                halfspaces=halfspaces, interior_point=np.zeros(self.new_A.shape[-1])
            ).intersections
            self.dim = vertices.shape[-1]
            try:
                delaunay = Delaunay(vertices)
            except ValueError as e:
                if "Points cannot contain NaN" in str(e):
                    raise ValueError("Polytope is unbounded.")
                raise e  # pragma: no cover
            polytopes = torch.from_numpy(
                np.array([delaunay.points[s] for s in delaunay.simplices]),
            ).to(self.A)
            volumes = torch.stack([torch.det(p[1:] - p[0]).abs() for p in polytopes])
            self._polytopes = polytopes
            self._p = volumes / volumes.sum()

    def draw(self, n: int = 1, seed: Optional[int] = None) -> Tensor:
        r"""Draw samples from the polytope.

        Args:
            n: The number of samples.
            seed: The random seed.

        Returns:
            A `n x d` Tensor of samples from the polytope.
        """
        if self.dim == 1:
            with manual_seed(seed):
                e = torch.rand(n, 1, device=self.new_A.device, dtype=self.new_A.dtype)
            transformed_samples = self.y_min + (self.y_max - self.y_min) * e
        else:
            if seed is None:
                generator = None
            else:
                generator = torch.Generator(device=self.A.device)
                generator.manual_seed(seed)
            index_rvs = torch.multinomial(
                self._p,
                num_samples=n,
                replacement=True,
                generator=generator,
            )
            simplex_rvs = sample_simplex(
                d=self.dim + 1, n=n, seed=seed, device=self.A.device, dtype=self.A.dtype
            )
            transformed_samples = torch.stack(
                [rv @ self._polytopes[idx] for rv, idx in zip(simplex_rvs, index_rvs)]
            )
        init_shift = self.x0.transpose(-1, -2)
        samples = init_shift + transformed_samples @ self.nullC.transpose(-1, -2)
        return samples


def normalize_linear_constraints(
    bounds: Tensor, constraints: List[Tuple[Tensor, Tensor, float]]
) -> List[Tuple[Tensor, Tensor, float]]:
    r"""Normalize linear constraints to the unit cube.

    Args:
        bounds (Tensor): A `2 x d`-dim tensor containing the box bounds.
        constraints (List[Tuple[Tensor, Tensor, float]]): A list of
            tuples (indices, coefficients, rhs), with each tuple encoding
            an inequality constraint of the form
            `\sum_i (X[indices[i]] * coefficients[i]) >= rhs` or
            `\sum_i (X[indices[i]] * coefficients[i]) = rhs`.
    """

    new_constraints = []
    for index, coefficient, rhs in constraints:
        lower, upper = bounds[:, index]
        s = upper - lower
        new_constraints.append(
            (index, s * coefficient, (rhs - torch.dot(coefficient, lower)).item())
        )
    return new_constraints


def get_polytope_samples(
    n: int,
    bounds: Tensor,
    inequality_constraints: Optional[List[Tuple[Tensor, Tensor, float]]] = None,
    equality_constraints: Optional[List[Tuple[Tensor, Tensor, float]]] = None,
    seed: Optional[int] = None,
    thinning: int = 32,
    n_burnin: int = 10_000,
) -> Tensor:
    r"""Sample from polytope defined by box bounds and (in)equality constraints.

    This uses a hit-and-run Markov chain sampler.

    TODO: make this method return the sampler object, to avoid doing burn-in
    every time we draw samples.

    Args:
        n: The number of samples.
        bounds: A `2 x d`-dim tensor containing the box bounds.
        inequality constraints: A list of tuples (indices, coefficients, rhs),
            with each tuple encoding an inequality constraint of the form
            `\sum_i (X[indices[i]] * coefficients[i]) >= rhs`.
        equality constraints: A list of tuples (indices, coefficients, rhs),
            with each tuple encoding an inequality constraint of the form
            `\sum_i (X[indices[i]] * coefficients[i]) = rhs`.
        seed: The random seed.
        thinning: The amount of thinning.
        n_burnin: The number of burn-in samples for the Markov chain sampler.

    Returns:
        A `n x d`-dim tensor of samples.
    """
    # create tensors representing linear inequality constraints
    # of the form Ax >= b.
    # TODO: remove this error handling functionality in a few releases.
    # Context: BoTorch inadvertently supported indices with unusual dtypes.
    # This is now not supported.
    index_dtype_error = (
        "Normalizing {var_name} failed. Check that the first "
        "element of {var_name} is the correct dtype following "
        "the previous IndexError."
    )
    if inequality_constraints:
        # normalize_linear_constraints is called to solve this issue:
        # https://github.com/pytorch/botorch/issues/1225
        try:
            # non-standard dtypes used to be supported for indices in constraints;
            # this is no longer true
            constraints = normalize_linear_constraints(bounds, inequality_constraints)
        except IndexError as e:
            msg = index_dtype_error.format(var_name="`inequality_constraints`")
            raise ValueError(msg) from e

        A, b = sparse_to_dense_constraints(
            d=bounds.shape[-1],
            constraints=constraints,
        )
        # Note the inequality constraints are of the form Ax >= b,
        # but PolytopeSampler expects inequality constraints of the
        # form Ax <= b, so we multiply by -1 below.
        dense_inequality_constraints = -A, -b
    else:
        dense_inequality_constraints = None
    if equality_constraints:
        try:
            # non-standard dtypes used to be supported for indices in constraints;
            # this is no longer true
            constraints = normalize_linear_constraints(bounds, equality_constraints)
        except IndexError as e:
            msg = index_dtype_error.format(var_name="`equality_constraints`")
            raise ValueError(msg) from e

        # normalize_linear_constraints is called to solve this issue:
        # https://github.com/pytorch/botorch/issues/1225
        dense_equality_constraints = sparse_to_dense_constraints(
            d=bounds.shape[-1], constraints=constraints
        )
    else:
        dense_equality_constraints = None
    normalized_bounds = torch.zeros_like(bounds)
    normalized_bounds[1, :] = 1.0
    polytope_sampler = HitAndRunPolytopeSampler(
        bounds=normalized_bounds,
        inequality_constraints=dense_inequality_constraints,
        equality_constraints=dense_equality_constraints,
        n_burnin=n_burnin,
    )
    samples = polytope_sampler.draw(n=n * thinning, seed=seed)[::thinning]
    return bounds[0] + samples * (bounds[1] - bounds[0])


def sparse_to_dense_constraints(
    d: int,
    constraints: List[Tuple[Tensor, Tensor, float]],
) -> Tuple[Tensor, Tensor]:
    r"""Convert parameter constraints from a sparse format into a dense format.

    This method converts sparse triples of the form (indices, coefficients, rhs)
    to constraints of the form Ax >= b or Ax = b.

    Args:
        d: The input dimension.
        inequality constraints: A list of tuples (indices, coefficients, rhs),
            with each tuple encoding an (in)equality constraint of the form
            `\sum_i (X[indices[i]] * coefficients[i]) >= rhs` or
            `\sum_i (X[indices[i]] * coefficients[i]) = rhs`.

    Returns:
        A two-element tuple containing:
            - A: A `n_constraints x d`-dim tensor of coefficients.
            - b: A `n_constraints x 1`-dim tensor of right hand sides.
    """
    _t = constraints[0][1]
    A = torch.zeros(len(constraints), d, dtype=_t.dtype, device=_t.device)
    b = torch.zeros(len(constraints), 1, dtype=_t.dtype, device=_t.device)
    for i, (indices, coefficients, rhs) in enumerate(constraints):
        A[i, indices.long()] = coefficients
        b[i] = rhs
    return A, b
